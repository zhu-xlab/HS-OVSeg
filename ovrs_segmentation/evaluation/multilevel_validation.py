from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ovrs_segmentation.modeling.unified_pixel import quality_weighted_query_scores
from ovrs_segmentation.training.prompt_pairs import expand_dataset_local_prompt_pairs
from ovrs_segmentation.inference.tiling import image_tiles, upsample_small_image


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype)


def _mask_logits_to_boxes(mask_logits: torch.Tensor) -> torch.Tensor:
    """Tight normalized boxes derived only from predicted Query masks."""
    foreground = mask_logits.float().gt(0)
    row_active = foreground.any(-1)
    column_active = foreground.any(-2)
    height, width = foreground.shape[-2:]
    x = torch.arange(width, device=foreground.device)
    y = torch.arange(height, device=foreground.device)
    x_min = torch.where(column_active, x, width).amin(-1).clamp_max(width)
    y_min = torch.where(row_active, y, height).amin(-1).clamp_max(height)
    x_max = torch.where(column_active, x + 1, 0).amax(-1)
    y_max = torch.where(row_active, y + 1, 0).amax(-1)
    boxes = torch.stack(
        (x_min / width, y_min / height, x_max / width, y_max / height), dim=-1
    )
    return torch.where(foreground.flatten(2).any(2)[..., None], boxes, 0.0)


@torch.inference_mode()
def _forward_single_tile(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    precision: str,
    *,
    include_semantic: bool = True,
) -> dict[str, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    if tuple(images.shape[-2:]) != model.sam3_input_size:
        raise ValueError("Tile input must match model input size; resizing is disabled")
    images = (images - model.image_mean) / model.image_std
    owners = batch.get("prompt_image_indices")
    if owners is None:
        if len(images) != 1:
            raise ValueError("Multiple images require explicit prompt_image_indices")
    else:
        owners = torch.as_tensor(owners, device=device, dtype=torch.long)
        if owners.shape != (len(batch["prompt_texts"]),) or torch.any((owners < 0) | (owners >= len(images))):
            raise ValueError("Every prompt must reference a valid image")
    output_size = tuple(batch["pixel_targets"].shape[-2:])

    with _autocast(device, precision):
        backbone = model.backbone.forward_image(images)
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    chunk_size = batch.get("prompt_chunk_size", model.prompt_chunk_size)
    prompts = batch["prompt_texts"]
    for start in range(0, len(prompts), chunk_size):
        text = prompts[start : start + chunk_size]
        indices = (torch.zeros(len(text), device=device, dtype=torch.long) if owners is None
                   else owners[start:start + len(text)])
        with _autocast(device, precision):
            text_features = model.backbone.forward_text(text, device=device)
            native = model._decode_grounding(
                backbone, text_features, indices, decoder_grad=False
            )
            query, presence = model._prompt_outputs(native, len(text))
            query_confidence = quality_weighted_query_scores(
                native["pred_masks"], query
            )
            boxes = _mask_logits_to_boxes(native["pred_masks"])
            if include_semantic:
                semantic = model._native_semantic_logits(native, len(text))
                if tuple(semantic.shape[-2:]) != output_size:
                    semantic = F.interpolate(
                        semantic[:, None].float(),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                if hasattr(model, "pixel_router"):
                    unified = model.pixel_router(
                        semantic, query, native["pred_masks"], output_size=output_size,
                    )
                else:
                    unified = {"unified_pixel_logits": semantic}
        values = (
            ("concept_logits", presence),
            ("query_confidence", query_confidence),
            ("boxes_xyxy", boxes),
            ("detection_scores", query_confidence * presence.reshape(-1, 1).sigmoid()),
        )
        if include_semantic:
            values += (("unified_pixel_logits", unified["unified_pixel_logits"]),)
        for name, value in values:
            chunks[name].append(value.detach().float())
    return {name: torch.cat(values) for name, values in chunks.items()}


@torch.inference_mode()
def _forward_prompt_chunks(model, batch, device, precision, *, include_semantic=True):
    """Run native-scale windows; stitch pixels and map detections to the full image."""
    images = batch["images"]
    original_size = tuple(images.shape[-2:])
    if original_size == model.sam3_input_size:
        return _forward_single_tile(model, batch, device, precision, include_semantic=include_semantic)
    images = upsample_small_image(images, model.sam3_input_size)
    height, width = images.shape[-2:]
    chunks = defaultdict(list)
    unified = None
    for top, left, valid_height, valid_width, tile in image_tiles(images, model.sam3_input_size):
        output = _forward_single_tile(
            model, {**batch, "images": tile, "pixel_targets": torch.empty(0, *model.sam3_input_size)},
            device, precision, include_semantic=include_semantic,
        )
        if include_semantic:
            pixels = output["unified_pixel_logits"]
            if unified is None:
                unified = pixels.new_empty((len(pixels), height, width))
            unified[:, top:top + valid_height, left:left + valid_width] = pixels[:, :valid_height, :valid_width]
        boxes = output["boxes_xyxy"].clone()
        tile_height, tile_width = model.sam3_input_size
        boxes *= boxes.new_tensor([tile_width, tile_height, tile_width, tile_height])
        boxes[..., 0::2].clamp_(0, valid_width)
        boxes[..., 1::2].clamp_(0, valid_height)
        visible = (boxes[..., 2:] > boxes[..., :2]).all(-1)
        output["detection_scores"] *= visible
        boxes += boxes.new_tensor([left, top, left, top])
        boxes /= boxes.new_tensor([width, height, width, height])
        chunks["boxes_xyxy"].append(boxes)
        for key in ("query_confidence", "detection_scores", "concept_logits"):
            chunks[key].append(output[key])
    result = {key: torch.cat(values, dim=1) for key, values in chunks.items() if key != "concept_logits"}
    result["concept_logits"] = torch.stack(chunks["concept_logits"]).amax(0)
    if include_semantic:
        if (height, width) != original_size:
            unified = F.interpolate(unified[:, None], size=original_size, mode="bilinear", align_corners=False)[:, 0]
        result["unified_pixel_logits"] = unified
    return result


def _reduce(value: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)


@torch.inference_mode()
def evaluate_scene(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    precision: str,
    class_count: int,
) -> dict[str, float]:
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    class_correct = torch.zeros(class_count, device=device, dtype=torch.float64)
    class_total = torch.zeros_like(class_correct)
    for batch in loader:
        batch = expand_dataset_local_prompt_pairs(
            batch, "scene", augment_prompts=False
        )
        logits = _forward_prompt_chunks(model, batch, device, precision, include_semantic=False)[
            "concept_logits"
        ].reshape(-1)
        target = int(batch["multiclass_scene_targets"][0])
        ranking = logits.argsort(descending=True)
        rank = int(ranking.eq(target).nonzero()[0, 0]) + 1
        correct = float(rank == 1)
        totals[0] += correct
        totals[1] += rank <= min(5, class_count)
        totals[2] += 1 / rank
        totals[3] += logits[target].sigmoid()
        totals[4] += 1
        class_correct[target] += correct
        class_total[target] += 1
    _reduce(totals)
    _reduce(class_correct)
    _reduce(class_total)
    observed = class_total > 0
    count = max(float(totals[4]), 1.0)
    return {
        "scene_full_vocab_top1_accuracy": float(totals[0] / count),
        "scene_full_vocab_top5_accuracy": float(totals[1] / count),
        "scene_mean_class_accuracy": float(
            (class_correct[observed] / class_total[observed]).mean()
        ),
        "scene_mean_reciprocal_rank": float(totals[2] / count),
        "scene_mean_gt_sigmoid_probability": float(totals[3] / count),
        "scene_num_classes": int(class_count),
        "scene_samples": int(totals[4]),
    }


@torch.inference_mode()
def evaluate_lulc(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    precision: str,
    class_count: int,
) -> dict[str, float]:
    intersection = torch.zeros(class_count, device=device, dtype=torch.float64)
    union = torch.zeros_like(intersection)
    pixel_totals = torch.zeros(2, device=device, dtype=torch.float64)
    for batch in loader:
        batch = expand_dataset_local_prompt_pairs(
            batch, "lulc", augment_prompts=False
        )
        logits = _forward_prompt_chunks(model, batch, device, precision)[
            "unified_pixel_logits"
        ]
        binary_target = batch["pixel_targets"].to(device=device).eq(1)
        valid = batch["valid_mask"].to(device=device, dtype=torch.bool)
        best_logit, prediction = logits.max(0)
        predicted = prediction.unsqueeze(0).eq(
            torch.arange(class_count, device=device)[:, None, None]
        ) & best_logit.ge(0)
        intersection += (predicted & binary_target & valid).flatten(1).sum(1)
        union += ((predicted | binary_target) & valid).flatten(1).sum(1)

        target = batch["multiclass_pixel_targets"][0].to(
            device=device, dtype=torch.long
        )
        pixel_valid = valid.any(0)
        target_is_foreground = target.ge(0) & target.lt(class_count)
        correct = (target_is_foreground & prediction.eq(target) & best_logit.ge(0)) | (
            ~target_is_foreground & best_logit.lt(0)
        )
        pixel_totals[0] += (correct & pixel_valid).sum()
        pixel_totals[1] += pixel_valid.sum()

    _reduce(intersection)
    _reduce(union)
    _reduce(pixel_totals)
    observed = union > 0
    return {
        "lulc_mean_iou": float((intersection[observed] / union[observed]).mean()),
        "lulc_pixel_accuracy": float(pixel_totals[0] / pixel_totals[1].clamp_min(1)),
        "lulc_valid_classes": int(observed.sum()),
    }


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    intersection = (
        (torch.minimum(box[2:], boxes[:, 2:]) - torch.maximum(box[:2], boxes[:, :2]))
        .clamp_min(0)
        .prod(1)
    )
    area = (box[2:] - box[:2]).clamp_min(0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    return intersection / (area + areas - intersection).clamp_min(1e-8)


def _nms(
    boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    order, keep = scores.argsort(descending=True), []
    while len(order) and len(keep) < 100:
        current = order[0]
        keep.append(current)
        remaining = order[1:]
        order = remaining[_box_iou(boxes[current], boxes[remaining]) <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def _expand_object_validation_prompts(batch: dict[str, Any]) -> dict[str, Any]:
    classes = batch["dataset_vocabularies"][0]["object"]["classes"]
    positive = dict(zip(batch["class_ids"].tolist(), batch["boxes_xyxy"]))
    count, (height, width) = len(classes), batch["images"].shape[-2:]
    return {
        **batch,
        "prompt_texts": [str(row["native_prompt_texts"][0]) for row in classes],
        "semantic_levels": ["object"] * count,
        "annotation_sample_indices": torch.zeros(count, dtype=torch.long),
        "class_ids": torch.tensor([int(row["local_class_id"]) for row in classes]),
        "boxes_xyxy": [
            positive.get(int(row["local_class_id"]), torch.empty(0, 4))
            for row in classes
        ],
        "pixel_targets": torch.full((count, height, width), 255, dtype=torch.uint8),
    }


def detection_ap50(
    predictions: list[tuple[int, str, float, list[float]]],
    ground_truth: list[tuple[int, str, list[list[float]]]],
) -> tuple[float, float]:
    """Return class-mean AP50 and maximum recall."""
    gt: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    pred: dict[int, list[tuple[str, float, torch.Tensor]]] = defaultdict(list)
    for class_id, image_id, boxes in ground_truth:
        gt[class_id][image_id] = torch.tensor(boxes).reshape(-1, 4)
    for class_id, image_id, score, box in predictions:
        pred[class_id].append((image_id, score, torch.tensor(box)))

    aps, recalls = [], []
    for class_id, image_boxes in gt.items():
        count = sum(len(boxes) for boxes in image_boxes.values())
        if not count:
            continue
        matched = {
            key: torch.zeros(len(boxes), dtype=torch.bool)
            for key, boxes in image_boxes.items()
        }
        tp, fp = [], []
        for image_id, _, box in sorted(
            pred[class_id], key=lambda row: row[1], reverse=True
        ):
            boxes = image_boxes.get(image_id)
            if boxes is None or not len(boxes):
                tp.append(0.0)
                fp.append(1.0)
                continue
            ious = _box_iou(box, boxes)
            best = int(ious.argmax())
            hit = float(ious[best]) >= 0.5 and not matched[image_id][best]
            tp.append(float(hit))
            fp.append(float(not hit))
            matched[image_id][best] |= hit
        if not tp:
            aps.append(0.0)
            recalls.append(0.0)
            continue
        cumulative_tp = torch.tensor(tp).cumsum(0)
        cumulative_fp = torch.tensor(fp).cumsum(0)
        recall = cumulative_tp / count
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = torch.cat((torch.zeros(1), recall, torch.ones(1)))
        precision = torch.cat((torch.zeros(1), precision, torch.zeros(1)))
        for index in range(len(precision) - 2, -1, -1):
            precision[index] = torch.maximum(precision[index], precision[index + 1])
        changes = recall[1:] != recall[:-1]
        aps.append(
            float(((recall[1:] - recall[:-1])[changes] * precision[1:][changes]).sum())
        )
        recalls.append(float(recall[-2]))
    return (
        sum(aps) / len(aps) if aps else 0.0,
        sum(recalls) / len(recalls) if recalls else 0.0,
    )


@torch.inference_mode()
def evaluate_object(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    predictions, ground_truth = [], []
    for batch in loader:
        batch = _expand_object_validation_prompts(batch)
        outputs = _forward_prompt_chunks(
            model, batch, device, precision, include_semantic=False
        )
        scores = outputs["detection_scores"]
        boxes = outputs["boxes_xyxy"].clamp(0, 1)
        sample_ids = [
            batch["sample_ids"][index] for index in batch["annotation_sample_indices"]
        ]
        for row, (class_id, image_id, target) in enumerate(
            zip(batch["class_ids"].tolist(), sample_ids, batch["boxes_xyxy"])
        ):
            for query in _nms(boxes[row], scores[row])[:100]:
                predictions.append(
                    (
                        class_id,
                        str(image_id),
                        float(scores[row, query]),
                        boxes[row, query].tolist(),
                    )
                )
            ground_truth.append((class_id, str(image_id), target.tolist()))

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    gathered_predictions, gathered_gt = [None] * world_size, [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered_predictions, predictions)
        dist.all_gather_object(gathered_gt, ground_truth)
    else:
        gathered_predictions[0], gathered_gt[0] = predictions, ground_truth
    predictions = [row for rows in gathered_predictions for row in rows]
    ground_truth = [row for rows in gathered_gt for row in rows]
    mean_ap, mean_recall = detection_ap50(predictions, ground_truth)
    return {
        "object_map50": mean_ap,
        "object_mean_recall50": mean_recall,
        "object_ground_truth_boxes": sum(len(row[2]) for row in ground_truth),
    }
