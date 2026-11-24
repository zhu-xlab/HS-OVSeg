"""Test any retained multilevel checkpoint against independent pixel ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ovrs_segmentation.inference.tiling import image_tiles, minimum_size

from ovrs_segmentation.datasets import SemanticSegmentationDataset
from ovrs_segmentation.datasets.class_prompts import combined_prompt_texts
from .multilevel_validation import _forward_prompt_chunks, _reduce
from .sam3_common import (
    RankShardSampler, close_distributed, deterministic_subset,
    distributed_context, validate_evaluation_args,
)
from .sam3_multilevel import load_adapted_model
from .presence import PresenceMetrics
from ovrs_segmentation.test_catalog import load_suite, SCOPES


def single_sample(samples):
    return samples[0]


def segmentation_metrics(confusion: torch.Tensor, names: list[str], *, background: bool):
    matrix = confusion.double()
    tp = matrix.diag()
    target, predicted = matrix.sum(1), matrix.sum(0)
    union = target + predicted - tp
    included = union.gt(0)
    if background:
        included[-1] = False
    iou = tp / union.clamp_min(1)
    dice = 2 * tp / (target + predicted).clamp_min(1)
    return {
        "mean_iou": float(iou[included].mean()) if included.any() else None,
        "mean_dice": float(dice[included].mean()) if included.any() else None,
        "pixel_accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else None,
        "valid_pixels": int(matrix.sum()),
        "per_class": [
            {"class_name": name, "iou": float(iou[i]) if union[i] else None,
             "dice": float(dice[i]) if union[i] else None,
             "target_pixels": int(target[i]), "predicted_pixels": int(predicted[i])}
            for i, name in enumerate(names)
        ],
    }


@torch.inference_mode()
def tiled_pixel_predictions(model, image, prompts, device, precision, *, background, presence_scores=None, tile_stride=None):
    """Cover each original pixel once, padding partial tiles without resizing."""
    height, width = image.shape[-2:]
    if tile_stride is not None:
        from ovrs_segmentation.inference.overlap import overlapping_vote
        from ovrs_segmentation.inference.tiling import upsample_small_image
        from torch.nn import functional as F
        tile_size = model.sam3_input_size[0]
        if tuple(model.sam3_input_size) != (tile_size, tile_size):
            raise ValueError('Overlapping inference requires square tiles')
        def predict(tile):
            batch = {'images': tile, 'prompt_texts': prompts,
                     'pixel_targets': torch.empty(0, tile_size, tile_size)}
            output = _forward_prompt_chunks(model, batch, device, precision)
            logits = output['unified_pixel_logits']
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite overlapping pixel logits')
            if presence_scores is not None and 'concept_logits' in output:
                presence_scores.append(output['concept_logits'].reshape(-1).sigmoid().cpu().numpy())
            best, labels = logits.max(0)
            if background:
                labels[best.lt(0)] = len(prompts)
            return labels
        prediction = overlapping_vote(upsample_small_image(image, model.sam3_input_size),
            tile_size, tile_stride, len(prompts) + int(background), predict)
        return F.interpolate(prediction[None, None].float(), size=(height, width),
                             mode='nearest')[0, 0].long().to(device)
    if minimum_size((height, width), model.sam3_input_size) != (height, width):
        batch = {"images": image, "prompt_texts": prompts, "pixel_targets": torch.empty(0, height, width)}
        output = _forward_prompt_chunks(model, batch, device, precision)
        logits = output["unified_pixel_logits"]
        if presence_scores is not None and "concept_logits" in output:
            presence_scores.append(output["concept_logits"].reshape(-1).sigmoid().cpu().numpy())
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite unified logits for upsampled image")
        best, prediction = logits.max(0)
        if background:
            prediction[best.lt(0)] = len(prompts)
        return prediction
    prediction = torch.empty((height, width), device=device, dtype=torch.long)
    for top, left, tile_height, tile_width, tile in image_tiles(image, model.sam3_input_size):
        batch = {"images": tile, "prompt_texts": prompts,
                 "pixel_targets": torch.empty((0, *model.sam3_input_size))}
        output = _forward_prompt_chunks(model, batch, device, precision)
        logits = output["unified_pixel_logits"]
        if presence_scores is not None and "concept_logits" in output:
            presence_scores.append(output["concept_logits"].reshape(-1).sigmoid().cpu().numpy())
        if not torch.isfinite(logits).all():
            raise ValueError(f"Non-finite unified logits in tile at ({left}, {top})")
        best, local_prediction = logits.max(0)
        if background:
            local_prediction[best.lt(0)] = len(prompts)
        prediction[top:top + tile_height, left:left + tile_width] = local_prediction[:tile_height, :tile_width]
    return prediction


@torch.inference_mode()
def evaluate_segmentation(model, loader, dataset, device, precision, tile_stride=None):
    classes = [
        (raw_id, spec) for raw_id, spec in sorted(dataset.class_specs_by_id.items())
        if raw_id not in {dataset.background_id, dataset.ignore_id}
        and not any(spec.get(key) for key in ("is_background", "is_ignore", "is_excluded"))
        and spec.get("queryable", True)
    ]
    if not classes:
        raise ValueError("Pixel ground truth has no queryable classes")
    prompts = [
        combined_prompt_texts(
            spec["class_text"], spec.get("semantic_level") or dataset.metadata.get("semantic_level", "lulc"),
            dataset.dataset_name,
        )["native_prompt_texts"][0]
        for _, spec in classes
    ]
    names = [spec["class_text"] for _, spec in classes]
    has_background = dataset.background_id is not None
    count = len(classes) + int(has_background)
    if has_background:
        names.append("background")
    confusion = torch.zeros((count, count), dtype=torch.int64, device=device)
    presence_classes = [(raw_id, spec["class_text"]) for raw_id, spec in classes]
    presence_ids = np.array([raw_id for raw_id, _ in classes] + ([dataset.background_id] if has_background else []))
    mask_presence = PresenceMetrics(presence_classes, source="segmentation_mask")
    native_presence = PresenceMetrics(presence_classes, source="native_head")
    for sample_index, sample in enumerate(loader, 1):
        dataset._validate_mask(sample["sample_id"], sample["class_id_mask"])
        image = torch.from_numpy(np.array(sample["image"], copy=True)).permute(2, 0, 1).float()[None] / 255
        mask = torch.as_tensor(np.array(sample["class_id_mask"], copy=True), device=device).long()
        target = torch.full_like(mask, -1)
        for local_id, (raw_id, _) in enumerate(classes):
            target[mask.eq(raw_id)] = local_id
        if has_background:
            target[mask.eq(dataset.background_id)] = len(classes)
        scores = []
        prediction = tiled_pixel_predictions(
            model, image, prompts, device, precision, background=has_background, presence_scores=scores,
            **({'tile_stride': tile_stride} if tile_stride is not None else {})
        )
        valid = target.ge(0) & mask.ne(dataset.ignore_id)
        target_np = mask.cpu().numpy()
        prediction_np = presence_ids[prediction.cpu().numpy()]
        valid_np = valid.cpu().numpy()
        mask_presence.update(target_np, prediction_np, valid_np)
        if scores:
            native_presence.update(target_np, prediction_np, valid_np, scores=np.max(scores, axis=0),
                                   sample_id=sample["sample_id"])
        confusion += torch.bincount(
            target[valid] * count + prediction[valid], minlength=count * count
        ).reshape(count, count)
        if sample_index == 1 or sample_index % 20 == 0:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[test rank={rank}] {dataset.dataset_name}: {sample_index} samples", flush=True)
    _reduce(confusion)
    result = segmentation_metrics(confusion, names, background=has_background)
    for key, accumulator in (("presence_mask", mask_presence), ("presence_native", native_presence)):
        if key == "presence_native" and torch.distributed.is_initialized():
            shards = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(shards, accumulator.score_rows)
            accumulator.score_rows = [row for shard in shards for row in shard]
        totals = torch.as_tensor(np.concatenate((accumulator.counts.ravel(), [accumulator.images, accumulator.exact])), device=device)
        _reduce(totals)
        totals = totals.cpu().numpy()
        accumulator.counts = totals[:-2].reshape(-1, 4)
        accumulator.images, accumulator.exact = map(int, totals[-2:])
        result[key] = accumulator.result() if accumulator.images and accumulator.images == mask_presence.images else None
    return result, confusion.cpu().tolist()


def main(*, model_loader=None, result_metadata=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset-root", type=Path)
    selection.add_argument("--suite", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/home/Datasets/RSOD"))
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--evaluation-scope", choices=SCOPES, default="all")
    parser.add_argument("--preflight", action="store_true", help="Validate all GT datasets without loading weights")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sam3-checkpoint", type=Path)
    source.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--sam3-bpe-path", type=Path)
    parser.add_argument("--sam3-input-size", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--prompt-chunk-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-stride", type=int, help="Opt-in overlapping inference with center-weighted label voting")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rank, world_size, device = distributed_context(args.device)
    try:
        entries = load_suite(args.suite, args.data_root, args.test_root, scope=args.evaluation_scope) if args.suite else [
            {"name": args.dataset_root.name, "root": args.dataset_root, "split": args.split}
        ]
        reports = []
        model = None
        for entry in entries:
            report = {"dataset": entry["name"], "source_task": entry.get("source_task"),
                      "dataset_root": str(entry["root"]), "split": entry.get("split", args.split)}
            try:
                dataset = SemanticSegmentationDataset(entry["root"], report["split"], validate_masks=False)
                if dataset.box_only:
                    raise ValueError("Segmentation test requires pixel ground truth, not boxes")
                subset = deterministic_subset(dataset, args.max_samples, args.seed)
                if not len(subset):
                    raise ValueError("Segmentation test split is empty")
            except (OSError, ValueError, KeyError) as error:
                reports.append({**report, "status": "unavailable_pixel_gt", "error": str(error)})
                if rank == 0:
                    print(f"[test] {entry['name']}: unavailable pixel GT: {error}")
                continue
            if args.preflight:
                reports.append({**report, "status": "ready", "samples": len(subset)})
                continue
            if model is None:
                model, state, settings, checkpoint = (model_loader or load_adapted_model)(args, device)
                validate_evaluation_args(args, settings["sam3_input_size"])
            loader = DataLoader(subset, batch_size=1, num_workers=args.workers,
                                sampler=RankShardSampler(len(subset), rank, world_size), collate_fn=single_sample)
            metrics, confusion = evaluate_segmentation(model, loader, dataset, device, args.precision, args.tile_stride)
            reports.append({**report, "status": "evaluated", "samples": len(subset),
                            "metrics": metrics, "confusion_matrix": confusion})
            if rank == 0:
                print(f"[test] {entry['name']}: mIoU={metrics['mean_iou']}, Dice={metrics['mean_dice']}", flush=True)
        result = {
            "evaluation_task": "segmentation", "prediction": "unified_pixel_logits",
            "trained_checkpoint": str(args.checkpoint.resolve()), "world_size": world_size,
            "preflight": args.preflight, "precision": args.precision,
            "metric_resolution": "original_ground_truth", "mean_includes_background": False,
            "inference_mode": "non_overlapping_tiles" if args.tile_stride is None else "overlap_center_weighted_label_vote", "tile_padding_value": 0.5,
            "small_image_policy": "aspect_ratio_preserving_upsample_to_minimum_tile_size",
            "background_rule": "all class logits < 0 when dataset defines background",
            "primary_metric": "mean_iou", "datasets": reports,
            "complete": all(row["status"] in {"ready", "evaluated"} for row in reports),
        }
        if model is not None:
            result.update(checkpoint_epoch=int(state["epoch"]) + 1,
                          checkpoint_global_step=int(state["global_step"]),
                          sam3_input_size=settings["sam3_input_size"],
                          tile_size=settings["sam3_input_size"], tile_stride=args.tile_stride or settings["sam3_input_size"])
            if result_metadata:
                result.update(result_metadata(state))
        if rank == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            print(f"[test] complete={result['complete']}; wrote {args.output_json}")
        if not result["complete"]:
            raise SystemExit(2)
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
