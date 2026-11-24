#!/usr/bin/env python
"""Run native SAM3 multi-level inference (scene/object/lulc) with batched text prompts."""
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from ovrs_segmentation.datasets import SemanticTextPromptDataset, open_text_prompt_dataset
from ovrs_segmentation.modeling.sam3_utils import (
    configure_sam3_vision_input_size,
    ensure_sam3_importable,
)
from .common import (
    autocast as _autocast,
    device as _device,
    empty_semantic_map as _empty_semantic_map,
    image_ids as _image_ids,
    load_gt_samples,
    load_queries,
    query_source_suffix as _query_source_suffix,
    select_images,
    semantic_background_id as _semantic_background_id,
    slug as _slug,
    usable_texts,
)
from ovrs_segmentation.visualization import (
    render_attention_overlay,
    render_semantic_overlay,
)


@dataclass(frozen=True)
class Query:
    class_id: int
    class_text: str
    prompt_texts: tuple[str, ...]
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="Image or directory; repeatable")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam3.pt"))
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--prompt", action="append", help="One noun phrase per class")
    query.add_argument("--classes-json", action="append", type=Path, help="Repeatable")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--gt-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-policy", choices=["native", "all"], default="native")
    parser.add_argument("--level", choices=["scene", "object", "lulc"], required=True)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--threshold", dest="confidence_threshold", type=float, default=0.5)
    parser.add_argument("--sam3-input-size", type=int, default=518)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_native"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overlay-alpha", type=float, default=0.50)
    parser.add_argument("--overlay-image-brightness", type=float, default=0.92)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overlay-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_queries(args: argparse.Namespace) -> list[Query]:
    return load_queries(
        args, lambda class_id, text, prompts: Query(class_id, text, prompts)
    )


def _load_gt_samples(
    dataset_root: Path, split: str, prompt_policy: str, level: str
) -> dict[Path, list[Query]]:
    return load_gt_samples(
        dataset_root,
        split,
        prompt_policy,
        level,
        lambda class_id, text, prompts: Query(class_id, text, prompts),
    )


def _load_gt_scene_samples(
    dataset_root: Path, split: str, prompt_policy: str
) -> dict[Path, list[Query]]:
    return _load_gt_samples(dataset_root, split, prompt_policy, "scene")


def _load_gt_object_samples(
    dataset_root: Path, split: str, prompt_policy: str
) -> dict[Path, list[Query]]:
    """Load positive Object prompts together with their native GT boxes."""
    dataset = open_text_prompt_dataset(
        dataset_root, split, validate_masks=False, level="object"
    )
    if isinstance(dataset, SemanticTextPromptDataset) and dataset.box_only:
        samples: dict[Path, list[Query]] = {}
        semantic = dataset.semantic
        for sample_id in semantic.sample_ids:
            boxes_by_class, _ = dataset._detection_geometry_for_sample(sample_id)
            queries = []
            for class_id, boxes in sorted(boxes_by_class.items()):
                spec = semantic.class_specs_by_id.get(class_id)
                if (
                    not boxes
                    or spec is None
                    or spec.get("queryable") is False
                    or dataset.scope_by_class.get(class_id) != "object"
                ):
                    continue
                class_text = semantic.class_text_by_id[class_id]
                queries.append(
                    Query(
                        class_id,
                        class_text,
                        usable_texts(
                            {**spec, "class_text": class_text},
                            prompt_policy,
                            semantic.dataset_name,
                        ),
                        tuple(tuple(map(float, box)) for box in boxes),
                    )
                )
            if queries:
                samples[semantic._images[sample_id].resolve()] = queries
        if not samples:
            raise ValueError(
                f"No GT Object boxes found below {dataset_root} for split={split}"
            )
        return samples

    samples: dict[Path, list[Query]] = {}
    for sample in dataset:
        grouped: dict[int, dict[str, Any]] = {}
        for annotation in sample["annotations"]:
            if (
                annotation["semantic_level"] != "object"
                or annotation["supervision_level"] != "box"
                or not annotation.get("present", True)
            ):
                continue
            boxes = annotation.get("boxes_xyxy", [])
            if not boxes:
                continue
            class_id = int(
                annotation.get(
                    "raw_dataset_class_id", annotation["dataset_class_id"]
                )
            )
            class_text = str(annotation["class_text"])
            row = grouped.setdefault(
                class_id,
                {
                    "class_text": class_text,
                    "prompt_texts": [],
                    "boxes_xyxy": [],
                },
            )
            row["prompt_texts"].extend(
                usable_texts(
                    {**annotation, "class_text": class_text},
                    prompt_policy,
                    str(sample["dataset_name"]),
                )
            )
            row["boxes_xyxy"].extend(boxes)
        if not grouped:
            continue
        image_path = Path(sample["image_path"]).resolve()
        samples[image_path] = [
            Query(
                class_id,
                row["class_text"],
                tuple(dict.fromkeys(row["prompt_texts"])),
                tuple(tuple(map(float, box)) for box in row["boxes_xyxy"]),
            )
            for class_id, row in sorted(grouped.items())
        ]
    if not samples:
        raise ValueError(
            f"No GT Object boxes found below {dataset_root} for split={split}"
        )
    return samples


def _validate_sam3_input_size(value: int) -> int:
    size = int(value)
    if size < 14 or size % 14:
        raise ValueError("--sam3-input-size must be a positive multiple of 14")
    return size


def _resolve_bpe_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    ensure_sam3_importable()
    import sam3

    package = Path(sam3.__file__).resolve().parent
    candidates = [
        package / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        package.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "SAM3 BPE vocabulary was not found. Pass --bpe-path explicitly; checked: "
        + ", ".join(map(str, candidates))
    )


def _build_processor(args: argparse.Namespace, device: torch.device):
    ensure_sam3_importable()
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as error:
        raise ImportError(
            "Official SAM3 is not importable. Run tools/install_sam3_local.py or use "
            "the Docker image containing third_party/sam3."
        ) from error

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    bpe_path = _resolve_bpe_path(args.bpe_path)
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        bpe_path=str(bpe_path),
        load_from_HF=False,
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
    )
    if args.sam3_input_size != 1008:
        configure_sam3_vision_input_size(
            model.backbone, (args.sam3_input_size, args.sam3_input_size)
        )
    model.eval()
    processor = Sam3Processor(
        model,
        resolution=args.sam3_input_size,
        device=str(device),
        # Preserve every native query candidate. The public threshold is
        # applied once, after score-weighted masks are merged into pixels.
        confidence_threshold=0.0,
    )
    # Batched prompt inference below calls model.forward_grounding directly and
    # consumes per-prompt native outputs, so no single-prompt output hook is needed.
    return model, processor, checkpoint, bpe_path


def _semantic_probabilities_from_native(
    outputs: dict[str, Any],
    count: int,
    height: int,
    width: int,
) -> np.ndarray:
    value = outputs["semantic_seg"]
    tensor = value.detach().float().reshape(count, *value.shape[-2:])
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor[:, None], size=(height, width), mode="bilinear", align_corners=False
        )[:, 0]
    return tensor.sigmoid().cpu().numpy().astype(np.float32, copy=False)


def _forward_prompt_batch(
    processor,
    state: dict[str, Any],
    prompts: Sequence[str],
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one native grounding pass for a batch of text prompts."""
    from sam3.model.data_misc import FindStage

    count = len(prompts)
    backbone_out = dict(state["backbone_out"])
    backbone_out.update(
        processor.model.backbone.forward_text(list(prompts), device=processor.device)
    )

    find_input = FindStage(
        img_ids=torch.zeros(count, dtype=torch.long, device=device),
        text_ids=torch.arange(count, dtype=torch.long, device=device),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    outputs = processor.model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        geometric_prompt=processor.model._get_dummy_prompt(num_prompts=count),
        find_target=None,
    )
    presence_logits = (
        outputs["presence_logit_dec"].detach().float().reshape(count, -1).amax(1)
    )
    presence_probabilities = presence_logits.sigmoid()
    query_scores = outputs["pred_logits"].detach().float().reshape(count, -1).sigmoid()
    value = outputs["pred_masks"]
    masks = (
        value.detach().float().reshape(count, query_scores.shape[1], *value.shape[-2:])
    )
    if tuple(masks.shape[-2:]) != (height, width):
        masks = torch.nn.functional.interpolate(
            masks, size=(height, width), mode="bilinear", align_corners=False
        )
    masks = masks.sigmoid()
    query_activation = masks.amax(dim=1)
    object_scores = query_scores.amax(dim=1)

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy().astype(np.float32, copy=False)

    return {
        "presence_logits": to_numpy(presence_logits),
        "presence_probabilities": to_numpy(presence_probabilities),
        "query_activation": to_numpy(query_activation),
        "object_scores": to_numpy(object_scores),
        "gated_object_scores": to_numpy(object_scores * presence_probabilities),
        "candidate_counts": np.full(count, query_scores.shape[1], dtype=np.int32),
        "semantic_probabilities": _semantic_probabilities_from_native(
            outputs, count, height, width
        ),
    }


def _soft_or_fuse_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fuse semantic and query probabilities with a differentiable OR."""
    return (1.0 - (1.0 - a) * (1.0 - b)).astype(np.float32, copy=False)


def _normalized_cxcywh(
    boxes_xyxy: Sequence[Sequence[float]], width: int, height: int
) -> torch.Tensor:
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
    boxes /= torch.tensor([width, height, width, height], dtype=torch.float32)
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=1
    ).clamp_(0.0, 1.0)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5),
        dim=-1,
    )


def _box_iou(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[..., :2], targets[:, None, :2])
    bottom_right = torch.minimum(boxes[..., 2:], targets[:, None, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    box_area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp_min(0).prod(-1)
    return intersection / (
        box_area + target_area[:, None] - intersection
    ).clamp_min(1e-6)


def _forward_box_prompt_batch(
    processor,
    state: dict[str, Any],
    prompts: Sequence[str],
    boxes: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Native teacher activation for paired class-text and GT-box prompts."""
    from sam3.model.data_misc import FindStage

    count = len(prompts)
    boxes = boxes.to(device=device, dtype=torch.float32)
    backbone_out = dict(state["backbone_out"])
    backbone_out.update(
        processor.model.backbone.forward_text(list(prompts), device=processor.device)
    )
    find_input = FindStage(
        img_ids=torch.zeros(count, dtype=torch.long, device=device),
        text_ids=torch.arange(count, dtype=torch.long, device=device),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = processor.model._get_dummy_prompt(num_prompts=count)
    geometric_prompt.append_boxes(
        boxes[None], torch.ones(1, count, device=device, dtype=torch.bool)
    )
    outputs = processor.model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )
    query_scores = outputs["pred_logits"].detach().float().reshape(count, -1).sigmoid()
    predicted_boxes = outputs["pred_boxes"].detach().float().reshape(
        count, query_scores.shape[1], 4
    )
    quality = _box_iou(_cxcywh_to_xyxy(predicted_boxes), _cxcywh_to_xyxy(boxes))
    selected_queries = (quality + query_scores).argmax(1)
    rows = torch.arange(count, device=device)
    masks = outputs["pred_masks"].detach().float().reshape(
        count, query_scores.shape[1], *outputs["pred_masks"].shape[-2:]
    )
    selected_masks = masks[rows, selected_queries, None]
    if tuple(selected_masks.shape[-2:]) != (height, width):
        selected_masks = torch.nn.functional.interpolate(
            selected_masks,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    presence_logits = outputs["presence_logit_dec"].detach().float().reshape(count, -1).amax(1)
    selected_scores = query_scores[rows, selected_queries]

    def numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy().astype(np.float32, copy=False)

    return {
        "presence_logits": numpy(presence_logits),
        "presence_probabilities": numpy(presence_logits.sigmoid()),
        "selected_query_scores": numpy(selected_scores),
        "selected_query_probabilities": numpy(selected_masks[:, 0].sigmoid()),
        "semantic_probabilities": _semantic_probabilities_from_native(
            outputs, count, height, width
        ),
        "candidate_counts": np.full(count, query_scores.shape[1], dtype=np.int32),
    }


def _predict_object_teacher_image(
    processor,
    image: Image.Image,
    queries: Sequence[Query],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, np.ndarray]]:
    """Reproduce the training teacher's text+bbox pixel pseudo-label path."""
    width, height = image.size
    activations = {
        query.class_id: np.zeros((height, width), dtype=np.float32)
        for query in queries
    }
    statistics = {
        query.class_id: {
            "presence_logit": -float("inf"),
            "presence_probability": 0.0,
            "object_score": 0.0,
            "candidate_count": 0,
            "best_prompt": None,
        }
        for query in queries
    }
    prompts, class_ids, boxes = [], [], []
    for query in queries:
        normalized = _normalized_cxcywh(query.boxes_xyxy, width, height)
        for prompt in query.prompt_texts:
            for box in normalized:
                prompts.append(prompt)
                class_ids.append(query.class_id)
                boxes.append(box)

    with torch.inference_mode(), _autocast(device, args.precision):
        state = processor.set_image(image)
        for start in range(0, len(prompts), args.prompt_batch_size):
            stop = min(start + args.prompt_batch_size, len(prompts))
            batch = _forward_box_prompt_batch(
                processor,
                state,
                prompts[start:stop],
                torch.stack(boxes[start:stop]),
                height,
                width,
                device,
            )
            for index, (class_id, prompt) in enumerate(
                zip(class_ids[start:stop], prompts[start:stop])
            ):
                activation = _soft_or_fuse_numpy(
                    batch["semantic_probabilities"][index],
                    batch["selected_query_probabilities"][index],
                )
                activations[class_id] = np.maximum(activations[class_id], activation)
                row = statistics[class_id]
                presence_logit = float(batch["presence_logits"][index])
                if presence_logit > row["presence_logit"]:
                    row["presence_logit"] = presence_logit
                    row["presence_probability"] = float(
                        batch["presence_probabilities"][index]
                    )
                    row["best_prompt"] = prompt
                row["object_score"] = max(
                    row["object_score"],
                    float(batch["selected_query_scores"][index]),
                )
                row["candidate_count"] += int(batch["candidate_counts"][index])

    records = []
    for query in queries:
        activation = activations[query.class_id]
        row = statistics[query.class_id]
        records.append(
            {
                "class_id": query.class_id,
                "class_text": query.class_text,
                "prompt_texts": list(query.prompt_texts),
                "best_prompt_text": row["best_prompt"],
                "present": bool((activation > args.confidence_threshold).any()),
                "box_prompt_count": len(query.boxes_xyxy),
                "query_candidate_count": row["candidate_count"],
                "presence_logit": row["presence_logit"],
                "presence_probability": row["presence_probability"],
                "object_score": row["object_score"],
                "gated_object_score": row["object_score"] * row["presence_probability"],
                "activation_source": "native_teacher_soft_or(semantic_seg,selected_query_mask[class_text+GT_bbox,IoU+score])",
                "activation_max": float(activation.max(initial=0.0)),
                "activation_mean": float(activation.mean()),
            }
        )

    stacked = np.stack([activations[query.class_id] for query in queries])
    winner, maximum = stacked.argmax(0), stacked.max(0)
    semantic = _empty_semantic_map(queries, (height, width))
    active = maximum > args.confidence_threshold
    semantic[active] = np.asarray(
        [query.class_id for query in queries], dtype=semantic.dtype
    )[winner[active]]
    return records, semantic, activations


def _predict_image(
    processor,
    image: Image.Image,
    queries: Sequence[Query],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray | None,
    dict[int, np.ndarray],
]:
    if args.level == "object" and any(query.boxes_xyxy for query in queries):
        return _predict_object_teacher_image(processor, image, queries, args, device)
    width, height = image.size
    class_activations = {
        query.class_id: np.zeros((height, width), dtype=np.float32) for query in queries
    }

    # One scalar decision score per class after prompt-variant aggregation.
    best_presence_logit = {query.class_id: -float("inf") for query in queries}
    best_presence_probability = {query.class_id: 0.0 for query in queries}
    best_object_score = {query.class_id: 0.0 for query in queries}
    best_gated_object_score = {query.class_id: 0.0 for query in queries}
    candidate_count_by_class = {query.class_id: 0 for query in queries}
    best_prompt_by_class: dict[int, str | None] = {
        query.class_id: None for query in queries
    }

    # Flatten prompt variants so one batch can span different classes.
    flat_prompts: list[str] = []
    flat_class_ids: list[int] = []
    for query in queries:
        for prompt in query.prompt_texts:
            flat_prompts.append(prompt)
            flat_class_ids.append(query.class_id)

    with torch.inference_mode(), _autocast(device, args.precision):
        state = processor.set_image(image)

        for start in range(0, len(flat_prompts), args.prompt_batch_size):
            stop = min(len(flat_prompts), start + args.prompt_batch_size)
            prompt_chunk = flat_prompts[start:stop]
            class_id_chunk = flat_class_ids[start:stop]

            batch = _forward_prompt_batch(
                processor,
                state,
                prompt_chunk,
                height,
                width,
                device,
            )

            for batch_index, (class_id, prompt) in enumerate(
                zip(class_id_chunk, prompt_chunk)
            ):
                presence_logit = float(batch["presence_logits"][batch_index])
                presence_probability = float(
                    batch["presence_probabilities"][batch_index]
                )
                object_score = float(batch["object_scores"][batch_index])
                gated_object_score = float(batch["gated_object_scores"][batch_index])
                candidate_count_by_class[class_id] += int(
                    batch["candidate_counts"][batch_index]
                )
                best_object_score[class_id] = max(
                    best_object_score[class_id], object_score
                )
                best_gated_object_score[class_id] = max(
                    best_gated_object_score[class_id], gated_object_score
                )
                semantic_probabilities = batch["semantic_probabilities"]
                assert semantic_probabilities is not None
                semantic_activation = semantic_probabilities[batch_index]
                fused_activation = (
                    _soft_or_fuse_numpy(
                        semantic_activation,
                        batch["query_activation"][batch_index],
                    )
                    if args.level == "object"
                    else semantic_activation
                )

                # Keep the strongest presence score for class-level scene decisions.
                if presence_logit > best_presence_logit[class_id]:
                    best_presence_logit[class_id] = presence_logit
                    best_presence_probability[class_id] = presence_probability
                    best_prompt_by_class[class_id] = prompt

                    # Scene selection remains presence-only and localization is
                    # taken exclusively from semantic_seg.
                    if args.level == "scene":
                        class_activations[class_id] = semantic_activation.copy()

                if args.level == "lulc":
                    class_activations[class_id] = np.maximum(
                        class_activations[class_id], semantic_activation
                    )
                elif args.level == "object":
                    class_activations[class_id] = np.maximum(
                        class_activations[class_id], fused_activation
                    )

    records: list[dict[str, Any]] = []

    if args.level == "scene":
        # Scene classification uses ONLY presence logits. No pixel/query score
        # participates in the class decision.
        winner_query = max(
            queries,
            key=lambda query: best_presence_logit[query.class_id],
        )
        winner_class_id = winner_query.class_id

        for query in queries:
            class_id = query.class_id
            activation = class_activations[class_id]
            records.append(
                {
                    "class_id": class_id,
                    "class_text": query.class_text,
                    "prompt_texts": list(query.prompt_texts),
                    "best_prompt_text": best_prompt_by_class[class_id],
                    "predicted": class_id == winner_class_id,
                    "presence_logit": float(best_presence_logit[class_id]),
                    "presence_probability": float(best_presence_probability[class_id]),
                    "activation_source": (
                        "sigmoid(semantic_seg) for the prompt variant with the "
                        "highest presence_logit"
                    ),
                    "activation_max": float(activation.max(initial=0.0)),
                    "activation_mean": float(activation.mean()),
                }
            )
        return records, None, class_activations

    for query in queries:
        class_id = query.class_id
        activation = class_activations[class_id]
        records.append(
            {
                "class_id": class_id,
                "class_text": query.class_text,
                "prompt_texts": list(query.prompt_texts),
                "present": bool((activation > args.confidence_threshold).any()),
                "query_candidate_count": candidate_count_by_class[class_id],
                "presence_logit": float(best_presence_logit[class_id]),
                "presence_probability": float(best_presence_probability[class_id]),
                "object_score": float(best_object_score[class_id]),
                "gated_object_score": float(best_gated_object_score[class_id]),
                "activation_source": (
                    "sigmoid(semantic_seg)"
                    if args.level == "lulc"
                    else "soft_or(sigmoid(semantic_seg),max_q(sigmoid(pred_masks_q)))"
                ),
                "activation_max": float(activation.max(initial=0.0)),
                "activation_mean": float(activation.mean()),
            }
        )

    # Object/LULC both expose a class-wise pixel map for optional mask saving.
    stacked = np.stack([class_activations[query.class_id] for query in queries])
    winner = stacked.argmax(axis=0)
    maximum = stacked.max(axis=0)
    semantic = _empty_semantic_map(queries, (height, width))
    active = maximum > args.confidence_threshold
    class_ids = np.asarray([query.class_id for query in queries], dtype=semantic.dtype)
    semantic[active] = class_ids[winner[active]]
    return records, semantic, class_activations


def _save_attention_heatmaps(
    output_dir: Path,
    subdir: str,
    image_id: str,
    image: np.ndarray,
    queries: Sequence[Query],
    records: list[dict[str, Any]],
    class_activations: dict[int, np.ndarray],
    args: argparse.Namespace,
    *,
    only_predicted_scene: bool = False,
) -> list[dict[str, Any]]:
    """Save per-class heatmaps, or only the predicted class for Scene."""
    heatmap_dir = output_dir / subdir
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    probability_dir = output_dir / f"{subdir}_probabilities"
    if args.save_probabilities and not args.overlay_only:
        probability_dir.mkdir(parents=True, exist_ok=True)

    record_by_id = {record["class_id"]: record for record in records}

    for query in queries:
        record = record_by_id[query.class_id]
        if only_predicted_scene and not bool(record.get("predicted", False)):
            continue

        activation = class_activations[query.class_id]
        name = f"{image_id}__{query.class_id}-{_slug(query.class_text)}"
        overlay_path = heatmap_dir / f"{name}.png"

        if args.level == "scene":
            score = float(record["presence_probability"])
        elif args.level == "object":
            score = float(record["gated_object_score"])
        else:
            score = float(record["activation_max"])

        rendered = render_attention_overlay(
            image,
            activation,
            query.class_text,
            class_id=query.class_id,
            score=score,
            alpha=args.overlay_alpha,
            image_brightness=args.overlay_image_brightness,
        )
        Image.fromarray(rendered, mode="RGB").save(overlay_path)
        record["activation_overlay"] = str(overlay_path.resolve())

        if args.save_probabilities and not args.overlay_only:
            probability_path = probability_dir / f"{name}.npy"
            np.save(
                probability_path,
                activation.astype(np.float16),
            )
            record["activation_probability"] = str(probability_path.resolve())

    return records


def _save_pixel_outputs(
    output_dir: Path,
    image_id: str,
    image: np.ndarray,
    queries: Sequence[Query],
    records: list[dict[str, Any]],
    semantic: np.ndarray,
    class_probabilities: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Save combined class map plus optional binary masks/probabilities."""
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay = render_semantic_overlay(
        image,
        semantic,
        {query.class_id: query.class_text for query in queries},
        alpha=args.overlay_alpha,
        image_brightness=args.overlay_image_brightness,
        background_id=_semantic_background_id(queries),
    )
    overlay_path = overlay_dir / f"{image_id}.png"
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    for record in records:
        record["overlay"] = str(overlay_path.resolve())

    if args.overlay_only:
        return records

    binary_dir = output_dir / "binary_masks"
    semantic_dir = output_dir / "semantic_masks"
    directories = [binary_dir, semantic_dir]
    probability_dir = output_dir / "probabilities"
    if args.save_probabilities:
        directories.append(probability_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    record_by_id = {record["class_id"]: record for record in records}
    for query in queries:
        record = record_by_id[query.class_id]
        probability = class_probabilities[query.class_id]
        binary = probability > args.confidence_threshold
        name = f"{image_id}__{query.class_id}-{_slug(query.class_text)}"
        binary_path = binary_dir / f"{name}.png"
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(binary_path)
        record["positive_pixels"] = int(binary.sum())
        record["binary_mask"] = str(binary_path.resolve())

        if args.save_probabilities:
            probability_path = probability_dir / f"{name}.npy"
            np.save(
                probability_path,
                probability.astype(np.float16),
            )
            record["pixel_probability"] = str(probability_path.resolve())

    semantic_path = semantic_dir / f"{image_id}.png"
    Image.fromarray(semantic).save(semantic_path)
    for record in records:
        record["semantic_mask"] = str(semantic_path.resolve())

    return records


def _save_level_outputs(
    output_dir: Path,
    image_id: str,
    image: np.ndarray,
    queries: Sequence[Query],
    records: list[dict[str, Any]],
    semantic: np.ndarray | None,
    class_activations: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.level == "scene":
        return _save_attention_heatmaps(
            output_dir,
            "scene_heatmaps",
            image_id,
            image,
            queries,
            records,
            class_activations,
            args,
            only_predicted_scene=not getattr(args, "gt_only", False),
        )

    if args.level == "object":
        if not args.overlay_only:
            records = _save_attention_heatmaps(
                output_dir,
                "object_heatmaps",
                image_id,
                image,
                queries,
                records,
                class_activations,
                args,
            )
        if semantic is None:
            raise RuntimeError("Object inference did not return a pixel map")
        return _save_pixel_outputs(
            output_dir,
            image_id,
            image,
            queries,
            records,
            semantic,
            class_activations,
            args,
        )

    # Overlay-only means one final cross-class argmax visualization per image.
    if not args.overlay_only:
        records = _save_attention_heatmaps(
            output_dir,
            "lulc_heatmaps",
            image_id,
            image,
            queries,
            records,
            class_activations,
            args,
        )
    if semantic is None:
        raise RuntimeError("LULC inference did not return a pixel map")
    return _save_pixel_outputs(
        output_dir,
        image_id,
        image,
        queries,
        records,
        semantic,
        class_activations,
        args,
    )


def main() -> None:
    args = parse_args()
    # GT-only Scene uses positive class labels; GT-only Object additionally uses
    # native boxes to reproduce the online teacher pseudo-mask mechanism.
    args.gt_only = bool(
        args.gt_only
        and args.level in {"scene", "object"}
        and args.dataset_root is not None
    )
    args.sam3_input_size = _validate_sam3_input_size(args.sam3_input_size)

    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.prompt_batch_size < 1:
        raise ValueError("--prompt-batch-size must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")
    if not 0.0 <= args.overlay_image_brightness <= 1.0:
        raise ValueError("--overlay-image-brightness must be between 0 and 1")

    images = select_images(args)

    if args.shuffle:
        random.Random(args.seed).shuffle(images)

    queries = _load_queries(args)
    gt_samples = None
    if args.gt_only:
        if args.level == "object":
            print(f"[infer] reading GT Object boxes split={args.split}...", flush=True)
        gt_samples = (
            _load_gt_scene_samples(args.dataset_root, args.split, args.prompt_policy)
            if args.level == "scene"
            else _load_gt_object_samples(
                args.dataset_root, args.split, args.prompt_policy
            )
        )
        if args.level == "object":
            box_count = sum(
                len(query.boxes_xyxy)
                for image_queries in gt_samples.values()
                for query in image_queries
            )
            print(
                f"[infer] GT Object boxes ready images={len(gt_samples)} "
                f"boxes={box_count}",
                flush=True,
            )
    device = _device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = args.output_dir.expanduser().resolve()
    predictions_path = output_dir / "predictions.jsonl"
    if not args.overlay_only and predictions_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {predictions_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, processor, checkpoint, bpe_path = _build_processor(args, device)
    print(
        f"[infer] loaded native SAM3 checkpoint={checkpoint} bpe={bpe_path} "
        f"device={device} level={args.level}",
        flush=True,
    )

    ids = _image_ids(images)
    query_source_suffix = _query_source_suffix(args)
    stream_context = (
        nullcontext(None)
        if args.overlay_only
        else predictions_path.open("w", encoding="utf-8")
    )

    with stream_context as stream:
        for index, image_path in enumerate(images, 1):
            image_queries = (
                gt_samples[image_path.resolve()] if gt_samples is not None else queries
            )
            with Image.open(image_path) as source:
                pil_image = source.convert("RGB")
                image = np.asarray(pil_image)

                records, semantic, class_activations = _predict_image(
                    processor,
                    pil_image,
                    image_queries,
                    args,
                    device,
                )

            result_image_id = f"{ids[image_path]}__{args.level}__{query_source_suffix}"
            records = _save_level_outputs(
                output_dir,
                result_image_id,
                image,
                image_queries,
                records,
                semantic,
                class_activations,
                args,
            )

            row = {
                "image_id": ids[image_path],
                "result_id": result_image_id,
                "query_source_suffix": query_source_suffix,
                "image": str(image_path),
                "checkpoint": str(checkpoint),
                "level": args.level,
                "decision": (
                    "argmax_presence_logit"
                    if args.level == "scene"
                    else (
                        (
                            "native_teacher_class_text_gt_bbox_pixel_activation"
                            if args.gt_only
                            else "presence_gated_score_weighted_query_object_activation"
                        )
                        if args.level == "object"
                        else "semantic_probability_then_per_pixel_argmax"
                    )
                ),
                "gt_only": args.gt_only,
                "threshold": (
                    None if args.level == "scene" else args.confidence_threshold
                ),
                "sam3_input_size": args.sam3_input_size,
                "prompt_batch_size": args.prompt_batch_size,
                "predictions": records,
            }
            if stream is not None:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"\r[infer] {index}/{len(images)} images",
                end="",
                flush=True,
            )

    print()
    print(
        f"[infer] complete images={len(images)} level={args.level} "
        f"prompt_batch_size={args.prompt_batch_size} "
        f"predictions={'disabled (overlay-only)' if args.overlay_only else predictions_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
