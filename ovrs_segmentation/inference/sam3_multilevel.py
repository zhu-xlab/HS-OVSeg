#!/usr/bin/env python
"""Run level-free inference with a trained SAM3 multilevel model."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from ovrs_segmentation.modeling import SAM3NativeMultilevelModel
from .tiling import image_tiles, upsample_small_image
from .common import (
    autocast as _autocast,
    device as _device,
    empty_semantic_map as _empty_semantic_map,
    image_ids as _image_ids,
    load_queries,
    query_source_suffix as _query_source_suffix,
    select_images,
    semantic_background_id as _semantic_background_id,
    slug as _slug,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="Image or directory; repeatable")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained last.pt or best.pt")
    parser.add_argument("--sam3-checkpoint", type=Path, help="Base sam3.pt")
    parser.add_argument("--sam3-bpe-path", type=Path)
    parser.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_inference"))
    parser.add_argument("--prompt", action="append", help="One noun phrase per class")
    parser.add_argument("--classes-json", action="append", type=Path, help="classes.json; repeatable")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--prompt-policy", choices=["native", "all"], default="native")
    parser.add_argument("--prompt-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--sam3-input-size", type=int)
    parser.add_argument("--tile-overlap", type=int, default=148,
                        help="Overlap in pixels (default: 148; 0 disables overlap)")
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overlay-alpha", type=float, default=0.62)
    parser.add_argument("--overlay-image-brightness", type=float, default=0.92)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overlay-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_queries(args: argparse.Namespace) -> list[Query]:
    return load_queries(
        args,
        lambda class_id, text, prompts: Query(class_id, text, prompts),
    )


def _load_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[SAM3NativeMultilevelModel, dict[str, Any]]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must use the current multilevel state format")
    training_args = state["args"]
    base_checkpoint = args.sam3_checkpoint or training_args.get("sam3_checkpoint")
    bpe_path = args.sam3_bpe_path or training_args.get("sam3_bpe_path")
    load_from_hf = bool(args.load_from_hf or training_args.get("load_from_hf", False))
    if args.load_from_hf:
        base_checkpoint = None
    if base_checkpoint is not None:
        base_checkpoint = Path(base_checkpoint).expanduser()
    if bpe_path is not None:
        bpe_path = Path(bpe_path).expanduser()
    common = {
        "bpe_path": None if bpe_path is None else str(bpe_path),
        "load_from_hf": load_from_hf,
        "device": device,
        "sam3_input_size": int(
            args.sam3_input_size or training_args.get("sam3_input_size", 518)
        ),
    }
    common.update(
        {
            "lora_rank": int(args.lora_rank or training_args["lora_rank"]),
            "lora_scale": float(args.lora_scale or training_args["lora_scale"]),
        }
    )
    model = SAM3NativeMultilevelModel.from_pretrained(
        None if base_checkpoint is None else str(base_checkpoint), **common
    )
    model.load_trainable_state_dict(state["model"])
    model.eval()
    return model, state


def _flatten_queries(queries: Sequence[Query]):
    prompts, owners = [], []
    for query_index, query in enumerate(queries):
        for prompt in query.prompt_texts:
            prompts.append(prompt)
            owners.append(query_index)
    return prompts, owners


def _require_finite_output(
    name: str,
    value: torch.Tensor,
    prompts: Sequence[str],
) -> None:
    """Fail before thresholding or saving masks from a corrupted checkpoint."""
    if value.numel() == 0:
        return
    minimum, maximum = torch.aminmax(value.detach().float())
    if bool(torch.isfinite(minimum) and torch.isfinite(maximum)):
        return
    raise FloatingPointError(
        f"Inference produced non-finite {name} for prompts {list(prompts)!r}. "
        "Do not use a checkpoint saved after training first reported NaN."
    )


def _dense_probabilities(
    value: torch.Tensor,
    count: int,
    height: int,
    width: int,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize a per-prompt dense tensor to B x H x W probabilities."""
    tensor = value.float()
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    elif tensor.ndim == 2 and count == 1:
        tensor = tensor[None]
    if tensor.ndim != 3 or tensor.shape[0] != count:
        raise ValueError(f"Unexpected {name} shape: {tuple(value.shape)}")
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor[:, None], size=(height, width), mode="bilinear", align_corners=False
        )[:, 0]
    return tensor.sigmoid()


def _unified_activation_from_outputs(
    outputs: dict[str, Any], count: int, height: int, width: int
) -> torch.Tensor:
    """Read the model's learned level-free concept map."""
    return _dense_probabilities(
        outputs["unified_pixel_logits"],
        count,
        height,
        width,
        name="unified_pixel_logits",
    )


def _joint_probability_semantic_map(
    queries: Sequence[Query],
    activations: np.ndarray,
    concept_probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select pixels using presence probability times unified pixel probability."""
    concept_active = concept_probabilities > threshold
    joint_probabilities = activations * concept_probabilities[:, None, None]
    winner = joint_probabilities.argmax(axis=0)
    maximum = joint_probabilities.max(axis=0)
    semantic = _empty_semantic_map(queries, activations.shape[1:])
    active = maximum > threshold
    class_ids = np.asarray([query.class_id for query in queries], dtype=semantic.dtype)
    semantic[active] = class_ids[winner[active]]
    return semantic, concept_active


@torch.inference_mode()
def _predict_tile(
    model: SAM3NativeMultilevelModel,
    image: np.ndarray,
    queries: Sequence[Query],
    *,
    device: torch.device,
    precision: str,
    prompt_batch_size: int,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-class unified concept logits, probabilities, and masks."""
    image_tensor = (
        torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div_(255.0)
    )
    image_tensor = image_tensor.to(device)
    prompts, owners = _flatten_queries(queries)
    height, width = image.shape[:2]

    prompt_concept_logits: list[torch.Tensor] = []
    prompt_activations: list[torch.Tensor] = []

    with _autocast(device, precision):
        for start in range(0, len(prompts), prompt_batch_size):
            stop = min(len(prompts), start + prompt_batch_size)
            prompt_chunk = prompts[start:stop]
            count = len(prompt_chunk)
            outputs = model(
                {
                    "images": image_tensor,
                    "valid_image_masks": torch.ones(
                        (1, height, width), dtype=torch.bool, device=device
                    ),
                    "prompt_texts": prompt_chunk,
                    "annotation_sample_indices": torch.zeros(
                        count, dtype=torch.long, device=device
                    ),
                    "pixel_targets": torch.full(
                        (count, height, width), 255, dtype=torch.uint8, device=device
                    ),
                }
            )

            decision_logits = outputs["concept_logits"]
            decision_logits = decision_logits.float().reshape(count, -1).amax(dim=1)
            _require_finite_output("concept_logits", decision_logits, prompt_chunk)

            activation = _unified_activation_from_outputs(outputs, count, height, width)

            _require_finite_output("unified_activation", activation, prompt_chunk)
            prompt_concept_logits.append(decision_logits.detach().cpu())
            prompt_activations.append(activation.detach().float().cpu())

    all_concept_logits = torch.cat(prompt_concept_logits).numpy()
    all_activations = torch.cat(prompt_activations).numpy()

    grouped_logits: list[float] = []
    grouped_activations: list[np.ndarray] = []
    for query_index in range(len(queries)):
        indices = np.asarray(
            [index for index, owner in enumerate(owners) if owner == query_index],
            dtype=np.int64,
        )
        class_logits = all_concept_logits[indices]
        class_activations = all_activations[indices]

        if aggregation == "max":
            grouped_logits.append(float(class_logits.max()))
            grouped_activations.append(class_activations.max(axis=0))
        else:
            grouped_logits.append(float(class_logits.mean()))
            grouped_activations.append(class_activations.mean(axis=0))

    concept_logits = np.asarray(grouped_logits, dtype=np.float32)
    concept_probabilities = (
        torch.from_numpy(concept_logits)
        .sigmoid()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return concept_logits, concept_probabilities, np.stack(grouped_activations)


@torch.inference_mode()
def _predict(model, image, queries, *, device, precision, prompt_batch_size, aggregation, overlap=148):
    """Blend overlapping native-scale windows, then restore the original grid."""
    tile_height, tile_width = model.sam3_input_size
    if not 0 <= overlap < min(tile_height, tile_width):
        raise ValueError("--tile-overlap must be nonnegative and smaller than the model input size")
    stride = (tile_height - overlap, tile_width - overlap)
    images = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div_(255)
    original_size = image.shape[:2]
    images = upsample_small_image(images, model.sam3_input_size)
    height, width = images.shape[-2:]
    concept_logits = None
    activations = np.zeros((len(queries), height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    weights = (np.maximum(np.outer(np.hanning(tile_height), np.hanning(tile_width)), 1e-3)
               .astype(np.float32) if overlap else np.ones(model.sam3_input_size, dtype=np.float32))
    for top, left, valid_height, valid_width, tile in image_tiles(images, model.sam3_input_size, stride=stride):
        tile_image = tile[0].permute(1, 2, 0).numpy() * 255
        logits, _, pixels = _predict_tile(
            model, tile_image, queries, device=device, precision=precision,
            prompt_batch_size=prompt_batch_size, aggregation=aggregation,
        )
        concept_logits = logits if concept_logits is None else np.maximum(concept_logits, logits)
        valid_weights = weights[:valid_height, :valid_width]
        activations[:, top:top + valid_height, left:left + valid_width] += (
            pixels[:, :valid_height, :valid_width] * valid_weights)
        weight_sum[top:top + valid_height, left:left + valid_width] += valid_weights
    activations /= weight_sum[None]
    probabilities = torch.from_numpy(concept_logits).sigmoid().numpy()
    if (height, width) != original_size:
        activations = torch.nn.functional.interpolate(
            torch.from_numpy(activations)[None], size=original_size, mode="bilinear", align_corners=False
        )[0].numpy()
    return concept_logits, probabilities, activations


def _save_unified_heatmaps(
    output_dir: Path,
    image_id: str,
    image: np.ndarray,
    queries: Sequence[Query],
    activations: np.ndarray,
    scores: np.ndarray,
    subdir: str,
    args: argparse.Namespace,
    *,
    normalize_activation: bool = True,
) -> list[dict[str, Any]]:
    """Build per-class activation records and optionally save heatmap images.

    ``--overlay-only`` now means exactly that: only the final image-level overlay
    is written. Per-class scene/object/LULC heatmaps and activation probability
    files are skipped.
    """
    save_heatmaps = not args.overlay_only

    heatmap_dir = output_dir / subdir
    probability_dir = output_dir / f"{subdir}_probabilities"
    if save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        if args.save_probabilities:
            probability_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for query, activation, score in zip(queries, activations, scores):
        record: dict[str, Any] = {
            "class_id": query.class_id,
            "class_text": query.class_text,
            "prompt_texts": list(query.prompt_texts),
            "score": float(score),
            "activation_min": float(activation.min()),
            "activation_mean": float(activation.mean()),
            "activation_max": float(activation.max()),
        }

        if save_heatmaps:
            name = f"{image_id}__{query.class_id}-{_slug(query.class_text)}"
            path = heatmap_dir / f"{name}.png"
            rendered = render_attention_overlay(
                image,
                activation,
                query.class_text,
                class_id=query.class_id,
                score=float(score),
                alpha=args.overlay_alpha,
                image_brightness=args.overlay_image_brightness,
                normalize=normalize_activation,
            )
            Image.fromarray(rendered, mode="RGB").save(path)
            record["activation_overlay"] = str(path.resolve())

            if args.save_probabilities:
                probability_path = probability_dir / f"{name}.npy"
                np.save(probability_path, activation.astype(np.float16))
                record["activation_probability"] = str(probability_path.resolve())

        records.append(record)
    return records


def _save_unified_outputs(
    output_dir: Path,
    image_id: str,
    image: np.ndarray,
    queries: Sequence[Query],
    concept_logits: np.ndarray,
    concept_probabilities: np.ndarray,
    activations: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Save the same fused per-concept output for every text prompt."""
    records = _save_unified_heatmaps(
        output_dir,
        image_id,
        image,
        queries,
        activations,
        concept_probabilities,
        "unified_heatmaps",
        args,
    )

    semantic, concept_active = _joint_probability_semantic_map(
        queries,
        activations,
        concept_probabilities,
        threshold=args.threshold,
    )

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlays_dir / f"{image_id}.png"
    overlay = render_semantic_overlay(
        image,
        semantic,
        {query.class_id: query.class_text for query in queries},
        alpha=args.overlay_alpha,
        image_brightness=args.overlay_image_brightness,
        background_id=_semantic_background_id(queries),
    )
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    semantic_path: Path | None = None
    if not args.overlay_only:
        semantic_dir = output_dir / "semantic_masks"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        semantic_path = semantic_dir / f"{image_id}.png"
        Image.fromarray(semantic).save(semantic_path)

    for (
        query,
        activation,
        concept_logit,
        concept_probability,
        is_present,
        record,
    ) in zip(
        queries,
        activations,
        concept_logits,
        concept_probabilities,
        concept_active,
        records,
    ):
        binary = (activation * concept_probability) > args.threshold
        record.update(
            {
                "concept_logit": float(concept_logit),
                "concept_probability": float(concept_probability),
                "present": bool(is_present),
                "positive_pixels": int(binary.sum()),
                "overlay": str(overlay_path.resolve()),
                "semantic_mask": (
                    None if semantic_path is None else str(semantic_path.resolve())
                ),
            }
        )
        if not args.overlay_only:
            masks_dir = output_dir / "binary_masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            name = f"{image_id}__{query.class_id}-{_slug(query.class_text)}"
            binary_path = masks_dir / f"{name}.png"
            Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(binary_path)
            record["binary_mask"] = str(binary_path.resolve())
    return records


def main(*, model_loader=None, result_metadata=None) -> None:
    args = parse_args()
    if args.prompt and args.classes_json is not None:
        raise ValueError("Use either --prompt or --classes-json, not both")
    explicit_queries = bool(args.prompt) or args.classes_json is not None
    if not explicit_queries:
        raise ValueError("Inference requires --prompt or --classes-json")
    if args.overlay_only:
        args.save_probabilities = False
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.prompt_batch_size < 1:
        raise ValueError("--prompt-batch-size must be positive")
    if args.tile_overlap < 0:
        raise ValueError("--tile-overlap must be nonnegative")
    if args.lora_rank is not None and args.lora_rank < 1:
        raise ValueError("--lora-rank must be positive")
    if args.lora_scale is not None and args.lora_scale <= 0:
        raise ValueError("--lora-scale must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")
    if not 0.0 <= args.overlay_image_brightness <= 1.0:
        raise ValueError("--overlay-image-brightness must be between 0 and 1")

    images = select_images(args)

    if args.shuffle:
        random.Random(args.seed).shuffle(images)
    queries = _load_queries(args)
    device = _device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    predictions_path = None if args.overlay_only else output_dir / "predictions.jsonl"
    if (
        predictions_path is not None
        and predictions_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(f"Output exists; pass --overwrite: {predictions_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = (model_loader or _load_model)(args, device)
    extra_metadata = result_metadata(checkpoint) if result_metadata else {}
    ids = _image_ids(images)
    query_source_suffix = _query_source_suffix(args)
    processed_queries = 0
    stream = (
        None
        if predictions_path is None
        else predictions_path.open("w", encoding="utf-8")
    )

    try:
        for index, image_path in enumerate(images, start=1):
            image_queries = queries
            processed_queries += len(image_queries)
            with Image.open(image_path) as source:
                image = np.asarray(source.convert("RGB"))

            result_image_id = f"{ids[image_path]}__unified__{query_source_suffix}"
            concept_logits, concept_probabilities, activations = _predict(
                model,
                image,
                image_queries,
                device=device,
                precision=args.precision,
                prompt_batch_size=args.prompt_batch_size,
                aggregation=args.prompt_aggregation,
                overlap=args.tile_overlap,
            )

            predictions = _save_unified_outputs(
                output_dir,
                result_image_id,
                image,
                image_queries,
                concept_logits,
                concept_probabilities,
                activations,
                args,
            )

            row = {
                "image_id": ids[image_path],
                "result_id": result_image_id,
                "query_source_suffix": query_source_suffix,
                "image": str(image_path),
                "output_type": "level_free_unified",
                "activation_source": "adaptive_semantic_query_router",
                "semantic_decision_source": "presence_probability_x_unified_pixel_probability",
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_global_step": checkpoint.get("global_step"),
                "threshold": args.threshold,
                "prompt_batch_size": args.prompt_batch_size,
                "prompt_aggregation": args.prompt_aggregation,
                "tile_size": list(model.sam3_input_size),
                "tile_overlap": args.tile_overlap,
                "tile_stride": [size - args.tile_overlap for size in model.sam3_input_size],
                "tile_blending": "hann_weighted_mean" if args.tile_overlap else "none",
                "predictions": predictions,
                **extra_metadata,
            }
            if stream is not None:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"\r[infer] {index}/{len(images)} images",
                end="",
                flush=True,
            )

    finally:
        if stream is not None:
            stream.close()

    print()
    result_path = (
        "disabled (overlay-only)" if predictions_path is None else predictions_path
    )
    print(
        f"[infer] complete images={len(images)} queries={processed_queries} "
        f"mode=level-free predictions={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
