"""Shared dataset, distributed, and reporting code for SAM3 evaluation CLIs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from ovrs_segmentation.datasets import open_text_prompt_dataset, text_prompt_collate
from ovrs_segmentation.evaluation import evaluate_lulc, evaluate_object, evaluate_scene


LEVELS = ("scene", "object", "lulc")
METRIC_NAMES = {
    "scene": "scene_full_vocab_top1_accuracy",
    "object": "object_map50",
    "lulc": "lulc_mean_iou",
}


def add_evaluation_args(
    parser: argparse.ArgumentParser, *, input_size_default: int | None
) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--sam3-bpe-path", type=Path)
    parser.add_argument("--sam3-input-size", type=int, default=input_size_default)
    parser.add_argument("--prompt-chunk-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validate-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json", type=Path)


class LevelDataset(Dataset):
    """Expose one level and its dataset-local vocabulary."""

    def __init__(self, dataset: Dataset, level: str) -> None:
        self.dataset, self.level = dataset, level

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample["dataset_vocabulary"] = sample["dataset_vocabularies"][self.level]
        sample["annotations"] = [
            row
            for row in sample["annotations"]
            if str(row.get("semantic_level")) == self.level
        ]
        sample["level"] = self.level
        return sample


class RankShardSampler(Sampler[int]):
    """Shard evaluation without padding or duplicate samples."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def distributed_context(choice: str, *, timeout: timedelta | None = None) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = choice == "cuda" or (choice == "auto" and torch.cuda.is_available())
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if world_size > 1:
        if cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if cuda else "gloo",
            device_id=device if cuda else None,
            timeout=timeout,
        )
    return rank, world_size, device


def close_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_file(path: Path | None, name: str) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def deterministic_subset(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("--max-samples must be positive")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    return Subset(dataset, sorted(order[:maximum].tolist()))


def primary_metric(level: str, metrics: dict[str, float]) -> dict[str, Any]:
    name = METRIC_NAMES[level]
    return {"name": name, "value": float(metrics[name])}


def validate_evaluation_args(args: argparse.Namespace, input_size: int) -> None:
    if input_size < 14 or input_size % 14:
        raise ValueError("--sam3-input-size must be a positive multiple of 14")
    if args.prompt_chunk_size < 1 or args.workers < 0:
        raise ValueError("prompt chunk size must be positive and workers non-negative")


def run_evaluation(
    args: argparse.Namespace,
    model: torch.nn.Module,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    input_size: int,
    output: Path,
    model_fields: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one level and write the common JSON report."""
    dataset_root = args.dataset_root.expanduser().resolve()
    raw = open_text_prompt_dataset(
        dataset_root,
        split=args.split,
        validate_masks=args.validate_masks,
        level=args.level,
    )
    dataset = LevelDataset(
        deterministic_subset(raw, args.max_samples, args.seed), args.level
    )
    first = dataset[0]
    vocabulary = first["dataset_vocabulary"]
    class_count = int(vocabulary["num_classes"])
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RankShardSampler(len(dataset), rank, world_size),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            text_prompt_collate, prompt_strategy="first", spatial_size=None
        ),
    )

    if rank == 0:
        print(
            f"[eval] {args.level} {args.split}: {len(dataset)} samples, {class_count} classes"
        )
    started = time.perf_counter()
    evaluators = {
        "scene": evaluate_scene,
        "object": evaluate_object,
        "lulc": evaluate_lulc,
    }
    kwargs = {} if args.level == "object" else {"class_count": class_count}
    metrics = evaluators[args.level](model, loader, device, args.precision, **kwargs)
    result = {
        **model_fields,
        "dataset_root": str(dataset_root),
        "dataset_name": str(first["dataset_name"]),
        "split": args.split,
        "level": args.level,
        "samples": len(dataset),
        "num_classes": class_count,
        "class_names": list(vocabulary["class_names"]),
        "sam3_input_size": input_size,
        "inference_mode": "native_non_overlapping_tiles",
        "small_image_policy": "aspect_ratio_preserving_upsample_to_minimum_tile_size",
        "tile_size": input_size,
        "prompt_chunk_size": args.prompt_chunk_size,
        "precision": args.precision,
        "world_size": world_size,
        "elapsed_seconds": time.perf_counter() - started,
        "primary_metric": primary_metric(args.level, metrics),
        "metrics": metrics,
    }
    if rank == 0:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        metric = result["primary_metric"]
        print(f"[eval] {metric['name']}={metric['value']:.6f}; wrote {output}")
    return result
