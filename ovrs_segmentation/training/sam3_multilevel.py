#!/usr/bin/env python
"""Train scene + LULC + object text prompts with one SAM3-based model."""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset

from ovrs_segmentation.datasets import open_text_prompt_dataset, text_prompt_collate
from ovrs_segmentation.evaluation import evaluate_lulc, evaluate_object, evaluate_scene
from ovrs_segmentation.evaluation.sam3_common import (
    RankShardSampler,
    deterministic_subset as _deterministic_subset,
)
from ovrs_segmentation.modeling import SAM3NativeMultilevelModel
from ovrs_segmentation.training import (
    MultiGranularityCriterion,
    MultiGranularityLossWeights,
)
from ovrs_segmentation.training.prompt_pairs import (
    expand_dataset_local_prompt_pairs as _expand_dataset_local_prompt_pairs,
    expand_object_prompt_pairs as _expand_object_prompt_pairs,
)


LEVEL_RATIOS = {"scene": 1.0, "lulc": 1.0, "object": 1.0}
LOG_LOSS_NAMES = (
    "loss_total", "loss_scene_presence", "loss_object_presence", "loss_lulc_presence",
    "loss_unified_pixel_bce", "loss_unified_pixel_dice",
    "loss_unified_pixel_multiclass_ce", "loss_unified_total",
)

PRIMARY_VALIDATION_METRIC = {
    "scene": "scene_full_vocab_top1_accuracy",
    "object": "object_map50",
    "lulc": "lulc_mean_iou",
}


def _polynomial_lr_multiplier(step: int, total_steps: int, power: float) -> float:
    """Decay from 1 at step zero to 0 with a polynomial schedule."""
    if step < 0 or total_steps < 1 or power <= 0:
        raise ValueError(
            "step must be non-negative; total_steps and power must be positive"
        )
    progress = min(step / total_steps, 1.0)
    return (1.0 - progress) ** power


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sam3_multilevel.yaml"), help="Scene–Object–LULC training configuration")
    for level in LEVEL_RATIOS:
        parser.add_argument(f"--{level}-root", action="append", type=Path, default=[], help=f"Prepared {level} dataset root; repeat for multiple datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-bpe-path", type=Path)
    parser.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_multilevel"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/sam3_multilevel"), help="Directory for weights; logs remain in --output-dir")
    parser.add_argument("--checkpoint-policy", choices=("each_epoch", "final", "none"), default="each_epoch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU image batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--sam3-input-size", type=int, default=518, help="Square size divisible by 14")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--prompt-chunk-size", type=int, default=4, help="Compatibility option; ignored during training")
    parser.add_argument("--object-teacher-chunk-size", type=int, default=10, help="GT boxes per frozen Object-teacher forward")
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gradient-audit", action="store_true")
    parser.add_argument("--log-every", type=int, default=50, help="Print one compact progress line every N steps; 0 disables it")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--validation-every", type=int, default=3, help="Validate every N epochs and always on the final epoch")
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--val-max-samples", type=int)
    parser.add_argument("--val-log-every", type=int, default=100, help="Print validation progress every N local samples; 0 disables it")
    args = parser.parse_args()
    if args.validation_every < 1:
        parser.error("--validation-every must be positive")
    args.training_spatial_policy = "min518_scale_0.75_1.5_random_crop"
    args.validation_spatial_policy = "min518_native_non_overlapping_tiles"
    args.training_objective = "unified-head"
    args.router_strategy = "pixel"
    args.fusion_loss_only = True
    args.fusion_end_to_end = True
    args.object_semantic_normalization = True
    args.pixel_loss_protocol = "bce_dice_pixel_ce_v2"
    return args


def _should_validate(epoch: int, total_epochs: int, interval: int) -> bool:
    return epoch == total_epochs or epoch % interval == 0


def _load_multigranularity_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def _level_options(value: Any, default: int | float, cast) -> dict[str, Any]:
    value = value or {}
    return {level: cast(value.get(level, default)) for level in LEVEL_RATIOS}


def _configure_run(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    data = config.get("data", {})
    ratios = _level_options(data.get("task_sampling_ratio"), 1.0, float)
    prompt_limits = {
        "scene": int(data.get("scene_prompts_per_image", 8)),
        "object": int(data.get("object_prompts_per_image", 8)),
        "lulc": int(data.get("lulc_prompts_per_image", 8)),
        "object_instances": int(
            config.get("loss", {})
            .get("object", {})
            .get("max_instances_per_batch", 32)
        ),
    }
    configured_val_caps = _level_options(data.get("validation_max_samples"), 0, int)
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("task sampling ratios must be positive")
    if min(prompt_limits.values()) < 1:
        raise ValueError("prompt limits must be positive")
    LEVEL_RATIOS.update(ratios)

    if args.init_weights and args.resume:
        raise ValueError("Use either --init-weights or --resume, not both")
    positive = (
        args.epochs,
        args.batch_size,
        args.lora_rank,
        args.object_teacher_chunk_size,
    )
    if any(value < 1 for value in positive):
        raise ValueError("epochs, batch size, LoRA rank and teacher chunk size must be positive")
    if min(args.lora_scale, args.lr_lora) <= 0:
        raise ValueError("adapter scale and learning rates must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")
    if args.val_log_every < 0:
        raise ValueError("--val-log-every must be non-negative")
    if args.val_max_samples is not None and args.val_max_samples < 1:
        raise ValueError("--val-max-samples must be positive")
    if any(value < 0 for value in configured_val_caps.values()):
        raise ValueError("validation sample caps must be non-negative")
    args.validation_max_samples = {
        level: (
            args.val_max_samples
            if args.val_max_samples is not None
            else configured_val_caps[level] or None
        )
        for level in LEVEL_RATIOS
    }
    return prompt_limits, config.get("loss", {}).get("scene", {}), config.get("loss", {})


class EpochPromptCollator:
    def __init__(self, seed: int, spatial_size: int) -> None:
        self.seed = seed
        self.spatial_size = spatial_size
        self.epoch = 0

    def __call__(self, samples):
        return text_prompt_collate(
            samples,
            prompt_strategy="random",
            seed=self.seed + self.epoch,
            spatial_size=self.spatial_size,
            spatial_augmentation=True,
            scale_range=(0.75, 1.5),
        )


class ValidationPromptCollator:
    def __call__(self, samples):
        return text_prompt_collate(
            samples, prompt_strategy="first", spatial_size=None
        )


class TrainingStep(nn.Module):
    """Keep large dense predictions inside DDP and return losses only."""

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.criterion(self.model(batch), batch)


class LevelDataset(Dataset):
    """Attach the configured training level without changing prepared data."""

    def __init__(self, dataset: Dataset, level: str) -> None:
        self.dataset = dataset
        self.level = level

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample["level"] = self.level
        vocabulary = sample.get("dataset_vocabularies", {}).get(self.level)
        if vocabulary is None:
            raise ValueError(
                f"Sample {sample.get('sample_id')!r} has no {self.level} vocabulary"
            )
        sample["dataset_vocabulary"] = vocabulary
        return sample


class DistributedLevelBatchSampler(Sampler[int]):
    """Sample level/dataset-homogeneous global batches and shard across ranks.

    Every DDP rank receives the same semantic level at a given loader step.
    ``LEVEL_RATIOS`` selects the level; datasets within that level receive an
    equal number of batches regardless of their number of samples.
    """

    def __init__(
        self,
        levels: Sequence[str],
        dataset_groups: Sequence[int],
        num_samples: int,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        self.levels = tuple(levels)
        self.dataset_groups = tuple(dataset_groups)
        if len(self.dataset_groups) != len(self.levels):
            raise ValueError("levels and dataset_groups must have the same length")
        if num_samples < 1 or batch_size < 1 or world_size < 1:
            raise ValueError("num_samples, batch_size and world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be between zero and world_size - 1")
        unknown = set(self.levels) - set(LEVEL_RATIOS)
        if unknown:
            raise ValueError(f"Unknown semantic levels: {sorted(unknown)}")
        self.batch_size = int(batch_size)
        self.global_batch_size = self.batch_size * int(world_size)
        self.steps = int(math.ceil(num_samples / self.global_batch_size))
        self.num_samples = self.steps * self.batch_size
        self.total_size = self.steps * self.global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.indices_by_level = {
            level: [
                torch.tensor(
                    [
                        index
                        for index, (sample_level, sample_group) in enumerate(
                            zip(self.levels, self.dataset_groups)
                        )
                        if sample_level == level and sample_group == group
                    ],
                    dtype=torch.long,
                )
                for group in dict.fromkeys(
                    group
                    for sample_level, group in zip(
                        self.levels, self.dataset_groups
                    )
                    if sample_level == level
                )
            ]
            for level in LEVEL_RATIOS
            if level in self.levels
        }
        self.level_names = tuple(self.indices_by_level)
        self.level_weights = torch.tensor(
            [LEVEL_RATIOS[level] for level in self.level_names], dtype=torch.double
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected_levels = torch.multinomial(
            self.level_weights,
            self.steps,
            replacement=True,
            generator=generator,
        ).tolist()
        dataset_orders: dict[str, list[int]] = {}
        for level_index, level in enumerate(self.level_names):
            dataset_count = len(self.indices_by_level[level])
            level_steps = selected_levels.count(level_index)
            cycles = math.ceil(level_steps / dataset_count)
            dataset_orders[level] = [
                dataset_index
                for _ in range(cycles)
                for dataset_index in torch.randperm(
                    dataset_count, generator=generator
                ).tolist()
            ]
        dataset_offsets = Counter()
        local_indices: list[int] = []
        local_start = self.rank * self.batch_size
        local_stop = local_start + self.batch_size
        for level_index in selected_levels:
            level = self.level_names[level_index]
            dataset_index = dataset_orders[level][dataset_offsets[level]]
            dataset_offsets[level] += 1
            candidates = self.indices_by_level[level][dataset_index]
            sampled_positions = torch.randint(
                len(candidates),
                (self.global_batch_size,),
                generator=generator,
            )
            global_indices = candidates.index_select(0, sampled_positions)
            local_indices.extend(global_indices[local_start:local_stop].tolist())
        return iter(local_indices)

    def __len__(self) -> int:
        return self.num_samples


def _balanced_validation_subset(
    dataset: Dataset, level: str, maximum: int | None, seed: int
) -> Dataset:
    """Select a deterministic subset by repeatedly serving the least-covered class."""
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("validation sample caps must be positive")

    labels = [
        dataset.class_ids_for_index(index, level) for index in range(len(dataset))
    ]
    class_ids = sorted(set().union(*labels))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    buckets = {
        class_id: [index for index in order if class_id in labels[index]]
        for class_id in class_ids
    }
    selected: set[int] = set()
    coverage = Counter({class_id: 0 for class_id in class_ids})
    while len(selected) < maximum:
        progressed = False
        for class_id in sorted(class_ids, key=lambda value: (coverage[value], value)):
            candidate = next(
                (index for index in buckets[class_id] if index not in selected), None
            )
            if candidate is None:
                continue
            selected.add(candidate)
            coverage.update(labels[candidate])
            progressed = True
            if len(selected) == maximum:
                break
        if not progressed:
            break
    if len(selected) < maximum:
        for index in order:
            selected.add(index)
            if len(selected) == maximum:
                break
    return Subset(dataset, sorted(selected))


def _distributed_context(device_choice: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device_choice == "cuda" or (
        device_choice == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if world_size > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            device_id=device if use_cuda else None,
        )

        def cleanup_process_group() -> None:
            if dist.is_initialized():
                dist.destroy_process_group()

        atexit.register(cleanup_process_group)
    return rank, world_size, local_rank, device


def _build_datasets(args: argparse.Namespace):
    roots_by_level = {
        level: [path.expanduser().resolve() for path in getattr(args, f"{level}_root")]
        for level in LEVEL_RATIOS
    }
    if not any(roots_by_level.values()):
        raise ValueError(
            "Specify at least one --scene-root, --lulc-root or --object-root"
        )
    datasets, dataset_levels, rows = [], [], []
    for level, roots in roots_by_level.items():
        for root in roots:
            dataset = open_text_prompt_dataset(
                root, split=args.split, validate_masks=False, level=level
            )
            if not len(dataset):
                raise ValueError(f"Dataset split is empty: {root} ({args.split})")
            dataset = _deterministic_subset(dataset, args.train_max_samples, args.seed)
            datasets.append(LevelDataset(dataset, level))
            dataset_levels.append(level)
            rows.append({"level": level, "root": str(root), "samples": len(dataset)})
    sample_levels, sample_dataset_groups = [], []
    for dataset_group, (dataset, level) in enumerate(zip(datasets, dataset_levels)):
        sample_levels.extend([level] * len(dataset))
        sample_dataset_groups.extend([dataset_group] * len(dataset))
    return ConcatDataset(datasets), sample_levels, sample_dataset_groups, rows


def _build_validation_loaders(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    loaders = []
    collator = ValidationPromptCollator()
    for level in LEVEL_RATIOS:
        for raw_root in getattr(args, f"{level}_root"):
            root = raw_root.expanduser().resolve()
            dataset = open_text_prompt_dataset(
                root,
                split=args.validation_split,
                validate_masks=False,
                level=level,
            )
            if not len(dataset):
                raise ValueError(
                    f"Validation split {args.validation_split!r} is empty: {root}"
                )
            dataset = _balanced_validation_subset(
                dataset,
                level,
                args.validation_max_samples[level],
                args.seed,
            )
            wrapped = LevelDataset(dataset, level)
            sample = wrapped[0]
            class_count = int(sample["dataset_vocabulary"]["num_classes"])
            loaders.append(
                {
                    "level": level,
                    "root": str(root),
                    "class_count": class_count,
                    "loader": DataLoader(
                        wrapped,
                        batch_size=1,
                        sampler=RankShardSampler(len(wrapped), rank, world_size),
                        num_workers=args.workers,
                        pin_memory=device.type == "cuda",
                        collate_fn=collator,
                        persistent_workers=False,
                    ),
                }
            )
    return loaders


def _evaluate_validation(
    model: SAM3NativeMultilevelModel,
    validation_loaders: list[dict[str, Any]],
    device: torch.device,
    precision: str,
    rank: int,
    log_every: int,
) -> dict[str, dict[str, float]]:
    results = {}
    model.eval()
    evaluators = {
        "scene": evaluate_scene,
        "object": evaluate_object,
        "lulc": evaluate_lulc,
    }
    for spec in validation_loaders:
        level, root = spec["level"], spec["root"]
        loader = spec["loader"]
        local_total = len(loader)
        started = time.monotonic()
        if rank == 0:
            print(
                f"[val] level={level} dataset={Path(root).name} "
                f"local_samples={local_total} started",
                flush=True,
            )

        def progress_loader():
            for index, batch in enumerate(loader, 1):
                yield batch
                if (
                    rank == 0
                    and log_every
                    and (index % log_every == 0 or index == local_total)
                ):
                    print(
                        f"[val] level={level} dataset={Path(root).name} "
                        f"local_samples={index}/{local_total} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )

        kwargs = (
            {"class_count": spec["class_count"]} if level in {"scene", "lulc"} else {}
        )
        metrics = evaluators[level](
            model, progress_loader(), device, precision, **kwargs
        )
        results[f"{level}/{Path(root).name}"] = metrics
        if rank == 0:
            metric_name = PRIMARY_VALIDATION_METRIC[level]
            print(
                f"[val] level={level} dataset={Path(root).name} complete "
                f"{metric_name}={metrics[metric_name]:.4f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    return results


def _format_validation(results: dict[str, dict[str, float]]) -> str:
    values = []
    for dataset, metrics in results.items():
        level = dataset.split("/", 1)[0]
        name = PRIMARY_VALIDATION_METRIC[level]
        if name in metrics:
            values.append(f"{dataset}:{name}={metrics[name]:.4f}")
    return " ".join(values)


def _mean_validation_score(results: dict[str, dict[str, float]]) -> float | None:
    if not results:
        return None
    scores = [float(metrics[PRIMARY_VALIDATION_METRIC[dataset.split("/", 1)[0]]])
              for dataset, metrics in results.items()]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("All dataset validation scores must be finite")
    return sum(scores) / len(scores)


def _best_validation_state(state: dict) -> tuple[float | None, int | None]:
    if state.get("best_validation_score") is not None:
        return state["best_validation_score"], state.get("best_epoch")
    score = _mean_validation_score(state.get("validation", {}))
    return score, int(state["epoch"]) + 1 if score is not None else None


def _parameter_groups(model: SAM3NativeMultilevelModel, args: argparse.Namespace):
    parameters = model.parameter_groups()["trainable"]
    if not parameters:
        raise ValueError("No trainable parameter groups were created")
    return [{"params": parameters, "lr": args.lr_lora, "name": "lora"}]


def _build_criterion(
    loss: dict[str, Any],
    device: torch.device,
) -> MultiGranularityCriterion:
    shared = loss.get("segmentation", {})
    weights = MultiGranularityLossWeights(
        presence=float(loss.get("presence_bce_weight", 1.0)),
        semantic_bce=float(shared.get("bce_weight", 1.0)),
        semantic_dice=float(shared.get("dice_weight", 1.0)),
        multiclass_ce=float(shared.get("multiclass_ce_weight", 1.0)),
    )
    return MultiGranularityCriterion(
        weights=weights,
        ignore_id=int(loss.get("ignore_id", 255)),
    ).to(device)


def _autocast(device: torch.device, precision: str):
    enabled = precision != "fp32"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _gradient_group(name: str) -> str:
    is_lora = ".parametrizations." in name and not name.endswith(".original")
    if "segmentation_head.cross_attend_prompt" in name:
        return "semantic_prompt_lora" if is_lora else "semantic_prompt_native"
    if "segmentation_head.semantic_seg_head" in name:
        return "semantic_output_lora" if is_lora else "semantic_output_native"
    if "segmentation_head.pixel_decoder" in name:
        return "pixel_decoder"
    if (
        "segmentation_head.instance_seg_head" in name
        or "segmentation_head.mask_predictor" in name
    ):
        return "query_mask_lora" if is_lora else "query_mask_native"
    if "transformer.encoder" in name:
        return "image_text_fusion_lora" if is_lora else "transformer_encoder_native"
    if "transformer.decoder" in name:
        return "transformer_decoder_lora" if is_lora else "transformer_decoder_native"
    if ".backbone." in name:
        return "backbone_lora" if is_lora else "backbone_native"
    return "other_lora" if is_lora else "other_native"


def _gradient_topology_report(model: nn.Module) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = grouped.setdefault(
            _gradient_group(name),
            {
                "trainable_parameters": 0,
                "parameter_tensors": 0,
                "tensors_with_grad": 0,
                "tensors_with_nonzero_grad": 0,
                "grad_squared_l2": 0.0,
                "grad_abs_max": 0.0,
            },
        )
        group["trainable_parameters"] += parameter.numel()
        group["parameter_tensors"] += 1
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared_l2 = float(gradient.square().sum().item())
        absolute_max = float(gradient.abs().max().item()) if gradient.numel() else 0.0
        group["tensors_with_grad"] += 1
        group["tensors_with_nonzero_grad"] += int(absolute_max > 0.0)
        group["grad_squared_l2"] += squared_l2
        group["grad_abs_max"] = max(group["grad_abs_max"], absolute_max)
    for group in grouped.values():
        group["grad_l2"] = math.sqrt(group.pop("grad_squared_l2"))
        group["gradient_coverage"] = group["tensors_with_grad"] / max(
            group["parameter_tensors"], 1
        )
    return {"groups": grouped}


def _globally_has_annotations(batch, device: torch.device, world_size: int) -> bool:
    available = torch.tensor(
        int(bool(batch["prompt_texts"])), dtype=torch.int32, device=device
    )
    if world_size > 1:
        # All ranks skip together if one rank receives an all-background batch.
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
    return bool(available.item())


def _sample_training_prompts(
    batch: dict[str, Any],
    limits: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    level = batch["sample_levels"][0]
    if level == "scene":
        return _expand_dataset_local_prompt_pairs(
            batch,
            level,
            max_prompts=limits[level],
            sampling_seed=seed,
        )
    if level == "lulc":
        return _expand_dataset_local_prompt_pairs(
            batch,
            level,
            max_prompts=limits[level],
            sampling_seed=seed,
        )
    return _expand_object_prompt_pairs(
        batch,
        max_prompts=limits["object"],
        max_instances=limits["object_instances"],
        sampling_seed=seed,
    )


def _global_loss_means(
    sums: Counter[str],
    counts: Counter[str],
    device: torch.device,
    world_size: int,
) -> dict[str, float | None]:
    """Average each loss only across steps where that loss was active."""
    names = list(LOG_LOSS_NAMES)
    packed = torch.tensor(
        [value for name in names for value in (sums[name], counts[name])],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    means = {}
    for index, name in enumerate(names):
        numerator = float(packed[2 * index].item())
        denominator = float(packed[2 * index + 1].item())
        means[name] = numerator / denominator if denominator > 0 else None
    return means


def _global_level_loss_means(
    sums: Counter[str],
    counts: Counter[str],
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    levels = list(LEVEL_RATIOS)
    packed = torch.tensor(
        [value for level in levels for value in (sums[level], counts[level])],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        level: float(packed[2 * index] / packed[2 * index + 1])
        for index, level in enumerate(levels)
        if packed[2 * index + 1] > 0
    }


def _global_named_means(
    sums: Counter[str], counts: Counter[str], world_size: int
) -> dict[str, float]:
    """Merge sparse per-dataset metrics whose keys differ across DDP ranks."""
    local = {"sums": dict(sums), "counts": dict(counts)}
    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    global_sums: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    for row in gathered:
        global_sums.update(row["sums"])
        global_counts.update(row["counts"])
    return {
        name: global_sums[name] / global_counts[name]
        for name in sorted(global_sums)
        if global_counts[name]
    }


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class _FixedDecimal(float):
    def __new__(cls, value: float, places: int):
        number = super().__new__(cls, value)
        number.places = places
        return number


def _metrics_json(value: Any) -> str:
    if isinstance(value, _FixedDecimal):
        return f"{value:.{value.places}f}"
    if isinstance(value, dict):
        items = (
            f"{json.dumps(key, ensure_ascii=False)}:{_metrics_json(item)}"
            for key, item in value.items()
        )
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_metrics_json(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _append_metrics_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_metrics_json(row) + "\n")


def _json_metrics(
    values: dict[str, float | int | None],
) -> dict[str, float | int | None]:
    percentage_terms = (
        "accuracy",
        "coverage",
        "f1",
        "iou",
        "map",
        "precision",
        "probability",
        "recall",
        "reciprocal_rank",
    )
    formatted = {}
    for name, value in values.items():
        if value is None or isinstance(value, int):
            formatted[name] = value
            continue
        percentage = any(term in name for term in percentage_terms)
        places = 2 if percentage else 4
        formatted[name] = _FixedDecimal(
            round(float(value) * (100.0 if percentage else 1.0), places),
            places,
        )
    return formatted


def _level_log_metrics(
    losses: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    return {
        level: {"presence": losses[f"loss_{level}_presence"]}
        for level in ("scene", "object", "lulc")
    }



def _resolve_required_file(path: Path | None, argument: str) -> Path | None:
    """Resolve an explicitly supplied input before any SAM3/DDP allocation."""
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{argument} does not exist or is not a file: {resolved}"
        )
    return resolved


def _validate_checkpoint_settings(
    state: dict[str, Any], args: argparse.Namespace, argument: str
) -> None:
    saved = state["args"]
    if not any(name.startswith("pixel_router.router.") for name in state.get("model", {})):
        raise ValueError("Main training requires a Normpixelhead checkpoint")
    if saved.get("router_strategy") != "pixel":
        raise ValueError("Main training requires learned pixel fusion")
    if argument == "--resume" and not (saved.get("training_objective") == "unified-head"
            and saved.get("fusion_loss_only") and saved.get("fusion_end_to_end")):
        raise ValueError("Main training resumes only end-to-end P+G(F) checkpoints")
    if argument == "--resume" and not saved.get("object_semantic_normalization"):
        raise ValueError("Cannot resume a checkpoint without normalized Object Semantic teacher targets")
    if argument == "--resume" and saved.get("training_spatial_policy") != args.training_spatial_policy:
        raise ValueError("Cannot resume a checkpoint with a different spatial policy; use --init-weights for a new run")
    expected = {
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
    }
    for name, current in expected.items():
        previous = saved[name]
        matches = (
            math.isclose(float(previous), float(current), abs_tol=1e-12)
            if name == "lora_scale"
            else previous == current
        )
        if not matches:
            raise ValueError(
                f"{argument} {name}={previous}, current run uses {current}"
            )


def _read_training_checkpoint(
    path: Path,
    model: SAM3NativeMultilevelModel,
    args: argparse.Namespace,
    argument: str,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_settings(state, args, argument)
    model.load_trainable_state_dict(state["model"])
    return state


def main() -> None:
    args = parse_args()
    multigranularity_config = _load_multigranularity_config(args.config)
    prompt_limits, scene_loss_config, loss_config = _configure_run(
        args, multigranularity_config
    )
    args.sam3_checkpoint = _resolve_required_file(
        args.sam3_checkpoint, "--sam3-checkpoint"
    )
    args.sam3_bpe_path = _resolve_required_file(args.sam3_bpe_path, "--sam3-bpe-path")
    args.init_weights = _resolve_required_file(args.init_weights, "--init-weights")
    args.resume = _resolve_required_file(args.resume, "--resume")
    rank, world_size, local_rank, device = _distributed_context(args.device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    dataset, sample_levels, sample_dataset_groups, dataset_rows = _build_datasets(args)
    validation_loaders = _build_validation_loaders(args, rank, world_size, device)
    samples_per_epoch = args.samples_per_epoch or len(dataset)
    sampler = DistributedLevelBatchSampler(
        sample_levels,
        sample_dataset_groups,
        samples_per_epoch,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    collator = EpochPromptCollator(args.seed, args.sam3_input_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=False,
        persistent_workers=False,
    )

    model = SAM3NativeMultilevelModel.from_pretrained(
        None if args.sam3_checkpoint is None else str(args.sam3_checkpoint),
        bpe_path=None if args.sam3_bpe_path is None else str(args.sam3_bpe_path),
        load_from_hf=args.load_from_hf,
        device=device,
        sam3_input_size=args.sam3_input_size,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        object_teacher_chunk_size=args.object_teacher_chunk_size,
        scene_pseudo_lower_quantile=float(
            scene_loss_config.get("pseudo_lower_quantile", 0.02)
        ),
        scene_pseudo_upper_quantile=float(
            scene_loss_config.get("pseudo_upper_quantile", 0.98)
        ),
        scene_pseudo_threshold=float(scene_loss_config.get("pseudo_threshold", 0.5)),
    )
    trainability = model.configure_trainable()
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, args), weight_decay=args.weight_decay
    )
    total_steps = max(1, len(loader) * args.epochs)

    def schedule(step: int) -> float:
        return _polynomial_lr_multiplier(step, total_steps, args.poly_power)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = _build_criterion(loss_config, device)
    start_epoch = 0
    global_step = 0
    best_validation_score, best_epoch = None, None
    if args.init_weights:
        _read_training_checkpoint(args.init_weights, model, args, "--init-weights")
    if args.resume:
        state = _read_training_checkpoint(args.resume, model, args, "--resume")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_validation_score, best_epoch = _best_validation_state(state)

    training_step: nn.Module = TrainingStep(model, criterion)
    if world_size > 1:
        training_step = DistributedDataParallel(
            training_step,
            device_ids=[local_rank] if device.type == "cuda" else None,
            # Each task reaches a different subset of SAM3's LoRA weights.
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    raw_model = model
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(args.precision == "fp16" and device.type == "cuda")
    )
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_path = checkpoint_dir / "last.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    metrics_path = output_dir / "train_metrics.jsonl"
    gradient_audit_path = output_dir / "gradient_audit.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Legacy checkpoints may carry a validation score without a saved best.pt.
        if args.resume and args.checkpoint_policy != "none" and best_epoch == start_epoch:
            torch.save({
                **state,
                "validation_score": best_validation_score,
                "best_validation_score": best_validation_score,
                "best_epoch": best_epoch,
            }, best_checkpoint_path)
        datasets_text = ", ".join(
            f"{row['level']}:{Path(row['root']).name}({row['samples']})"
            for row in dataset_rows
        )
        print(
            f"[train] device={device} world={world_size} "
            f"trainable={trainability['trainable']:,} rank={args.lora_rank} "
            f"steps/epoch={len(loader)} datasets=[{datasets_text}]",
            flush=True,
        )

    audited_tasks: set[str] = set()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        collator.epoch = epoch
        training_step.train()
        epoch_start = time.monotonic()
        if rank == 0:
            print(
                f"[train] epoch={epoch + 1}/{args.epochs} loading first batch...",
                flush=True,
            )
        epoch_loss_sums: Counter[str] = Counter()
        epoch_loss_counts: Counter[str] = Counter()
        dataset_metric_sums: Counter[str] = Counter()
        dataset_metric_counts: Counter[str] = Counter()
        level_loss_sums: Counter[str] = Counter()
        level_loss_counts: Counter[str] = Counter()
        used_steps = 0
        for step, batch in enumerate(loader):
            first_step = step == 0
            if rank == 0 and first_step:
                print("[train] first batch loaded; preparing prompts...", flush=True)
            prompt_sampling_seed = args.seed + epoch * len(loader) + step
            batch = _sample_training_prompts(batch, prompt_limits, prompt_sampling_seed)
            if not _globally_has_annotations(batch, device, world_size):
                continue
            batch["images"] = batch["images"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if rank == 0 and first_step:
                level = batch["task_type"]
                print(
                    f"[train] first batch level={level} images={len(batch['images'])} "
                    f"prompts={len(batch['prompt_texts'])}; forward...",
                    flush=True,
                )
            with _autocast(device, args.precision):
                losses = training_step(batch)
            task_type = batch["task_type"]
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            if args.gradient_audit and rank == 0 and task_type not in audited_tasks:
                audit_row = {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "global_step": global_step,
                    "task_type": task_type,
                    "loss_total": float(losses["loss_total"].detach()),
                    **_gradient_topology_report(raw_model),
                }
                _append_jsonl(gradient_audit_path, audit_row)
                audited_tasks.add(task_type)
            nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            used_steps += 1
            level_loss_sums[task_type] += float(losses["loss_total"].detach())
            level_loss_counts[task_type] += 1
            should_log = first_step or (
                args.log_every and (step + 1) % args.log_every == 0
            )
            if should_log:
                level_loss_means = _global_level_loss_means(
                    level_loss_sums, level_loss_counts, device, world_size
                )
            if rank == 0 and should_log:
                elapsed = time.monotonic() - epoch_start
                average_losses = ",".join(
                    f"{level}:{level_loss_means[level]:.4f}"
                    for level in LEVEL_RATIOS
                    if level in level_loss_means
                )
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(loader)} "
                    f"level={task_type} avg_loss=[{average_losses}] "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            for key in LOG_LOSS_NAMES:
                value = losses.get(key)
                if value is not None:
                    epoch_loss_sums[key] += float(value.detach())
                    epoch_loss_counts[key] += 1
            for key, value in losses.get("dataset_metrics", {}).items():
                dataset_metric_sums[key] += float(value)
                dataset_metric_counts[key] += 1

        mean_losses = _global_loss_means(
            epoch_loss_sums, epoch_loss_counts, device, world_size
        )
        level_loss_means = _global_level_loss_means(
            level_loss_sums, level_loss_counts, device, world_size
        )
        mean_dataset_metrics = _global_named_means(
            dataset_metric_sums, dataset_metric_counts, world_size
        )
        run_validation = _should_validate(epoch + 1, args.epochs, args.validation_every)
        validation_metrics = _evaluate_validation(
            raw_model,
            validation_loaders,
            device,
            args.precision,
            rank,
            args.val_log_every,
        ) if run_validation else {}
        if rank == 0:
            validation_score = _mean_validation_score(validation_metrics)
            improved = validation_score is not None and (
                best_validation_score is None or validation_score > best_validation_score
            )
            if improved:
                best_validation_score, best_epoch = validation_score, epoch + 1
            should_save_checkpoint = args.checkpoint_policy == "each_epoch" or (
                args.checkpoint_policy == "final" and epoch + 1 == args.epochs
            )
            should_save_best = improved and args.checkpoint_policy != "none"
            if should_save_checkpoint or should_save_best:
                checkpoint = {
                    "model": raw_model.trainable_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "trainability": trainability,
                    "args": vars(args),
                    "multigranularity_config": multigranularity_config,
                    "datasets": dataset_rows,
                    "validation": validation_metrics,
                    "validation_performed": run_validation,
                    "validation_score": validation_score,
                    "best_validation_score": best_validation_score,
                    "best_epoch": best_epoch,
                }
                if should_save_checkpoint:
                    torch.save(checkpoint, checkpoint_path)
                if should_save_best:
                    torch.save(checkpoint, best_checkpoint_path)
            mean_losses = _json_metrics(mean_losses)
            json_validation = {
                dataset: _json_metrics(metrics)
                for dataset, metrics in validation_metrics.items()
            }
            level_metrics = _level_log_metrics(mean_losses)
            epoch_row = {
                "epoch": epoch + 1,
                "completed_steps": used_steps,
                "trainability": trainability,
                **mean_losses,
                **level_metrics,
                "loss_components": mean_losses,
                "dataset_metrics": _json_metrics(mean_dataset_metrics),
                "level_mean_loss": _json_metrics(level_loss_means),
                "validation": json_validation,
                "validation_performed": run_validation,
                "checkpoint": str(checkpoint_path) if should_save_checkpoint else None,
                "validation_score": validation_score,
                "best_validation_score": best_validation_score,
                "best_epoch": best_epoch,
                "best_checkpoint": str(best_checkpoint_path) if should_save_best else None,
            }
            print(
                f"[train] epoch={epoch + 1} complete steps={used_steps} "
                f"loss={mean_losses['loss_total']:.4f} "
                f"val=[{_format_validation(validation_metrics) if run_validation else 'skipped'}] "
                f"checkpoint={'saved' if should_save_checkpoint else 'not-saved'}",
                flush=True,
            )
            if validation_score is not None:
                print(
                    f"[val] mean_score={validation_score:.4f} "
                    f"best_score={best_validation_score:.4f} best_epoch={best_epoch} "
                    f"best_checkpoint={'saved' if should_save_best else 'not-saved'}",
                    flush=True,
                )
            _append_metrics_jsonl(metrics_path, epoch_row)
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
