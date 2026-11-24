#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from ovrs_segmentation.datasets import open_text_prompt_dataset
from ovrs_segmentation.datasets.class_prompts import normalize_class_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate scene/LULC/object text-prompt supervision"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def validate(root: Path, split: str | None, max_samples: int | None = None) -> dict:
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max-samples must be positive")
    dataset = open_text_prompt_dataset(root, split=split)
    level_counts: Counter[str] = Counter()
    supervision_counts: Counter[str] = Counter()
    class_counts: Counter[tuple[int, str]] = Counter()
    prompt_count = positive_pixels = ignored_pixels = box_count = 0
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(limit):
        sample = dataset[index]
        if sample["image"].ndim != 3 or sample["image"].shape[2] != 3:
            raise ValueError(f"Sample {sample['sample_id']} is not stored as RGB")
        for annotation in sample["annotations"]:
            required = {
                "dataset_class_id", "class_text", "prompt_texts", "semantic_level",
                "supervision_level", "ignore_id", "annotation_quality",
                "optimization_targets", "native_prompt_texts",
                "canonical_prompt_texts", "variant_prompt_texts",
            }
            missing = required - set(annotation)
            if missing:
                raise ValueError(
                    f"Annotation {annotation.get('annotation_id')} misses {sorted(missing)}"
                )
            if annotation["supervision_level"] != "box" and "target" not in annotation:
                raise ValueError(
                    f"Annotation {annotation.get('annotation_id')} has no target"
                )
            if not annotation["prompt_texts"]:
                raise ValueError(f"Annotation {annotation['annotation_id']} has no prompts")
            expected = list(
                dict.fromkeys(
                    annotation["native_prompt_texts"]
                    + annotation["canonical_prompt_texts"]
                    + annotation["variant_prompt_texts"]
                )
            )
            if annotation["prompt_texts"] != expected:
                raise ValueError(
                    f"Annotation {annotation['annotation_id']} prompt_texts is not the "
                    "ordered union of its native, canonical, and variant groups"
                )
            mechanical_land_cover = normalize_class_text(annotation["class_text"]) + " land cover"
            if mechanical_land_cover in {
                normalize_class_text(prompt) for prompt in annotation["prompt_texts"]
            }:
                raise ValueError(
                    f"Annotation {annotation['annotation_id']} contains the mechanical "
                    f"prompt {mechanical_land_cover!r}; regenerate classes.json"
                )
            level_counts[annotation["semantic_level"]] += 1
            supervision_counts[annotation["supervision_level"]] += 1
            class_counts[(annotation["dataset_class_id"], annotation["class_text"])] += 1
            prompt_count += len(annotation["prompt_texts"])
            if annotation["supervision_level"] == "pixel":
                target = np.asarray(annotation["target"])
                if target.shape != sample["image"].shape[:2]:
                    raise ValueError(
                        f"Target/image mismatch in {annotation['annotation_id']}: "
                        f"{target.shape} vs {sample['image'].shape[:2]}"
                    )
                unknown = set(map(int, np.unique(target))) - {0, 1, annotation["ignore_id"]}
                if unknown:
                    raise ValueError(
                        f"Binary text target contains invalid values: {sorted(unknown)}"
                    )
                positive_pixels += int((target == 1).sum())
                ignored_pixels += int((target == annotation["ignore_id"]).sum())
            elif annotation["supervision_level"] == "box":
                boxes = annotation.get("boxes_xyxy", [])
                if not boxes:
                    raise ValueError(
                        f"Box annotation {annotation['annotation_id']} contains no boxes"
                    )
                box_count += len(boxes)
            elif not isinstance(annotation["target"], (bool, np.bool_)):
                raise ValueError("Image-level presence target must be boolean")
    return {
        "valid": True,
        "dataset": dataset.dataset_name if hasattr(dataset, "dataset_name") else dataset.semantic.dataset_name,
        "split": split,
        "checked_samples": limit,
        "total_samples": len(dataset),
        "semantic_level_counts": dict(sorted(level_counts.items())),
        "supervision_level_counts": dict(sorted(supervision_counts.items())),
        "annotation_counts_by_local_class": {
            str(class_id): {"class_text": text, "annotations": count}
            for (class_id, text), count in sorted(class_counts.items())
        },
        "prompt_variant_count": prompt_count,
        "positive_pixel_count": positive_pixels,
        "ignored_pixel_count": ignored_pixels,
        "detection_box_count": box_count,
        "global_class_id_mapping_used": False,
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(
        validate(args.dataset_root, args.split, args.max_samples),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
