#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from ovrs_segmentation.datasets import SemanticSegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an OVRS unified semantic dataset")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", help="Validate only one split, for example train or val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SemanticSegmentationDataset(args.dataset_root, split=args.split, validate_masks=False)
    class_pixels: Counter[int] = Counter()
    ignored_pixels = 0
    for sample in dataset:
        mask = sample["class_id_mask"]
        ids, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(ids, counts):
            if int(class_id) == dataset.ignore_id:
                ignored_pixels += int(count)
            else:
                class_pixels[int(class_id)] += int(count)
    print(json.dumps({
        "valid": True,
        "format_version": dataset.format_version,
        "dataset_name": dataset.dataset_name,
        "split": args.split,
        "sample_count": len(dataset),
        "class_text_by_id": dataset.class_text_by_id,
        "class_pixel_counts": {
            str(class_id): {
                "class_text": dataset.class_text_by_id[class_id],
                "pixels": class_pixels.get(class_id, 0),
            }
            for class_id in dataset.class_text_by_id
            if class_id != dataset.ignore_id
        },
        "ignore_id": dataset.ignore_id,
        "ignored_pixels": ignored_pixels,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
