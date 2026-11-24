#!/usr/bin/env python3
"""Decode every sample in one unified dataset and report all broken files."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

import _bootstrap  # noqa: F401
from ovrs_segmentation.datasets import open_text_prompt_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Failure JSONL path; default: "
            "outputs/data_integrity/<dataset>_<split>_errors.jsonl"
        ),
    )
    return parser.parse_args()


def _dataset_name(dataset: Any) -> str:
    if hasattr(dataset, "dataset_name"):
        return str(dataset.dataset_name)
    return str(dataset.semantic.dataset_name)


def _sample_id(dataset: Any, index: int) -> str:
    if hasattr(dataset, "sample_ids"):
        return str(dataset.sample_ids[index])
    return str(dataset.semantic.sample_ids[index])


def check_dataset(
    dataset_root: Path,
    *,
    split: str | None,
    workers: int,
    max_samples: int | None,
    report: Path | None,
) -> dict[str, Any]:
    if workers < 0:
        raise ValueError("--workers must be non-negative")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max-samples must be positive")

    dataset = open_text_prompt_dataset(
        dataset_root.expanduser().resolve(), split=split, validate_masks=False
    )
    dataset_name = _dataset_name(dataset)
    sample_count = len(dataset)
    check_count = sample_count if max_samples is None else min(sample_count, max_samples)

    def check(index: int) -> dict[str, Any] | None:
        sample_id = _sample_id(dataset, index)
        try:
            sample = dataset[index]
            # Force all decoded arrays to be materialized before returning.
            _ = sample["image"].shape
            for annotation in sample["annotations"]:
                target = annotation.get("target")
                if hasattr(target, "shape"):
                    _ = target.shape
            return None
        except Exception as error:  # continue so the report contains every bad file
            return {
                "index": index,
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    indices = range(check_count)
    if workers == 0:
        results = map(check, indices)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(check, indices)

    failures: list[dict[str, Any]] = []
    try:
        for result in tqdm(
            results,
            total=check_count,
            desc=f"Checking {dataset_name}",
            unit="sample",
        ):
            if result is not None:
                failures.append(result)
                print(
                    f"ERROR [{result['sample_id']}] {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if workers != 0:
            executor.shutdown(wait=True)

    if report is None:
        safe_name = dataset_name.replace("/", "_").replace(" ", "_")
        split_name = split or "all"
        report = (
            Path("outputs")
            / "data_integrity"
            / f"{safe_name}_{split_name}_errors.jsonl"
        )
    report = report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")

    return {
        "valid": not failures,
        "dataset": dataset_name,
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "split": split,
        "checked_samples": check_count,
        "total_samples": sample_count,
        "failed_samples": len(failures),
        "report": str(report),
    }


def main() -> None:
    args = parse_args()
    summary = check_dataset(
        args.dataset_root,
        split=args.split,
        workers=args.workers,
        max_samples=args.max_samples,
        report=args.report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
