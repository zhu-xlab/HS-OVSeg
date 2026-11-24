#!/usr/bin/env python3
"""Prepare scene labels as weak full-image query-region supervision."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import _bootstrap  # noqa: F401
from ovrs_segmentation.datasets import (
    combined_prompt_texts,
    normalize_class_text,
    write_class_catalog,
)


CLASSES = {
    "resisc45": (
        "airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge",
        "chaparral", "church", "circular farmland", "cloud", "commercial area",
        "dense residential", "desert", "forest", "freeway", "golf course", "ground track field",
        "harbor", "industrial area", "intersection", "island", "lake", "meadow",
        "medium residential", "mobile home park", "mountain", "overpass", "palace",
        "parking lot", "railway", "railway station", "rectangular farmland", "river",
        "roundabout", "runway", "sea ice", "ship", "snowberg", "sparse residential",
        "stadium", "storage tank", "tennis court", "terrace", "thermal power station", "wetland",
    ),
    "patternnet": (
        "airplane", "baseball field", "basketball court", "beach", "bridge", "cemetery",
        "chaparral", "christmas tree farm", "closed road", "coastal mansion",
        "crosswalk", "dense residential", "ferry terminal", "football field", "forest",
        "freeway", "golf course", "harbor", "intersection", "mobile home park",
        "nursing home", "oil gas field", "oil well", "overpass", "parking lot",
        "parking space", "railway", "river", "runway", "runway marking", "shipping yard",
        "solar panel", "sparse residential", "storage tank", "swimming pool", "tennis court",
        "transformer station", "wastewater treatment plant",
    ),
    "aid": (
        "airport", "bare land", "baseball field", "beach", "bridge", "center", "church",
        "commercial", "dense residential", "desert", "farmland", "forest", "industrial",
        "meadow", "medium residential", "mountain", "park", "parking", "playground",
        "pond", "port", "railway station", "resort", "river", "school",
        "sparse residential", "square", "stadium", "storage tanks", "viaduct",
    ),
}
ROOT_NAMES = {
    "resisc45": "NWPU-RESISC45",
    "patternnet": "PatternNet",
    "aid": "AID",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image-level scene text-prompt annotations")
    parser.add_argument("dataset", choices=sorted(CLASSES))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover(dataset: str, root: Path) -> list[dict]:
    raw = root.expanduser().resolve() / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    class_lookup = {_key(text): (index, text) for index, text in enumerate(CLASSES[dataset])}
    official = {}
    if dataset == "resisc45":
        for split in ("train", "val", "test"):
            path = raw / f"resisc45-{split}.txt"
            if not path.is_file():
                raise FileNotFoundError(f"Official RESISC45 split is absent: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    official[Path(line.strip()).name.lower()] = split
    records = []
    for path in raw.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = class_lookup.get(_key(path.parent.name))
        if label is None:
            continue
        class_id, class_text = label
        records.append({
            "source": path,
            "class_id": class_id,
            "class_text": class_text,
            "split": official.get(path.name.lower()),
        })
    if not records:
        raise FileNotFoundError(f"No class-folder images found below {raw}")
    if dataset == "resisc45" and any(row["split"] is None for row in records):
        missing = [row["source"].name for row in records if row["split"] is None]
        raise ValueError(f"{len(missing)} RESISC45 images are absent from official splits: {missing[:10]}")
    return records


def _assign_stratified(records: list[dict], seed: int) -> None:
    by_class: dict[int, list[dict]] = {}
    for row in records:
        by_class.setdefault(row["class_id"], []).append(row)
    for class_id, values in by_class.items():
        random.Random(seed + class_id).shuffle(values)
        train_end, val_end = round(len(values) * 0.7), round(len(values) * 0.85)
        for index, row in enumerate(values):
            row["split"] = "train" if index < train_end else "val" if index < val_end else "test"


def _convert(task: tuple[str, str, int, bool]) -> tuple[str, int, int]:
    source_text, destination_text, size, overwrite = task
    source, destination = Path(source_text), Path(destination_text)
    if destination.is_file() and not overwrite:
        try:
            with Image.open(destination) as image:
                image.load()
                return destination.stem, image.width, image.height
        except (UnidentifiedImageError, OSError, SyntaxError):
            # A previous interrupted conversion may have left a valid PNG
            # header but an incomplete payload. Rebuild just that sample.
            pass
    with Image.open(source) as stream:
        image = stream.convert("RGB")
        original_width, original_height = image.size
        scale = min(size / original_width, size / original_height)
        resized = image.resize(
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        canvas.save(temporary, format="PNG")
        # Decode the complete temporary payload before publishing it.
        with Image.open(temporary) as stream:
            stream.load()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.stem, original_width, original_height


def prepare(dataset: str, root: Path, size: int, workers: int, seed: int, overwrite: bool) -> dict:
    if size < 1 or workers < 1:
        raise ValueError("--size and --workers must be positive")
    root = root.expanduser().resolve()
    records = discover(dataset, root)
    if any(row["split"] is None for row in records):
        _assign_stratified(records, seed)
    used_ids = set()
    for row in records:
        base = f"{_key(row['class_text'])}__{row['source'].stem}"
        sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
        if sample_id in used_ids:
            raise ValueError(f"Duplicate scene sample ID: {sample_id}")
        used_ids.add(sample_id)
        row["sample_id"] = sample_id
        row["destination"] = root / "images" / f"{sample_id}.png"
    tasks = [(str(row["source"]), str(row["destination"]), size, overwrite) for row in records]
    if workers == 1:
        converted = list(tqdm(map(_convert, tasks), total=len(tasks), desc=f"Preparing {dataset}"))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            converted = list(tqdm(executor.map(_convert, tasks, chunksize=8), total=len(tasks), desc=f"Preparing {dataset}"))
    dimensions = {sample_id: (width, height) for sample_id, width, height in converted}
    active = {row["sample_id"] for row in records}
    if overwrite:
        for old in (root / "images").glob("*.png"):
            if old.stem not in active:
                old.unlink()
    rows = []
    splits = {name: [] for name in ("train", "val", "test")}
    for row in sorted(records, key=lambda item: item["sample_id"]):
        sample_id = row["sample_id"]
        width, height = dimensions[sample_id]
        class_text = normalize_class_text(row["class_text"])
        prompt_groups = combined_prompt_texts(
            class_text, "scene", ROOT_NAMES[dataset]
        )
        splits[row["split"]].append(sample_id)
        rows.append({
            "sample_id": sample_id,
            "image": f"images/{sample_id}.png",
            "dataset": ROOT_NAMES[dataset],
            "source_size": [width, height],
            "annotations": [{
                "annotation_id": f"{sample_id}:scene",
                "dataset_class_id": row["class_id"],
                "class_text": class_text,
                **prompt_groups,
                "semantic_level": "scene",
                "supervision_level": "image",
                "optimization_targets": ["region_text_alignment", "image_presence"],
                "target": {"type": "presence", "present": True},
                "present": True,
                "ignore_id": 255,
                "annotation_quality": {
                    "label": "native",
                    "source": "human_scene_classification",
                    "confidence": 1.0,
                },
            }],
        })
    (root / "annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    classes = {index: text for index, text in enumerate(CLASSES[dataset])}
    write_class_catalog(
        root / "classes.json",
        ROOT_NAMES[dataset],
        classes,
        default_semantic_level="scene",
        background_id=None,
    )
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (split_dir / f"{name}.txt").write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    metadata = {
        "dataset_name": ROOT_NAMES[dataset],
        "task": "text_prompt_multilevel_supervision",
        "native_supervision": "scene_classification",
        "semantic_level": "scene",
        "supervision_level": "image",
        "optimization_by_level": {
            "scene": ["region_text_alignment", "image_presence"],
        },
        "image_dir": "images",
        "annotations_file": "annotations.jsonl",
        "classes_file": "classes.json",
        "classes_format": "ovrs-classes-v1",
        "image_size": [size, size],
        "resize_policy": "letterbox_full_field_of_view",
        "pixel_masks_available": False,
        "scene_labels_are_not_pixel_labels": True,
        "class_ids_are_dataset_local": True,
        "cross_dataset_alignment": "normalized_class_text_and_prompt_texts",
        "sample_count": len(rows),
        "quality_control": {
            "decode_and_rgb_conversion_required": True,
        },
        "source_scope": "official_distribution",
        "splits": {name: len(values) for name, values in splits.items()},
    }
    (root / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"dataset": dataset, "root": str(root), "samples": len(rows), "splits": metadata["splits"]}


def main() -> None:
    args = parse_args()
    root = args.dataset_root or Path("/home/Datasets/RSOD") / ROOT_NAMES[args.dataset]
    print(json.dumps(prepare(args.dataset, root, args.size, args.workers, args.seed, args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
