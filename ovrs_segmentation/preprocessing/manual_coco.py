"""Convert reviewed per-image COCO polygons to independent segmentation test GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import SemanticSegmentationDataset, write_class_catalog
from ovrs_segmentation.datasets.class_prompts import normalize_class_text
from .manual_dota import rasterize_annotations


def _categories(rows):
    result = {}
    for row in rows:
        class_id, name = int(row["id"]), str(row["name"]).strip()
        if not 0 < class_id < 255 or not name or class_id in result:
            raise ValueError(f"Invalid or duplicate foreground category: {row}")
        result[class_id] = name
    if not result or len({normalize_class_text(name) for name in result.values()}) != len(result):
        raise ValueError("Categories must have unique, non-empty names")
    return result


def convert(source: Path, output: Path, *, dataset_name: str, semantic_level: str):
    source, output = source.resolve(), output.resolve()
    if semantic_level not in {"scene", "object", "lulc"}:
        raise ValueError(f"Invalid semantic level: {semantic_level}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output}")
    images = sorted(p for p in (source / "Images").iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError("Source images are empty or have duplicate sample IDs")
    annotations_dir = source / "Annotations"
    expected = {p.stem for p in images}
    available = {p.stem for p in annotations_dir.glob("*.json") if p.name != "_workspace_categories.json"}
    if available != expected:
        raise ValueError(f"Image/annotation mismatch: missing={expected - available}, orphan={available - expected}")
    category_file = annotations_dir / "_workspace_categories.json"
    if not category_file.is_file():
        category_file = annotations_dir / f"{images[0].stem}.json"
    categories = _categories(json.loads(category_file.read_text(encoding="utf-8-sig"))["categories"])
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix=".manual-coco-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        for name in ("images", "masks", "splits"):
            (stage / name).mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(images, 1):
            annotation_path = annotations_dir / f"{image_path.stem}.json"
            data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
            if len(data["images"]) != 1:
                raise ValueError(f"Expected one image in {annotation_path}")
            record = data["images"][0]
            if Path(record["file_name"]).name != image_path.name:
                raise ValueError(f"Image filename mismatch: {annotation_path}")
            local = _categories(data["categories"])
            for class_id in categories.keys() & local.keys():
                if normalize_class_text(local[class_id]) != normalize_class_text(categories[class_id]):
                    raise ValueError(f"Category name mismatch: {annotation_path}, {class_id}")
            annotations = data["annotations"]
            ids = [row["id"] for row in annotations]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate annotation IDs: {annotation_path}")
            if any(row["image_id"] != record["id"] for row in annotations):
                raise ValueError(f"Annotation image_id mismatch: {annotation_path}")
            if any(int(row["category_id"]) not in local for row in annotations):
                raise ValueError(f"Undeclared annotation category: {annotation_path}")
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                if (width, height) != (int(record["width"]), int(record["height"])):
                    raise ValueError(f"Image dimensions mismatch: {annotation_path}")
                mode = image.mode
                if mode == "RGB":
                    shutil.copyfile(image_path, stage / "images" / image_path.name)
                else:
                    image.convert("RGB").save(stage / "images" / image_path.name)
            mask = rasterize_annotations(annotations, (height, width), categories)
            Image.fromarray(mask).save(stage / "masks" / f"{image_path.stem}.png")
            values, counts = np.unique(mask, return_counts=True)
            reports.append({
                "sample_id": image_path.stem, "width": width, "height": height,
                "source_image_mode": mode, "instances": len(annotations),
                "source_annotation": f"Annotations/{annotation_path.name}",
                "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "unused_extra_categories": {str(k): v for k, v in local.items() if k not in categories},
                "pixel_counts": dict(zip(map(str, values.tolist()), counts.tolist())),
            })
            if index == 1 or index % 20 == 0 or index == len(images):
                print(f"[convert] {dataset_name}: {index}/{len(images)}", flush=True)
        write_class_catalog(stage / "classes.json", dataset_name, {0: "background", **categories},
                            default_semantic_level=semantic_level, background_id=0)
        metadata = {
            "format_version": "ovrs-semantic-v1", "dataset_name": dataset_name,
            "dataset_id": output.name, "image_dir": "images", "mask_dir": "masks",
            "classes_file": "classes.json", "background_id": 0, "ignore_id": 255,
            "semantic_level": semantic_level, "supervision_level": "pixel",
            "annotation_source": "user-provided manually reviewed COCO polygons",
            "overlap_policy": "same-class union; different-class overlap ignored (255)",
            "polygon_rasterization": "round vertices to nearest integer; OpenCV fillPoly",
            "splits": {"test": len(images)},
        }
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (stage / "splits/test.txt").write_text("".join(f"{p.stem}\n" for p in images), encoding="utf-8")
        (stage / "export_report.json").write_text(json.dumps({"samples": reports}, indent=2) + "\n", encoding="utf-8")
        validation = SemanticSegmentationDataset(stage, "test", validate_masks=True)
        if len(validation) != len(images):
            raise ValueError("Converted test sample count differs from source")
        # Sync clients on Windows can briefly lock a newly written directory.
        for attempt in range(8):
            try:
                stage.rename(output)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.5)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--semantic-level", choices=("scene", "object", "lulc"), required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, dataset_name=args.dataset_name,
                             semantic_level=args.semantic_level), indent=2))


if __name__ == "__main__":
    main()
