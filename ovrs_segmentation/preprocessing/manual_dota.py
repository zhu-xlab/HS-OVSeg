"""Convert reviewed per-image DOTA COCO polygons to independent pixel test GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import SemanticSegmentationDataset, write_class_catalog


DOTA_CLASSES = (
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
    "container-crane", "airport", "helipad",
)


def rasterize_annotations(annotations, shape, categories):
    """Union same-class polygons; ignore all conflicting class overlaps."""
    semantic = np.zeros(shape, dtype=np.uint8)
    by_class = {}
    for annotation in annotations:
        class_id = int(annotation["category_id"])
        if class_id not in categories:
            raise ValueError(f"Unsupported annotated category: {class_id}")
        polygons = annotation.get("segmentation")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Missing polygon segmentation in instance {annotation['id']}")
        converted = []
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float64)
            if points.ndim != 1 or points.size < 6 or points.size % 2 or not np.isfinite(points).all():
                raise ValueError(f"Invalid polygon in instance {annotation['id']}")
            points = points.reshape(-1, 2)
            if (np.abs(points) > 2**30).any():
                raise ValueError("Polygon coordinates exceed rasterizer limits")
            converted.append(np.rint(points).astype(np.int32))
        if class_id not in by_class:
            by_class[class_id] = np.zeros(shape, dtype=np.uint8)
        class_mask = by_class[class_id]
        # Separate fills preserve union for overlapping components.
        for points in converted:
            cv2.fillPoly(class_mask, [points], 1)
    for class_id, class_mask in sorted(by_class.items()):
        active = class_mask.astype(bool)
        conflict = active & (semantic != 0)
        semantic[active & (semantic == 0)] = class_id
        semantic[conflict] = 255
    return semantic


def convert(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output}")
    images = sorted(path for path in (source / "Images").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError("Source images are empty or have duplicate sample IDs")
    categories = dict(enumerate(DOTA_CLASSES, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix=".dota-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        for name in ("images", "masks", "splits"):
            (stage / name).mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(images):
            annotation_path = source / "Annotations" / f"{image_path.stem}.json"
            data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
            if len(data["images"]) != 1:
                raise ValueError(f"Expected one image in {annotation_path}")
            record = data["images"][0]
            if Path(record["file_name"]).name != image_path.name:
                raise ValueError(f"Image filename mismatch: {annotation_path}")
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                source_mode = image.mode
                if source_mode == "RGB":
                    shutil.copyfile(image_path, stage / "images" / image_path.name)
                else:
                    image.convert("RGB").save(stage / "images" / image_path.name)
            if (width, height) != (int(record["width"]), int(record["height"])):
                raise ValueError(f"Image dimensions mismatch: {annotation_path}")
            local_categories = {int(row["id"]): row["name"] for row in data["categories"]}
            for class_id, name in categories.items():
                if local_categories.get(class_id) != name:
                    raise ValueError(f"DOTA category mapping mismatch: {annotation_path}, {class_id}")
            if any(row["image_id"] != record["id"] for row in data["annotations"]):
                raise ValueError(f"Annotation image_id mismatch: {annotation_path}")
            mask = rasterize_annotations(data["annotations"], (height, width), categories)
            Image.fromarray(mask).save(stage / "masks" / f"{image_path.stem}.png")
            values, counts = np.unique(mask, return_counts=True)
            reports.append({
                "sample_id": image_path.stem, "width": width, "height": height,
                "source_image_mode": source_mode,
                "instances": len(data["annotations"]),
                "source_annotation": f"Annotations/{annotation_path.name}",
                "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "pixel_counts": dict(zip(map(str, values.tolist()), counts.tolist())),
                "source_info": data.get("info", {}),
            })
            print(f"[convert] {index + 1}/{len(images)} {image_path.stem}", flush=True)
        write_class_catalog(stage / "classes.json", "DOTA-v2", {0: "background", **categories},
                            default_semantic_level="object", background_id=0)
        metadata = {
            "format_version": "ovrs-semantic-v1", "dataset_name": "DOTA-v2",
            "dataset_id": output.name, "image_dir": "images", "mask_dir": "masks",
            "classes_file": "classes.json", "background_id": 0, "ignore_id": 255,
            "semantic_level": "object", "supervision_level": "pixel",
            "annotation_source": "user-provided manually reviewed COCO polygons",
            "overlap_policy": "same-class union; different-class overlap ignored (255)",
            "polygon_rasterization": "round vertices to nearest integer; OpenCV fillPoly",
            "splits": {"test": len(images)},
        }
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (stage / "splits/test.txt").write_text("".join(f"{p.stem}\n" for p in images), encoding="utf-8")
        (stage / "export_report.json").write_text(json.dumps({"samples": reports}, indent=2) + "\n", encoding="utf-8")
        validation = SemanticSegmentationDataset(stage, "test", validate_masks=True)
        assert len(validation) == len(images)
        stage.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
