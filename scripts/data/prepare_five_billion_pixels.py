#!/usr/bin/env python3
"""Tile Five-Billion-Pixels into the ovrs-semantic-v1 RGB/mask interface."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import _bootstrap  # noqa: F401

from ovrs_segmentation.datasets import write_class_catalog
from ovrs_segmentation.visualization import render_semantic_overlay


Image.MAX_IMAGE_PIXELS = None
CLASSES = {
    1: "industrial area", 2: "paddy field", 3: "irrigated field",
    4: "dry cropland", 5: "garden land", 6: "arbor forest",
    7: "shrub forest", 8: "park", 9: "natural meadow",
    10: "artificial meadow", 11: "river", 12: "urban residential",
    13: "lake", 14: "pond", 15: "fish pond", 16: "snow",
    17: "bareland", 18: "rural residential", 19: "stadium",
    20: "square", 21: "road", 22: "overpass",
    23: "railway station", 24: "airport",
}
TEST_SCENES = {
    "GF2_PMS1__L1A0001064454-MSS1", "GF2_PMS1__L1A0001118839-MSS1",
    "GF2_PMS1__L1A0001344822-MSS1", "GF2_PMS1__L1A0001348919-MSS1",
    "GF2_PMS1__L1A0001366278-MSS1", "GF2_PMS1__L1A0001366284-MSS1",
    "GF2_PMS1__L1A0001395956-MSS1", "GF2_PMS1__L1A0001432972-MSS1",
    "GF2_PMS1__L1A0001670888-MSS1", "GF2_PMS1__L1A0001680857-MSS1",
    "GF2_PMS1__L1A0001680858-MSS1", "GF2_PMS1__L1A0001757429-MSS1",
    "GF2_PMS1__L1A0001765574-MSS1", "GF2_PMS2__L1A0000607677-MSS2",
    "GF2_PMS2__L1A0000607681-MSS2", "GF2_PMS2__L1A0000718813-MSS2",
    "GF2_PMS2__L1A0001038935-MSS2", "GF2_PMS2__L1A0001038936-MSS2",
    "GF2_PMS2__L1A0001119060-MSS2", "GF2_PMS2__L1A0001367840-MSS2",
    "GF2_PMS2__L1A0001378491-MSS2", "GF2_PMS2__L1A0001378501-MSS2",
    "GF2_PMS2__L1A0001396036-MSS2", "GF2_PMS2__L1A0001396037-MSS2",
    "GF2_PMS2__L1A0001416129-MSS2", "GF2_PMS2__L1A0001471436-MSS2",
    "GF2_PMS2__L1A0001517494-MSS2", "GF2_PMS2__L1A0001591676-MSS2",
    "GF2_PMS2__L1A0001787564-MSS2", "GF2_PMS2__L1A0001821754-MSS2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/Datasets/RSOD/Five-Billion-Pixels"))
    parser.add_argument("--source-root", type=Path, help="Extracted official folder; defaults to DATASET_ROOT/raw")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-labeled-fraction", type=float, default=0.0)
    parser.add_argument("--num-overlays", type=int, default=50)
    parser.add_argument("--overlay-dir", type=Path, default=Path("outputs/five-billion-pixels/overlays"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scene_id(path: Path) -> str:
    return re.sub(r"_24label$", "", path.stem, flags=re.IGNORECASE)


def discover_pairs(source_root: Path, seed: int) -> list[dict]:
    source_root = source_root.expanduser().resolve()
    images = {
        _scene_id(path): path
        for path in source_root.rglob("*.tif")
        if "image__8bit_nirrgb" in path.as_posix().lower()
    }
    masks = {
        _scene_id(path): path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".tif", ".tiff"}
        and "annotation__index" in path.as_posix().lower()
    }
    image_count, mask_count = len(images), len(masks)
    missing_images = sorted(set(masks) - set(images))
    missing_masks = sorted(set(images) - set(masks))
    paired = set(images) & set(masks)
    if missing_images or missing_masks:
        print(
            "WARNING: incomplete official scene pairs will be skipped: "
            f"images={image_count}, masks={mask_count}, paired={len(paired)}, "
            f"missing_image_count={len(missing_images)}, "
            f"missing_mask_count={len(missing_masks)}, "
            f"missing_image_examples={missing_images[:5]}, "
            f"missing_mask_examples={missing_masks[:5]}",
            flush=True,
        )
        images = {scene: images[scene] for scene in paired}
        masks = {scene: masks[scene] for scene in paired}
    if not images:
        raise FileNotFoundError(
            "No complete Image__8bit_NirRGB TIFF/Annotation__index PNG-or-TIFF pairs were found: "
            f"images={image_count}, masks={mask_count}, paired={len(paired)} below {source_root}. "
            "Resume scripts/data/download_native_dataset.py five-billion-pixels before preparing."
        )
    remaining = sorted(set(images) - TEST_SCENES)
    random.Random(seed).shuffle(remaining)
    train_count = round(len(remaining) * 0.75)
    train = set(remaining[:train_count])
    records = []
    for scene in sorted(images):
        split = "test" if scene in TEST_SCENES else "train" if scene in train else "val"
        records.append({"scene": scene, "image": images[scene], "mask": masks[scene], "split": split})
    return records


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _read_nirrgb(path: Path) -> np.ndarray:
    # The official 8-bit TIFF order is NIR,R,G,B and the publisher explicitly
    # recommends PIL/CMYK for reading it.  CMYK here is a four-channel carrier;
    # channels 1..3 are the actual visible RGB bands.
    with Image.open(path) as stream:
        nirrgb = np.asarray(stream.convert("CMYK"), dtype=np.uint8)
    return np.ascontiguousarray(nirrgb[..., 1:4])


def _read_index_mask(path: Path) -> np.ndarray:
    with Image.open(path) as stream:
        mask = np.asarray(stream)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.uint8, copy=True)
    unknown = set(map(int, np.unique(mask))) - set(range(25))
    if unknown:
        raise ValueError(f"Mask {path} contains unknown official IDs: {sorted(unknown)}")
    mask[mask == 0] = 255
    return mask


def _process_scene(task: tuple) -> dict:
    record, root_text, tile_size, stride, minimum, overwrite = task
    root = Path(root_text)
    image = _read_nirrgb(record["image"])
    mask = _read_index_mask(record["mask"])
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Scene shape mismatch {record['scene']}: {image.shape[:2]} vs {mask.shape}")
    height, width = mask.shape
    rows = []
    converted = skipped = rejected = 0
    for y in _starts(height, tile_size, stride):
        for x in _starts(width, tile_size, stride):
            image_tile = image[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            mask_tile = mask[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            labeled_fraction = float(np.count_nonzero(mask_tile != 255) / mask_tile.size)
            if labeled_fraction < minimum:
                rejected += 1
                continue
            sample_id = f"{record['scene']}__x{x}_y{y}"
            image_out, mask_out = root / "images" / f"{sample_id}.png", root / "masks" / f"{sample_id}.png"
            valid_height, valid_width = mask_tile.shape
            if (valid_height, valid_width) != (tile_size, tile_size):
                padded_image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_mask = np.full((tile_size, tile_size), 255, dtype=np.uint8)
                padded_image[:valid_height, :valid_width] = image_tile
                padded_mask[:valid_height, :valid_width] = mask_tile
                image_tile, mask_tile = padded_image, padded_mask
            if image_out.is_file() and mask_out.is_file() and not overwrite:
                skipped += 1
            else:
                image_out.parent.mkdir(parents=True, exist_ok=True)
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image_tile, mode="RGB").save(image_out, compress_level=3)
                Image.fromarray(mask_tile, mode="L").save(mask_out)
                converted += 1
            rows.append({
                "sample_id": sample_id, "scene_id": record["scene"], "split": record["split"],
                "x": x, "y": y, "width": int(valid_width), "height": int(valid_height),
                "labeled_fraction": labeled_fraction,
            })
    return {"rows": rows, "converted": converted, "skipped": skipped, "rejected": rejected}


def prepare(
    dataset_root: Path, source_root: Path | None, tile_size: int, stride: int,
    workers: int, seed: int, minimum: float, overlays: int, overlay_dir: Path,
    overwrite: bool,
) -> dict:
    if tile_size < 1 or stride < 1 or workers < 1:
        raise ValueError("--tile-size, --stride and --workers must be positive")
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("--min-labeled-fraction must be between 0 and 1")
    root = dataset_root.expanduser().resolve()
    source = (source_root or root / "raw").expanduser().resolve()
    records = discover_pairs(source, seed)
    tasks = [(record, str(root), tile_size, stride, minimum, overwrite) for record in records]
    results = []
    if workers == 1:
        iterator = map(_process_scene, tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Tiling Five-Billion-Pixels", unit="scene"):
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_process_scene, tasks, chunksize=1)
            for result in tqdm(iterator, total=len(tasks), desc="Tiling Five-Billion-Pixels", unit="scene"):
                results.append(result)
    rows = [row for result in results for row in result["rows"]]
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ids = sorted(row["sample_id"] for row in rows if row["split"] == split)
        (split_dir / f"{split}.txt").write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")
    write_class_catalog(
        root / "classes.json",
        "Five-Billion-Pixels",
        CLASSES,
        default_semantic_level="lulc",
        background_id=None,
    )
    with (root / "tiles.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    overlay_dir = overlay_dir.expanduser().resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    selected = random.Random(seed).sample(rows, min(overlays, len(rows)))
    selected_ids = {row["sample_id"] for row in selected}
    for old in overlay_dir.glob("*.png"):
        if old.stem not in selected_ids:
            old.unlink()
    for row in tqdm(selected, desc="Writing overlays", unit="image"):
        sample_id = row["sample_id"]
        image = np.asarray(Image.open(root / "images" / f"{sample_id}.png").convert("RGB"))
        mask = np.asarray(Image.open(root / "masks" / f"{sample_id}.png"))
        fused = render_semantic_overlay(
            image, mask, CLASSES, background_id=None, ignore_id=255,
            alpha=0.62, image_brightness=0.82,
        )
        Image.fromarray(fused, mode="RGB").save(overlay_dir / f"{sample_id}.png")
    splits = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    metadata = {
        "format_version": "ovrs-semantic-v1", "dataset_name": "Five-Billion-Pixels",
        "image_dir": "images", "mask_dir": "masks", "classes_file": "classes.json",
        "classes_format": "ovrs-classes-v1",
        "background_id": None, "ignore_id": 255, "stored_image_channels": ["red", "green", "blue"],
        "native_source_channels": ["near infrared", "red", "green", "blue"],
        "native_masks_preserved_semantically": True, "supervision_level": "pixel",
        "optimization_by_level": {"lulc": ["region_text_alignment", "mask_localization"]},
        "pixel_supervision_exhaustive": True,
        "annotation_quality": {"label": "native", "source": "native_pixel_annotation", "confidence": 1.0},
        "class_ids_are_dataset_local": True,
        "cross_dataset_alignment": "normalized_class_text_and_prompt_texts",
        "query_scope_by_class": {str(class_id): "lulc" for class_id in CLASSES},
        "source": "https://x-ytong.github.io/project/Five-Billion-Pixels.html",
        "spatial_resolution": "4 m", "source_scene_count": len(records), "sample_count": len(rows),
        "tile_size": tile_size, "stride": stride, "min_labeled_fraction": minimum,
        "split_unit": "source_scene", "splits": splits, "overlay_dir": str(overlay_dir),
    }
    (root / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "root": str(root), "source_scenes": len(records), "samples": len(rows),
        "converted": sum(result["converted"] for result in results),
        "skipped": sum(result["skipped"] for result in results),
        "rejected_tiles": sum(result["rejected"] for result in results), "splits": splits,
        "overlay_dir": str(overlay_dir), "overlay_count": len(selected),
    }


def main() -> None:
    args = parse_args()
    summary = prepare(
        args.dataset_root, args.source_root, args.tile_size, args.stride, args.workers,
        args.seed, args.min_labeled_fraction, args.num_overlays, args.overlay_dir, args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
