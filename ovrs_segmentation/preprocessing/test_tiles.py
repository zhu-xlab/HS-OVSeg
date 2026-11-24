"""Select a fixed diverse test subset after cropping native image/GT pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import SemanticSegmentationDataset
from .test_subset import select_diverse, source_group


def prepare(source: Path, output: Path, *, tile_size=518, size=2000, seed=20261028):
    source, output = source.resolve(), output.resolve()
    if tile_size < 1 or size < 1 or source == output or source in output.parents:
        raise ValueError("Use positive sizes and a separate output directory")
    split = "test_full" if (source / "splits/test_full.txt").exists() else "test"
    identity = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("dataset.json", "classes.json", f"splits/{split}.txt")}
    settings = dict(protocol="native-tiles-before-diverse-selection-v1", tile_size=tile_size,
                    limit=size, seed=seed, source_identity=identity)
    if output.exists():
        prior = json.loads((output / "test_tiles.json").read_text())
        if any(prior.get(k) != v for k, v in settings.items()):
            raise ValueError("Output already contains a different tile selection")
        return prior
    dataset = SemanticSegmentationDataset(source, split, validate_masks=False)
    class_ids = list(dataset.class_text_by_id)
    candidates, histograms = [], []
    for sample_id in dataset.sample_ids:
        mask = dataset._read_mask(dataset._masks[sample_id])
        dataset._validate_mask(sample_id, mask)
        height, width = mask.shape
        with Image.open(dataset._images[sample_id]) as image:
            if image.size != (width, height):
                raise ValueError(f"Image/GT size mismatch: {sample_id}")
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                crop = mask[top:top + tile_size, left:left + tile_size]
                counts = np.bincount(crop.ravel().astype(np.int64), minlength=max(class_ids) + 1)[class_ids]
                if not counts.sum():
                    continue
                candidates.append(dict(id=f"{sample_id}__x{left}_y{top}", source_id=sample_id,
                    left=left, top=top, width=crop.shape[1], height=crop.shape[0],
                    source_width=width, source_height=height))
                histograms.append(counts)
    if not candidates:
        raise ValueError("No tiles with labeled pixels")
    counts = np.asarray(histograms)
    selected = (select_diverse([r["id"] for r in candidates], counts, size, seed)
                if len(candidates) > size else list(range(len(candidates))))
    if np.any((counts.sum(axis=0) > 0) & (counts[selected].sum(axis=0) == 0)):
        raise ValueError("Selected tiles lost a class present in the candidate pool")
    rows = sorted((candidates[i] for i in selected), key=lambda r: r["id"])
    summary = dict(settings, dataset=dataset.dataset_name, original_samples=len(dataset),
        candidate_tiles=len(candidates), samples=len(rows),
        source_groups_selected=len({source_group(r["id"]) for r in rows}),
        selection_uses="original-image quotas and GT class histograms; no predictions",
        spatial_policy="non-overlapping native grid; RGB zero padding; GT ignore padding",
        selected_class_image_counts={str(c): int(n) for c, n in zip(class_ids, (counts[selected] > 0).sum(0))})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".test-tiles-") as temporary:
        stage = Path(temporary) / output.name
        for directory in ("images", "masks", "splits"):
            (stage / directory).mkdir(parents=True)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["source_id"]].append(row)
        for index, (sample_id, tiles) in enumerate(grouped.items(), 1):
            with Image.open(dataset._images[sample_id]) as image:
                rgb = np.asarray(image.convert("RGB"))
            mask = dataset._read_mask(dataset._masks[sample_id])
            for row in tiles:
                x, y, w, h = (row[k] for k in ("left", "top", "width", "height"))
                image_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                mask_tile = np.full((tile_size, tile_size), dataset.ignore_id, dtype=np.uint8)
                image_tile[:h, :w] = rgb[y:y + h, x:x + w]
                mask_tile[:h, :w] = mask[y:y + h, x:x + w]
                for directory, array in (("images", image_tile), ("masks", mask_tile)):
                    target = (stage / directory / (row["id"] + ".png")).resolve()
                    if not target.is_relative_to((stage / directory).resolve()):
                        raise ValueError("Unsafe tile ID")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(target, compress_level=3)
            if index % 100 == 0 or index == len(grouped):
                print(f"[test-tiles] {dataset.dataset_name}: {index}/{len(grouped)} source images", flush=True)
        shutil.copyfile(source / "classes.json", stage / "classes.json")
        (stage / "splits/test.txt").write_text("\n".join(r["id"] for r in rows) + "\n", encoding="utf-8", newline="\n")
        for name, records in (("tile_manifest.jsonl", rows), ("tile_candidates.jsonl", candidates)):
            (stage / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
        metadata = dict(dataset.metadata)
        metadata.update(image_dir="images", mask_dir="masks", splits={"test": len(rows)},
            split_counts={"test": len(rows)}, usage="out_of_domain_test_only",
            class_pixel_counts={str(c): int(n) for c, n in zip(class_ids, counts[selected].sum(0))},
            evaluation_subset=summary, split_policy="Native 518-grid tiles selected after cropping; source test images only")
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
        (stage / "test_tiles.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        SemanticSegmentationDataset(stage, "test", validate_masks=True)
        for attempt in range(6):
            try:
                stage.rename(output)
                break
            except PermissionError:
                if attempt < 5:
                    time.sleep(1)
                    continue
                # Windows sync clients may hold directory handles after files close.
                shutil.copytree(stage, output, ignore=shutil.ignore_patterns("test_tiles.json"))
                SemanticSegmentationDataset(output, "test", validate_masks=True)
                shutil.copyfile(stage / "test_tiles.json", output / "test_tiles.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile-size", type=int, default=518)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, tile_size=args.tile_size, size=args.size, seed=args.seed)))
