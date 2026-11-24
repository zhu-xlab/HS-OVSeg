"""Download and prepare all public EarthVLSet semantic annotations."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ovrs_segmentation.datasets import write_class_catalog
from ovrs_segmentation.datasets.lmdb_store import FORMAT_VERSION, INDEX_KEY, META_KEY


REPO_ID = "Kingdrone-Junjue/EarthVLSet"
SEMANTIC_SUBSETS = {
    "earthvqa": "EarthVQA",
    "global_loveda": "EarthVL-GLOBAL",
}
DOWNLOAD_PATTERNS = [
    f"{prefix}/{split}/{directory}/*"
    for prefix in SEMANTIC_SUBSETS.values()
    for split in ("Train", "Val")
    for directory in ("images_png", "masks_png")
]
CLASSES = {
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
    8: "playground",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare all labeled EarthVLSet images for LULC training")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/Datasets/RSOD/Global-LoveDA"))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retile-lmdb", action="store_true")
    parser.add_argument("--split-val-test", action="store_true")
    parser.add_argument("--restore-splits-from-lmdb", action="store_true")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--accept-license", action="store_true")
    return parser.parse_args()


def download_semantic_subset(
    output_dir: Path,
    repo_id: str,
    revision: str,
    download_workers: int = 1,
) -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("Install project dependencies before downloading EarthVLSet") from error

    for pattern in DOWNLOAD_PATTERNS:
        print(f"Downloading {pattern}", flush=True)
        for attempt in range(4):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=[pattern],
                    local_dir=output_dir / "raw",
                    token=True,
                    max_workers=download_workers,
                )
                break
            except Exception as error:
                if "429" not in str(error) or attempt == 3:
                    raise
                delay = 60 * (attempt + 1)
                print(f"Hugging Face rate limit reached; retrying in {delay}s", flush=True)
                time.sleep(delay)


def _split_sources(raw_dir: Path, source_prefix: str, split: str) -> tuple[Path, Path]:
    source_split = "Train" if split == "train" else "Val"
    base = raw_dir / source_prefix / source_split
    return base / "images_png", base / "masks_png"


def _paired_sources(raw_dir: Path) -> dict[str, list[tuple[str, Path, Path]]]:
    pairs: dict[str, list[tuple[str, Path, Path]]] = {"train": [], "val": []}
    for subset, source_prefix in SEMANTIC_SUBSETS.items():
        for split in pairs:
            image_dir, mask_dir = _split_sources(raw_dir, source_prefix, split)
            images = {path.stem: path for path in image_dir.glob("*.png")}
            masks = {path.stem: path for path in mask_dir.glob("*.png")}
            if not images or not masks:
                raise FileNotFoundError(f"Missing {source_prefix} {split} images or masks below {raw_dir}")
            if images.keys() != masks.keys():
                missing_images = sorted(masks.keys() - images.keys())
                missing_masks = sorted(images.keys() - masks.keys())
                raise ValueError(
                    f"Unpaired {source_prefix} {split} files: "
                    f"missing_images={missing_images[:5]}, missing_masks={missing_masks[:5]}"
                )
            pairs[split].extend(
                (f"{subset}/{split}/{stem}", images[stem], masks[stem])
                for stem in sorted(images)
            )
    return pairs


def _expose_image(source: Path, destination: Path, mode: str, overwrite: bool) -> str:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return "skipped"
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as stream:
        if stream.mode != "RGB":
            stream.convert("RGB").save(destination)
            return "written"
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return "written"


def _convert_mask(task: tuple[str, str, bool]) -> str:
    source_text, destination_text, overwrite = task
    source, destination = Path(source_text), Path(destination_text)
    if destination.is_file() and not overwrite:
        return "skipped"
    with Image.open(source) as stream:
        mask = np.asarray(stream)
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel mask, got {mask.shape}: {source}")
    unknown = set(map(int, np.unique(mask))) - set(range(9))
    if unknown:
        raise ValueError(f"Unknown EarthVLSet mask IDs {sorted(unknown)}: {source}")
    converted = mask.astype(np.uint8, copy=True)
    converted[converted == 0] = 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.png")
    Image.fromarray(converted, mode="L").save(temporary)
    os.replace(temporary, destination)
    return "written"


def prepare_global_loveda(
    output_dir: Path,
    *,
    workers: int = 8,
    link_mode: str = "hardlink",
    overwrite: bool = False,
    source_repo: str = REPO_ID,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    pairs_by_split = _paired_sources(output_dir / "raw")
    split_ids = {
        split: [sample_id for sample_id, _image, _mask in pairs]
        for split, pairs in pairs_by_split.items()
    }
    all_pairs = [pair for split in ("train", "val") for pair in pairs_by_split[split]]

    image_counts = {"written": 0, "skipped": 0}
    for sample_id, image, _mask in tqdm(all_pairs, desc="Linking EarthVLSet images", unit="image"):
        result = _expose_image(image, output_dir / "images" / f"{sample_id}.png", link_mode, overwrite)
        image_counts[result] += 1

    mask_tasks = [
        (str(mask), str(output_dir / "masks" / f"{sample_id}.png"), overwrite)
        for sample_id, _image, mask in all_pairs
    ]
    mask_counts = {"written": 0, "skipped": 0}
    if workers == 1:
        results = map(_convert_mask, mask_tasks)
        for result in tqdm(results, total=len(mask_tasks), desc="Converting EarthVLSet masks", unit="mask"):
            mask_counts[result] += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_convert_mask, mask_tasks, chunksize=16)
            for result in tqdm(results, total=len(mask_tasks), desc="Converting EarthVLSet masks", unit="mask"):
                mask_counts[result] += 1

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, sample_ids in split_ids.items():
        (split_dir / f"{split}.txt").write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
    (split_dir / "test.txt").write_text("", encoding="utf-8")

    write_class_catalog(
        output_dir / "classes.json",
        "Global-LoveDA",
        CLASSES,
        default_semantic_level="lulc",
        background_id=1,
    )
    query_scope = {str(class_id): "lulc" for class_id in CLASSES if class_id != 1}
    split_counts = {split: len(sample_ids) for split, sample_ids in split_ids.items()}
    split_counts["test"] = 0
    metadata = {
        "format_version": "ovrs-semantic-v1",
        "dataset_name": "Global-LoveDA",
        "source_dataset": "EarthVLSet",
        "dataset_variant": "semantic segmentation",
        "image_dir": "images",
        "mask_dir": "masks",
        "classes_file": "classes.json",
        "classes_format": "ovrs-classes-v1",
        "background_id": 1,
        "ignore_id": 255,
        "stored_image_channels": ["red", "green", "blue"],
        "native_masks_preserved_semantically": True,
        "supervision_level": "pixel",
        "optimization_by_level": {"lulc": ["region_text_alignment", "mask_localization"]},
        "pixel_supervision_exhaustive": True,
        "annotation_quality": {"label": "native", "source": "native_pixel_annotation", "confidence": 1.0},
        "class_ids_are_dataset_local": True,
        "cross_dataset_alignment": "normalized_class_text_and_prompt_texts",
        "query_scope_by_class": query_scope,
        "mask_encoding": "single-channel PNG; 1-8 native class IDs; 255 ignore converted from native 0",
        "source": f"https://huggingface.co/datasets/{source_repo}",
        "challenge": "https://www.codabench.org/competitions/13040/",
        "semantic_subsets": SEMANTIC_SUBSETS,
        "downloaded_subset": DOWNLOAD_PATTERNS,
        "sample_count": len(all_pairs),
        "splits": split_counts,
    }
    (output_dir / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "samples": len(all_pairs),
        "splits": split_counts,
        "images": image_counts,
        "masks": mask_counts,
    }
    (output_dir / "download_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _png_bytes(image: Image.Image, mode: str) -> bytes:
    output = BytesIO()
    image.convert(mode).save(output, format="PNG", compress_level=3)
    return output.getvalue()


def split_validation_for_test(output_dir: Path, seed: int = 42) -> dict[str, int]:
    output_dir = output_dir.expanduser().resolve()
    split_dir = output_dir / "splits"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    val_ids = [line for line in val_path.read_text(encoding="utf-8").splitlines() if line]
    test_ids = [line for line in test_path.read_text(encoding="utf-8").splitlines() if line]
    if test_ids:
        raise ValueError(f"Test split is already populated: {test_path}")
    source_ids = sorted({sample_id.rsplit("__x", 1)[0] for sample_id in val_ids})
    random.Random(seed).shuffle(source_ids)
    test_sources = set(source_ids[: len(source_ids) // 2])
    new_test_ids = [sample_id for sample_id in val_ids if sample_id.rsplit("__x", 1)[0] in test_sources]
    new_val_ids = [sample_id for sample_id in val_ids if sample_id.rsplit("__x", 1)[0] not in test_sources]
    for path, sample_ids in ((val_path, new_val_ids), (test_path, new_test_ids)):
        temporary = path.with_suffix(".txt.tmp")
        temporary.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
        os.replace(temporary, path)
    metadata_path = output_dir / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["splits"] = {
        "train": len([line for line in (split_dir / "train.txt").read_text(encoding="utf-8").splitlines() if line]),
        "val": len(new_val_ids),
        "test": len(new_test_ids),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return metadata["splits"]


def restore_splits_from_lmdb(output_dir: Path, seed: int = 42) -> dict[str, int]:
    import lmdb

    output_dir = output_dir.expanduser().resolve()
    metadata_path = output_dir / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lmdb_path = output_dir / metadata["storage"]["path"]
    environment = lmdb.open(str(lmdb_path), subdir=True, readonly=True, lock=False, readahead=False)
    with environment.begin() as transaction:
        indexed_paths = json.loads(transaction.get(INDEX_KEY).decode("utf-8"))
    environment.close()

    image_prefix = str(metadata.get("image_dir", "images")).rstrip("/") + "/"
    sample_ids = [
        Path(path[len(image_prefix):]).with_suffix("").as_posix()
        for path in indexed_paths
        if path.startswith(image_prefix)
    ]

    def tile_order(sample_id: str) -> tuple[str, int, int]:
        source_id, coordinates = sample_id.rsplit("__x", 1)
        x, y = coordinates.split("_y", 1)
        return source_id, int(y), int(x)

    sample_ids.sort(key=tile_order)
    train_ids = [sample_id for sample_id in sample_ids if "/train/" in f"/{sample_id}/"]
    validation_pool = [sample_id for sample_id in sample_ids if "/val/" in f"/{sample_id}/"]
    validation_sources = sorted({sample_id.rsplit("__x", 1)[0] for sample_id in validation_pool})
    random.Random(seed).shuffle(validation_sources)
    test_sources = set(validation_sources[: len(validation_sources) // 2])
    val_ids = [sample_id for sample_id in validation_pool if sample_id.rsplit("__x", 1)[0] not in test_sources]
    test_ids = [sample_id for sample_id in validation_pool if sample_id.rsplit("__x", 1)[0] in test_sources]

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split, ids in splits.items():
        (split_dir / f"{split}.txt").write_text("".join(f"{sample_id}\n" for sample_id in ids), encoding="utf-8")
    metadata["splits"] = {split: len(ids) for split, ids in splits.items()}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata["splits"]


def retile_lmdb(output_dir: Path, tile_size: int = 512, overwrite: bool = False, split_seed: int = 42) -> dict[str, object]:
    import lmdb

    output_dir = output_dir.expanduser().resolve()
    metadata_path = output_dir / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = output_dir / metadata["storage"]["path"]
    destination = output_dir / "data.lmdb"
    backup = output_dir / "data_rgba1024.lmdb"
    temporary = output_dir / ".data_rgb512.lmdb.tmp"
    if backup.exists():
        raise FileExistsError(backup)
    if temporary.exists():
        if not overwrite:
            raise FileExistsError(temporary)
        shutil.rmtree(temporary)

    source_env = lmdb.open(str(source), subdir=True, readonly=True, lock=False, readahead=False, max_readers=32)
    with source_env.begin() as transaction:
        indexed_paths = json.loads(transaction.get(INDEX_KEY).decode("utf-8"))
    image_prefix = str(metadata.get("image_dir", "images")).rstrip("/") + "/"
    mask_prefix = str(metadata.get("mask_dir", "masks")).rstrip("/") + "/"
    images = {
        Path(path[len(image_prefix):]).with_suffix("").as_posix(): path
        for path in indexed_paths
        if path.startswith(image_prefix)
    }
    masks = {
        Path(path[len(mask_prefix):]).with_suffix("").as_posix(): path
        for path in indexed_paths
        if path.startswith(mask_prefix)
    }
    if images.keys() != masks.keys():
        raise ValueError("Global-LoveDA LMDB image and mask IDs do not match")

    split_dir = output_dir / "splits"
    split_ids = {
        split: [line for line in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line]
        for split in ("train", "val", "test")
    }
    if not split_ids["test"]:
        validation_ids = list(split_ids["val"])
        random.Random(split_seed).shuffle(validation_ids)
        midpoint = len(validation_ids) // 2
        split_ids["test"] = validation_ids[:midpoint]
        split_ids["val"] = validation_ids[midpoint:]
    source_bytes = (source / "data.mdb").stat().st_size
    destination_env = lmdb.open(str(temporary), subdir=True, map_size=int(source_bytes * 1.25) + 1_073_741_824, lock=True, meminit=False)
    output_paths: list[str] = []
    tiled_splits = {split: [] for split in split_ids}
    writer = destination_env.begin(write=True)
    written = 0
    for split, sample_ids in split_ids.items():
        for sample_id in tqdm(sample_ids, desc=f"Tiling Global-LoveDA {split}", unit="image"):
            with source_env.begin() as reader:
                image_data = reader.get(f"file:{images[sample_id]}".encode("utf-8"))
                mask_data = reader.get(f"file:{masks[sample_id]}".encode("utf-8"))
            with Image.open(BytesIO(image_data)) as stream:
                image = stream.convert("RGB").copy()
            with Image.open(BytesIO(mask_data)) as stream:
                mask = stream.convert("L").copy()
            if image.size != mask.size or image.width % tile_size or image.height % tile_size:
                raise ValueError(f"Expected aligned dimensions divisible by {tile_size}, got image={image.size}, mask={mask.size}: {sample_id}")
            for top in range(0, image.height, tile_size):
                for left in range(0, image.width, tile_size):
                    tile_id = f"{sample_id}__x{left}_y{top}"
                    image_path = f"{image_prefix}{tile_id}.png"
                    mask_path = f"{mask_prefix}{tile_id}.png"
                    box = (left, top, left + tile_size, top + tile_size)
                    writer.put(f"file:{image_path}".encode("utf-8"), _png_bytes(image.crop(box), "RGB"))
                    writer.put(f"file:{mask_path}".encode("utf-8"), _png_bytes(mask.crop(box), "L"))
                    output_paths.extend((image_path, mask_path))
                    tiled_splits[split].append(tile_id)
                    written += 1
            if written % 128 == 0:
                writer.commit()
                writer = destination_env.begin(write=True)
    writer.put(INDEX_KEY, json.dumps(output_paths, separators=(",", ":")).encode("utf-8"))
    writer.put(META_KEY, json.dumps({"format_version": FORMAT_VERSION, "dataset_name": metadata["dataset_name"], "file_count": len(output_paths)}, separators=(",", ":")).encode("utf-8"))
    writer.commit()
    destination_env.sync()
    destination_env.close()
    source_env.close()
    os.replace(source, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        os.replace(backup, source)
        raise

    for split, sample_ids in tiled_splits.items():
        path = split_dir / f"{split}.txt"
        temporary_split = path.with_suffix(".txt.tmp")
        temporary_split.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
        os.replace(temporary_split, path)
    metadata["storage"] = {"backend": "lmdb", "path": destination.name, "format_version": FORMAT_VERSION}
    metadata["stored_image_channels"] = ["red", "green", "blue"]
    metadata["tile_size"] = tile_size
    metadata["sample_count"] = written
    metadata["splits"] = {split: len(sample_ids) for split, sample_ids in tiled_splits.items()}
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return {"source_lmdb": str(backup), "output_lmdb": str(destination), "tile_size": tile_size, "samples": written, "splits": metadata["splits"]}


def main() -> None:
    args = parse_args()
    if args.restore_splits_from_lmdb:
        print(json.dumps(restore_splits_from_lmdb(args.output_dir, args.split_seed), ensure_ascii=False, indent=2))
        return
    if args.split_val_test:
        print(json.dumps(split_validation_for_test(args.output_dir, args.split_seed), ensure_ascii=False, indent=2))
        return
    if args.retile_lmdb:
        print(json.dumps(retile_lmdb(args.output_dir, args.tile_size, args.overwrite, args.split_seed), ensure_ascii=False, indent=2))
        return
    if not args.accept_license:
        raise SystemExit("EarthVLSet is academic-use-only. Rerun with --accept-license after accepting its Hugging Face terms.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only:
        download_semantic_subset(output_dir, args.repo_id, args.revision, args.download_workers)
    summary = prepare_global_loveda(output_dir, workers=args.workers, link_mode=args.link_mode, overwrite=args.overwrite, source_repo=args.repo_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
