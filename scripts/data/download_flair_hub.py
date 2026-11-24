#!/usr/bin/env python3
"""Download the FLAIR-HUB aerial RGB/CoSIA subset used by OVRS."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Iterable

from tqdm import tqdm

import _bootstrap  # noqa: F401
from ovrs_segmentation.datasets import write_class_catalog


DATASET_REPO = "IGNF/FLAIR-HUB"
SPLIT_REPO = "IGNF/MAESTRO_FLAIR-HUB_base"
SPLIT_FILES = {
    "train": "dataset_splits/FLAIR-HUB/train.csv",
    "val": "dataset_splits/FLAIR-HUB/val.csv",
    "test": "dataset_splits/FLAIR-HUB/test.csv",
}
IMAGE_TOKEN = "_AERIAL_RGBI.zip"
MASK_TOKEN = "_AERIAL_LABEL-COSIA.zip"

# Native FLAIR-HUB CoSIA IDs. ID 0 is a real class, not background.
COSIA_CLASSES = {
    0: "building",
    1: "greenhouse",
    2: "swimming pool",
    3: "impervious surface",
    4: "pervious surface",
    5: "bare soil",
    6: "water",
    7: "snow",
    8: "herbaceous vegetation",
    9: "agricultural land",
    10: "plowed land",
    11: "vineyard",
    12: "deciduous",
    13: "coniferous",
    14: "brushwood",
    15: "clear cut",
    16: "ligneous",
    17: "mixed",
    18: "undefined",
}
COSIA_IGNORED_CLASS_IDS = frozenset({15, 16, 17, 18})
TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FLAIR-HUB AERIAL_RGBI and AERIAL_LABEL-COSIA, then "
            "write three-channel RGB inputs and expose native labels through ovrs-semantic-v1"
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/Datasets/RSOD/FLAIR-HUB"))
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--departments",
        nargs="*",
        help="Optional department prefixes, e.g. D004 D005. Omit to download every department.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How masks/ references native label TIFFs (default: hardlink).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="CPU processes used to write three-channel RGB TIFFs (default: up to 8).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Skip network downloads and rebuild the unified view from output-dir/raw.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing unified links and re-extract completed archives.",
    )
    parser.add_argument("--cleanup-only", action="store_true", help="Delete prepared-data sources that training does not read, then exit.")
    return parser.parse_args()


def _department(path: str) -> str:
    return PurePosixPath(path).name.split("-", 1)[0].upper()


def _selected_archives(files: Iterable[str], departments: set[str] | None) -> list[str]:
    selected = []
    for path in files:
        name = PurePosixPath(path).name
        if not (
            name.endswith(IMAGE_TOKEN)
            or name.endswith(MASK_TOKEN)
        ):
            continue
        if departments is not None and _department(path) not in departments:
            continue
        selected.append(path)
    return sorted(selected)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in archive {archive}: {member.filename}")
        stream.extractall(destination)


def _download_and_extract(
    repo_id: str,
    revision: str,
    archives: Iterable[str],
    raw_dir: Path,
    overwrite: bool,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("Install project dependencies with: python -m pip install -e .") from error

    raw_dir.mkdir(parents=True, exist_ok=True)
    for repo_path in tqdm(list(archives), desc="FLAIR-HUB archives", unit="archive"):
        archive_stem = Path(PurePosixPath(repo_path).name).stem
        destination = raw_dir / archive_stem
        marker = destination / ".extraction-complete"
        if marker.is_file() and not overwrite:
            continue
        if destination.exists() and overwrite:
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        # One temporary cache per archive: no .cache/huggingface tree remains in output-dir.
        with tempfile.TemporaryDirectory(prefix="flair-hub-") as temporary:
            downloaded = Path(hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                repo_type="dataset",
                revision=revision,
                cache_dir=temporary,
            ))
            _safe_extract(downloaded, destination)
        marker.write_text(repo_path + "\n", encoding="utf-8")


def _download_splits(output_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("Install project dependencies with: python -m pip install -e .") from error

    destination = output_dir / "official_splits"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flair-splits-") as temporary:
        for split, repo_path in SPLIT_FILES.items():
            source = Path(hf_hub_download(
                repo_id=SPLIT_REPO,
                filename=repo_path,
                repo_type="model",
                cache_dir=temporary,
            ))
            shutil.copy2(source, destination / f"{split}.csv")


def _modality_roots(raw_dir: Path, token: str) -> list[Path]:
    suffix = token.removesuffix(".zip")
    return sorted(path for path in raw_dir.iterdir() if path.is_dir() and path.name.endswith(suffix))


def _tiffs(roots: Iterable[Path]) -> list[Path]:
    return sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def _normal_key(stem: str) -> str:
    value = stem.strip()
    value = re.sub(r"^(?:IMG|MSK|MASK|LABEL)[_-]?", "", value, flags=re.IGNORECASE)
    # Official FLAIR-HUB files encode the modality inside the filename, e.g.
    # D012-2019_AERIAL_RGBI_AF-S1-27_5-10.tif and
    # D012-2019_AERIAL_LABEL-COSIA_AF-S1-27_5-10.tif.  Their shared patch ID
    # is D012-2019_AF-S1-27_5-10.
    value = re.sub(
        r"_(?:AERIAL_RGBI|AERIAL_LABEL-COSIA|SPOT_RGBI)(?=_|$)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.lower()


def _unique_index(paths: Iterable[Path], kind: str) -> tuple[dict[str, Path], dict[str, Path]]:
    exact: dict[str, Path] = {}
    normalized: dict[str, Path] = {}
    duplicate_normalized: set[str] = set()
    for path in paths:
        if path.stem in exact:
            raise ValueError(f"Duplicate {kind} filename stem {path.stem!r}: {path}")
        exact[path.stem] = path
        key = _normal_key(path.stem)
        if key in normalized:
            duplicate_normalized.add(key)
        else:
            normalized[key] = path
    for key in duplicate_normalized:
        normalized.pop(key, None)
    return exact, normalized


def _pair_native_files(raw_dir: Path) -> list[tuple[str, Path, Path]]:
    images = _tiffs(_modality_roots(raw_dir, IMAGE_TOKEN))
    masks = _tiffs(_modality_roots(raw_dir, MASK_TOKEN))
    if not images or not masks:
        raise FileNotFoundError(
            f"No extracted RGBI images or CoSIA masks below {raw_dir}; run without --prepare-only first"
        )
    masks_exact, masks_normalized = _unique_index(masks, "mask")
    pairs: list[tuple[str, Path, Path]] = []
    used_masks: set[Path] = set()
    used_sample_ids: set[str] = set()
    for image in images:
        mask = masks_exact.get(image.stem) or masks_normalized.get(_normal_key(image.stem))
        if mask is None:
            continue
        sample_id = image.stem
        if sample_id in used_sample_ids:
            raise ValueError(f"Duplicate output sample ID {sample_id!r}; source filenames are not unique")
        pairs.append((sample_id, image, mask))
        used_sample_ids.add(sample_id)
        used_masks.add(mask)
    if not pairs:
        image_examples = ", ".join(
            f"{path.name} -> {_normal_key(path.stem)}" for path in images[:3]
        )
        mask_examples = ", ".join(
            f"{path.name} -> {_normal_key(path.stem)}" for path in masks[:3]
        )
        raise ValueError(
            "No RGBI/CoSIA TIFF pairs were found. "
            f"Image examples: [{image_examples}]. Mask examples: [{mask_examples}]"
        )
    missing_images = len(images) - len(pairs)
    unused_masks = len(masks) - len(used_masks)
    if missing_images or unused_masks:
        print(f"Warning: {missing_images} images and {unused_masks} masks could not be paired")
    return pairs


def _replace_destination(destination: Path, overwrite: bool) -> bool:
    if not destination.exists() and not destination.is_symlink():
        return True
    if not overwrite:
        return False
    destination.unlink()
    return True


def _expose(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if not _replace_destination(destination, overwrite):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _read_first_three_channels(path: Path):
    try:
        import tifffile
    except ImportError as error:
        raise ImportError("tifffile is required; run: python -m pip install -e .") from error
    image = tifffile.imread(path)
    if image.ndim != 3:
        raise ValueError(f"Expected a multi-channel TIFF, got shape {image.shape}: {path}")
    if image.shape[-1] >= 3:
        return image[..., :3]
    if image.shape[0] >= 3:
        return image[:3, ...].transpose(1, 2, 0)
    raise ValueError(f"TIFF has fewer than three channels, got shape {image.shape}: {path}")


def _stored_rgb(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import tifffile

        shape = tuple(tifffile.TiffFile(path).series[0].shape)
    except (ImportError, OSError, ValueError):
        return False
    return len(shape) == 3 and (shape[-1] == 3 or (shape[0] == 3 and shape[-1] != 3))


def _check_tiff_decoder(source: Path) -> None:
    """Fail before spawning workers when the source uses an optional codec."""
    try:
        import tifffile
    except ImportError as error:
        raise ImportError("tifffile is required; run: python -m pip install -e .") from error
    with tifffile.TiffFile(source) as stream:
        compression = stream.pages[0].compression.name.upper()
    if compression == "LZW" and importlib.util.find_spec("imagecodecs") is None:
        raise ImportError(
            "FLAIR-HUB uses LZW-compressed TIFFs. Install the decoder first: "
            "python -m pip install 'imagecodecs>=2024.1' "
            "(or reinstall the project with: python -m pip install -e .)."
        )


def _convert_rgb_task(task: tuple[str, str, bool]) -> str:
    source_text, destination_text, overwrite = task
    source, destination = Path(source_text), Path(destination_text)
    if destination.exists() or destination.is_symlink():
        if not overwrite and _stored_rgb(destination):
            return "skipped"
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = _read_first_three_channels(source)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.tif")
    try:
        import tifffile

        tifffile.imwrite(temporary, rgb, photometric="rgb", compression="deflate", metadata=None)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "converted"


def _write_rgb_images(
    sources: dict[str, Path],
    destination_dir: Path,
    overwrite: bool,
    workers: int,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    tasks = [
        (str(source), str(destination_dir / f"{sample_id}.tif"), overwrite)
        for sample_id, source in sorted(sources.items())
    ]
    if tasks:
        _check_tiff_decoder(Path(tasks[0][0]))
    counts = {"converted": 0, "skipped": 0}
    if workers == 1:
        results = map(_convert_rgb_task, tasks)
        for result in tqdm(results, total=len(tasks), desc=f"Writing {destination_dir.name}", unit="image"):
            counts[result] += 1
        return counts
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_convert_rgb_task, tasks, chunksize=8)
        for result in tqdm(results, total=len(tasks), desc=f"Writing {destination_dir.name}", unit="image"):
            counts[result] += 1
    return counts


def _csv_values(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            return set()
        preferred = ("patch_id", "sample_id", "image_id", "aerial_id", "name")
        column = next((name for name in preferred if name in reader.fieldnames), reader.fieldnames[0])
        values = set()
        for row in reader:
            raw = str(row.get(column, "")).strip()
            if raw:
                values.add(Path(raw).stem)
        return values


def _write_splits(output_dir: Path, pairs: list[tuple[str, Path, Path]]) -> dict[str, int]:
    source_dir = output_dir / "official_splits"
    if not all((source_dir / f"{split}.csv").is_file() for split in SPLIT_FILES):
        raise FileNotFoundError(
            f"Official split CSVs are absent from {source_dir}; rerun without --prepare-only"
        )
    memberships = {split: _csv_values(source_dir / f"{split}.csv") for split in SPLIT_FILES}
    normalized_memberships = {
        split: {_normal_key(value) for value in values}
        for split, values in memberships.items()
    }
    assigned: dict[str, list[str]] = {split: [] for split in SPLIT_FILES}
    unassigned: list[str] = []
    for sample_id, image, mask in pairs:
        aliases = {sample_id, image.stem, mask.stem}
        normalized_aliases = {_normal_key(value) for value in aliases}
        matches = [
            split
            for split, values in memberships.items()
            if aliases & values or normalized_aliases & normalized_memberships[split]
        ]
        if len(matches) == 1:
            assigned[matches[0]].append(sample_id)
        elif not matches:
            unassigned.append(sample_id)
        else:
            raise ValueError(f"Sample {sample_id!r} occurs in multiple official splits: {matches}")
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, sample_ids in assigned.items():
        (split_dir / f"{split}.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in sorted(sample_ids)), encoding="utf-8"
        )
    (split_dir / "unassigned.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in sorted(unassigned)), encoding="utf-8"
    )
    counts = {split: len(values) for split, values in assigned.items()}
    counts["unassigned"] = len(unassigned)
    return counts


def _prepare(output_dir: Path, mode: str, overwrite: bool, workers: int = 1) -> dict[str, object]:
    pairs = _pair_native_files(output_dir / "raw")
    aerial_by_sample = {sample_id: image for sample_id, image, _mask in pairs}
    aerial_conversion = _write_rgb_images(
        aerial_by_sample, output_dir / "images", overwrite, workers
    )
    for sample_id, _image, mask in tqdm(pairs, desc="Linking native masks", unit="mask"):
        _expose(mask, output_dir / "masks" / f"{sample_id}{mask.suffix.lower()}", mode, overwrite)
    split_counts = _write_splits(output_dir, pairs)
    write_class_catalog(
        output_dir / "classes.json",
        "FLAIR-HUB",
        COSIA_CLASSES,
        default_semantic_level="lulc",
        background_id=None,
        excluded_class_ids=sorted(COSIA_IGNORED_CLASS_IDS),
        exclusion_reason="ignored by the official 15-class FLAIR-HUB benchmark",
    )
    metadata = {
        "format_version": "ovrs-semantic-v1",
        "dataset_name": "FLAIR-HUB",
        "dataset_variant": "CoSIA",
        "image_dir": "images",
        "mask_dir": "masks",
        "classes_file": "classes.json",
        "classes_format": "ovrs-classes-v1",
        "background_id": None,
        "ignore_id": 255,
        "ignored_native_class_ids": sorted(COSIA_IGNORED_CLASS_IDS),
        "source_image_channels": ["red", "green", "blue", "near infrared"],
        "stored_image_channels": ["red", "green", "blue"],
        "spatial_resolution_m": 0.2,
        "native_masks_preserved": True,
        "supervision_level": "pixel",
        "optimization_by_level": {
            "lulc": ["region_text_alignment", "mask_localization"],
        },
        "pixel_supervision_exhaustive": True,
        "annotation_quality": {
            "label": "native",
            "source": "native_pixel_annotation",
            "confidence": 1.0,
        },
        "class_ids_are_dataset_local": True,
        "cross_dataset_alignment": "normalized_class_text_and_prompt_texts",
        "query_scope_by_class": {
            str(class_id): "lulc"
            for class_id in COSIA_CLASSES
            if class_id not in COSIA_IGNORED_CLASS_IDS
        },
        "mask_encoding": "native single-channel CoSIA class-ID GeoTIFF",
        "source": DATASET_REPO,
        "official_split": "split_1",
        "official_split_source": SPLIT_REPO,
        "sample_count": len(pairs),
        "aerial_rgb_conversion": aerial_conversion,
        "split_counts": split_counts,
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "output_dir": str(output_dir.resolve()),
        "sample_count": len(pairs),
        "aerial_rgb_conversion": aerial_conversion,
        "split_counts": split_counts,
        "link_mode": mode,
        "native_masks_preserved": True,
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _cleanup_unused(output_dir: Path) -> dict[str, object]:
    required = [output_dir / "images", output_dir / "masks", output_dir / "splits", output_dir / "classes.json", output_dir / "dataset.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared dataset is incomplete; missing: {', '.join(missing)}")
    if any(path.is_symlink() for path in (output_dir / "masks").iterdir()):
        raise RuntimeError("masks/ contains symlinks into raw/; rebuild with --link-mode hardlink before cleanup")
    removed = []
    for name in ("raw", "spot_images", "official_splits"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(name)
    return {"output_dir": str(output_dir), "removed": removed}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cleanup_only:
        print(json.dumps(_cleanup_unused(output_dir), ensure_ascii=False, indent=2))
        return
    departments = None
    if args.departments:
        departments = {value.upper() for value in args.departments}
    if not args.prepare_only:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise ImportError("Install project dependencies with: python -m pip install -e .") from error
        files = HfApi().list_repo_files(
            repo_id=args.repo_id, repo_type="dataset", revision=args.revision
        )
        archives = _selected_archives(files, departments)
        if not archives:
            raise FileNotFoundError("No matching RGBI/CoSIA archives were found")
        image_count = sum(PurePosixPath(path).name.endswith(IMAGE_TOKEN) for path in archives)
        mask_count = sum(PurePosixPath(path).name.endswith(MASK_TOKEN) for path in archives)
        print(f"Selected {image_count} aerial RGBI and {mask_count} CoSIA archives")
        _download_splits(output_dir)
        _download_and_extract(args.repo_id, args.revision, archives, output_dir / "raw", args.overwrite)
    manifest = _prepare(output_dir, args.link_mode, args.overwrite, args.workers)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
