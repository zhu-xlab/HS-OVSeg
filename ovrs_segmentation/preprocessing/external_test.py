"""Download and prepare held-out semantic benchmarks without changing training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import SemanticSegmentationDataset, write_class_catalog


DLRSD_URL = "https://huggingface.co/datasets/calebrob6/dlrsd/resolve/dcd622dd05f327cd0fd10951ad6cd7ba52d0e832/DLRSD.zip"
DLRSD_SHA256 = "cb49d850855a7622c51b5f09dbdf98952e064b661f35cb105e8749513e75796f"
DLRSD_CLASSES = dict(enumerate((
    "airplane", "bare soil", "buildings", "cars", "chaparral", "court", "dock", "field",
    "grass", "mobile home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water",
)))
VDD_SHA256 = "0935827859cf04ae8846a776ac8f46084f67478cc5d462cd85e04a1e73b82d78"
VDD_CLASSES = dict(enumerate(("other", "wall", "road", "vegetation", "vehicle", "roof", "water")))


def download_dlrsd(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path if path.is_file() else path.with_suffix(".zip.part")
    if candidate != path:
        urllib.request.urlretrieve(DLRSD_URL, candidate)
    with candidate.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != DLRSD_SHA256:
        raise ValueError(f"DLRSD archive SHA-256 mismatch: {candidate}")
    if candidate != path:
        candidate.rename(path)
    return path


def prepare_dlrsd(archive: Path, output: Path, *, expected_count=2100):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(dir=output.parent, prefix=".dlrsd-") as temp:
        stage = Path(temp) / "DLRSD"
        for directory in ("images", "masks", "splits"):
            (stage / directory).mkdir(parents=True)
        images = sorted(name for name in source.namelist() if name.startswith("DLRSD/Images/") and name.lower().endswith(".tif"))
        if len(images) != expected_count:
            raise ValueError(f"Expected {expected_count} DLRSD images, found {len(images)}")
        ids = []
        pixel_counts = np.zeros(17, dtype=np.int64)
        for index, name in enumerate(images, 1):
            relative = PurePosixPath(name).relative_to("DLRSD/Images")
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {name}")
            sample_id = relative.with_suffix("").as_posix()
            label_name = "DLRSD/Labels/" + relative.with_suffix(".png").as_posix()
            with source.open(name) as stream, Image.open(stream) as image:
                image = image.convert("RGB")
            with source.open(label_name) as stream, Image.open(stream) as label:
                # Author palette indices are 1..17, not RGB colors or 0-based IDs.
                mask = np.asarray(label).copy()
            if mask.ndim != 2 or not np.isin(mask, np.arange(1, 18)).all():
                raise ValueError(f"Unexpected native DLRSD mask IDs: {label_name}")
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image/mask size mismatch: {name}")
            mask = (mask - 1).astype(np.uint8)
            pixel_counts += np.bincount(mask.ravel(), minlength=17)
            for directory in ("images", "masks"):
                (stage / directory / relative).parent.mkdir(parents=True, exist_ok=True)
            image.save(stage / "images" / relative.with_suffix(".png"))
            Image.fromarray(mask).save(stage / "masks" / relative.with_suffix(".png"))
            ids.append(sample_id)
            if index % 200 == 0:
                print(f"DLRSD {index}/{len(images)}", flush=True)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate DLRSD sample IDs")
        (stage / "splits/test.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        write_class_catalog(stage / "classes.json", "DLRSD", DLRSD_CLASSES,
                            default_semantic_level="lulc", background_id=None, ignore_id=255)
        metadata = dict(format_version="ovrs-semantic-v1", dataset_name="DLRSD", dataset_id="DLRSD",
                        image_dir="images", mask_dir="masks", classes_file="classes.json",
                        background_id=None, ignore_id=255, semantic_level="lulc", supervision_level="pixel",
                        splits={"test": len(ids)}, usage="external_test_only", native_class_id_offset=1,
                        split_policy="Entire 2100-image collection held out; not an official train/test split",
                        source_url=DLRSD_URL, source_sha256=DLRSD_SHA256,
                        original_source="UC Merced imagery with author DLRSD semantic masks",
                        class_pixel_counts={str(i): int(n) for i, n in enumerate(pixel_counts)})
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        SemanticSegmentationDataset(stage, "test", validate_masks=True)
        stage.rename(output)
    return metadata


def prepare_vdd(archive: Path, output: Path, *, expected_splits=None):
    """Preserve the original VDD splits; never mix in IDD/UDD/UAVid images."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {output}")
    expected_splits = expected_splits or {"train": 280, "val": 80, "test": 40}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(dir=output.parent, prefix=".vdd-") as temp:
        stage = Path(temp) / "VDD"
        counts = {}
        all_ids = set()
        pixel_counts = np.zeros(7, dtype=np.int64)
        for split, expected in expected_splits.items():
            images = sorted(name for name in source.namelist() if name.startswith(f"VDD/{split}/src/") and name.lower().endswith(".jpg"))
            manifest = source.read(f"VDD/metadata/{split}.txt").decode("utf-8-sig").splitlines()
            if {line.split()[0] for line in manifest if line.strip()} != {name.removeprefix("VDD/") for name in images}:
                raise ValueError(f"VDD {split} images differ from the author split manifest")
            if len(images) != expected:
                raise ValueError(f"Expected {expected} original VDD {split} images, got {len(images)}")
            ids = []
            for name in images:
                stem = PurePosixPath(name).stem
                if stem in all_ids or stem in {".", ".."}:
                    raise ValueError(f"Duplicate or unsafe VDD sample ID: {stem}")
                all_ids.add(stem)
                sample_id = f"{split}/{stem}"
                image_path = stage / "images" / f"{sample_id}.jpg"
                mask_path = stage / "masks" / f"{sample_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                # The author test manifest misspells GT extensions as .JPG; actual masks are .png.
                mask_name = f"VDD/{split}/gt/{stem}.png"
                with source.open(mask_name) as stream, Image.open(stream) as label:
                    mask = np.asarray(label).copy()
                if mask.ndim != 2 or not np.isin(mask, np.arange(7)).all():
                    raise ValueError(f"Unexpected native VDD mask IDs: {mask_name}")
                with source.open(name) as stream, Image.open(stream) as image:
                    if image.size != (mask.shape[1], mask.shape[0]) or image.mode != "RGB":
                        raise ValueError(f"VDD RGB image/mask mismatch: {name}")
                with source.open(name) as stream, image_path.open("wb") as out:
                    shutil.copyfileobj(stream, out)
                Image.fromarray(mask.astype(np.uint8)).save(mask_path)
                pixel_counts += np.bincount(mask.ravel(), minlength=7)
                ids.append(sample_id)
            (stage / "splits").mkdir(exist_ok=True)
            (stage / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            counts[split] = len(ids)
            print(f"VDD {split}: {len(ids)}", flush=True)
        write_class_catalog(stage / "classes.json", "VDD", VDD_CLASSES,
                            default_semantic_level="lulc", background_id=None, ignore_id=255)
        metadata = dict(format_version="ovrs-semantic-v1", dataset_name="VDD", dataset_id="VDD",
                        image_dir="images", mask_dir="masks", classes_file="classes.json",
                        background_id=None, ignore_id=255, semantic_level="lulc", supervision_level="pixel",
                        splits=counts, usage="external_test_only", dataset_variant="original VDD; no IDD",
                        split_policy="Official original VDD split; comparison evaluates test only",
                        source_url="https://huggingface.co/datasets/RussRobin/VDD/tree/3916e1c31a8d2e561e16fdf3cc682fb85c87e330",
                        source_sha256=VDD_SHA256, source_checksum_note="Locally computed SHA-256; not an author-published checksum",
                        other_class_policy="Native other=0 is an evaluated class, not ignored/background",
                        class_pixel_counts={str(i): int(n) for i, n in enumerate(pixel_counts)})
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for split in counts:
            SemanticSegmentationDataset(stage, split, validate_masks=True)
        stage.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=("DLRSD", "VDD"), default="DLRSD")
    args = parser.parse_args()
    archive = args.data_root / "external_test_sources" / f"{args.dataset}.zip"
    if args.dataset == "DLRSD":
        download_dlrsd(archive)
        prepare = prepare_dlrsd
    else:
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != VDD_SHA256:
                raise ValueError("VDD archive differs from the verified official download")
        prepare = prepare_vdd
    print(json.dumps(prepare(archive, args.data_root / "segmentation_test" / args.dataset)))


if __name__ == "__main__":
    main()
