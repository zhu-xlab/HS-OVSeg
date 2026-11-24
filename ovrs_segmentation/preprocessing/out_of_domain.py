"""Convert untouched external pixel GT into test-only image/mask pairs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import zipfile

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import SemanticSegmentationDataset, write_class_catalog


CATALOGS = {
    "WHDLD": ("building", "road", "pavement", "vegetation", "bare soil", "water"),
    "AeroScapes": ("background", "person", "bicycle", "car", "drone", "boat", "animal",
                   "obstacle", "construction", "vegetation", "road", "sky"),
    "UDD6": ("other", "facade", "road", "vegetation", "vehicle", "roof"),
}
SOURCES = {
    "WHDLD": "https://sites.google.com/view/zhouwx/dataset",
    "AeroScapes": "https://github.com/ishann/aeroscapes",
    "UDD6": "https://github.com/MarcWong/UDD",
}


@contextmanager
def source_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            yield archive.namelist(), archive.open
    else:
        # A compressed tar cannot cheaply seek between hundreds of image/mask pairs.
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".aeroscapes-") as temp:
            stage, names = Path(temp), []
            with tarfile.open(path, "r|*") as archive:
                for member in archive:
                    relative = PurePosixPath(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    if not member.isfile() or not any(member.name.startswith("aeroscapes/" + folder + "/")
                            for folder in ("JPEGImages", "SegmentationClass", "ImageSets")):
                        continue
                    target = stage.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                    names.append(member.name)
            yield names, lambda name: stage.joinpath(*PurePosixPath(name).parts).open("rb")


def safe_id(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."} or "\\" in value:
        raise ValueError(f"Unsafe external sample ID: {value!r}")
    return value


def decode_mask(label, name):
    mask = np.asarray(label)
    # Palette indices are the authors' class IDs, not palette RGB values.
    if mask.ndim != 2:
        raise ValueError(f"Expected native indexed mask for {name}")
    lower, upper = (1, 6) if name == "WHDLD" else (0, len(CATALOGS[name]) - 1)
    if not np.isin(mask, np.arange(lower, upper + 1)).all():
        raise ValueError(f"Unexpected native {name} mask IDs: {np.unique(mask).tolist()}")
    return (mask.astype(np.uint8) - lower).astype(np.uint8)


def records(name, members, read):
    if name == "WHDLD":
        return [(safe_id(PurePosixPath(n).stem), n,
                 "WHDLD/Labels/" + PurePosixPath(n).stem + ".png")
                for n in sorted(members) if n.startswith("WHDLD/Images/") and n.lower().endswith(".jpg")]
    if name == "AeroScapes":
        with read("aeroscapes/ImageSets/val.txt") as stream:
            ids = [safe_id(line.strip()) for line in stream.read().decode().splitlines() if line.strip()]
        return [(i, f"aeroscapes/JPEGImages/{i}.jpg", f"aeroscapes/SegmentationClass/{i}.png") for i in ids]
    with read("UDD/UDD6/metadata/val.txt") as stream:
        pairs = [line.split() for line in stream.read().decode().splitlines() if line.strip()]
    result = []
    for image, mask in pairs:
        stem = safe_id(PurePosixPath(image).stem)
        if image != f"val/src/{stem}.JPG" or mask != f"val/gt/{stem}.png":
            raise ValueError(f"Unexpected UDD6 manifest pair: {image}, {mask}")
        result.append((stem, "UDD/UDD6/" + image, "UDD/UDD6/" + mask))
    return result


def prepare(name, archive, output, *, expected_count=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    expected_count = expected_count if expected_count is not None else {"WHDLD": 4940, "AeroScapes": 648, "UDD6": 35}[name]
    output.parent.mkdir(parents=True, exist_ok=True)
    with source_archive(archive) as (members, read), tempfile.TemporaryDirectory(dir=output.parent, prefix=".ood-") as temp:
        stage = Path(temp) / name
        for folder in ("images", "masks", "splits"):
            (stage / folder).mkdir(parents=True)
        pairs = records(name, members, read)
        ids = [r[0] for r in pairs]
        if len(ids) != expected_count or len(set(ids)) != len(ids):
            raise ValueError(f"Expected {expected_count} unique {name} samples, got {len(ids)}")
        counts = np.zeros(len(CATALOGS[name]), dtype=np.int64)
        hashes = []
        for index, (sample_id, image_name, mask_name) in enumerate(pairs, 1):
            with read(mask_name) as stream, Image.open(stream) as label:
                mask = decode_mask(label, name)
            with read(image_name) as stream:
                content = stream.read()
            (stage / "images" / (sample_id + ".jpg")).write_bytes(content)
            with Image.open(stage / "images" / (sample_id + ".jpg")) as image:
                if image.mode != "RGB" or image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"RGB image and mask mismatch: {image_name}")
                decoded_sha = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            Image.fromarray(mask).save(stage / "masks" / (sample_id + ".png"))
            counts += np.bincount(mask.ravel(), minlength=len(counts))
            hashes.append(dict(id=sample_id, file_sha256=hashlib.sha256(content).hexdigest(), rgb_sha256=decoded_sha))
            if index % 250 == 0:
                print(f"[prepare] {name}: {index}/{len(ids)}", flush=True)
        background = 0 if name in {"AeroScapes", "UDD6"} else None
        classes = dict(enumerate(CATALOGS[name]))
        write_class_catalog(stage / "classes.json", name, classes,
                            default_semantic_level="lulc", background_id=background, ignore_id=255)
        (stage / "splits/test.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        with archive.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = dict(format_version="ovrs-semantic-v1", dataset_name=name, dataset_id=name,
            image_dir="images", mask_dir="masks", classes_file="classes.json", background_id=background,
            ignore_id=255, semantic_level="lulc", supervision_level="pixel", splits={"test": len(ids)},
            usage="out_of_domain_test_only", source_url=SOURCES[name], source_sha256=source_hash,
            original_split="entire collection" if name == "WHDLD" else "author validation",
            split_policy="Held out from this project's adaptation; author validation is named test locally for evaluation only",
            native_class_id_offset=1 if name == "WHDLD" else 0,
            background_policy="Excluded from mIoU; retained in confusion matrix" if background is not None else "All six classes evaluated",
            class_pixel_counts={str(i): int(n) for i, n in enumerate(counts)})
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (stage / "image_hashes.json").write_text(json.dumps(hashes) + "\n", encoding="utf-8")
        SemanticSegmentationDataset(stage, "test", validate_masks=True)
        stage.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CATALOGS, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.archive, args.output)), flush=True)


if __name__ == "__main__":
    main()
