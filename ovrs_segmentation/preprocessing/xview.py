from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from ovrs_segmentation.datasets.class_prompts import write_class_catalog
from ovrs_segmentation.preprocessing.lmdb_dataset import pack_dataset


XVIEW_CLASSES = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck w/Box",
    26: "Truck Tractor",
    27: "Trailer",
    28: "Truck w/Flatbed",
    29: "Truck w/Liquid",
    32: "Crane Truck",
    33: "Railway Vehicle",
    34: "Passenger Car",
    35: "Cargo Car",
    36: "Flat Car",
    37: "Tank car",
    38: "Locomotive",
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
    53: "Engineering Vehicle",
    54: "Tower crane",
    55: "Container Crane",
    56: "Reach Stacker",
    57: "Straddle Carrier",
    59: "Mobile Crane",
    60: "Dump Truck",
    61: "Haul Truck",
    62: "Scraper/Tractor",
    63: "Front loader/Bulldozer",
    64: "Excavator",
    65: "Cement Mixer",
    66: "Ground Grader",
    71: "Hut/Tent",
    72: "Shed",
    73: "Building",
    74: "Aircraft Hangar",
    76: "Damaged Building",
    77: "Facility",
    79: "Construction Site",
    83: "Vehicle Lot",
    84: "Helipad",
    86: "Storage Tank",
    89: "Shipping container lot",
    91: "Shipping Container",
    93: "Pylon",
    94: "Tower",
}
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _archive_is_complete(destination):
        print(f"Using existing archive: {destination}")
        return
    offset = destination.stat().st_size if destination.is_file() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code != 416 or offset == 0:
            raise
        destination.unlink()
        return _download(url, destination)
    with response:
        content_range = response.headers.get("Content-Range", "")
        append = (
            offset > 0
            and response.status == 206
            and content_range.startswith(f"bytes {offset}-")
        )
        if not append:
            offset = 0
        total = response.headers.get("Content-Length")
        total = None if total is None else offset + int(total)
        with destination.open("ab" if append else "wb") as stream, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                progress.update(len(block))


def _archive_is_complete(path: Path) -> bool:
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as stream:
                return stream.testzip() is None
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as stream:
                stream.getmembers()
            return True
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False
    return False


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if zipfile.is_zipfile(archive):
        try:
            with zipfile.ZipFile(archive) as stream:
                for member in stream.infolist():
                    target = (root / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise ValueError(f"Unsafe archive path: {member.filename}")
                stream.extractall(root)
        except zipfile.BadZipFile as error:
            bsdtar = shutil.which("bsdtar")
            if bsdtar:
                subprocess.run([bsdtar, "-xf", str(archive), "-C", str(root)], check=True)
            else:
                extractor = shutil.which("7zz") or shutil.which("7z")
                if extractor is None:
                    raise RuntimeError(
                        "Python cannot extract this xView ZIP. Install libarchive-tools "
                        "and retry so the script can use bsdtar."
                    ) from error
                subprocess.run([extractor, "x", "-y", f"-o{root}", str(archive)], check=True)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            for member in members:
                target = (root / member.name).resolve()
                if member.issym() or member.islnk() or (
                    target != root and root not in target.parents
                ):
                    raise ValueError(f"Unsafe archive path: {member.name}")
            stream.extractall(root, members=members)
        return
    raise ValueError(f"Unsupported xView archive: {archive}")


def _find_raw_inputs(root: Path) -> tuple[dict[str, Path], Path]:
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in images:
                raise ValueError(f"Duplicate xView image name: {path.name}")
            images[path.name] = path
    labels = list(root.rglob("xView_train.geojson"))
    if len(labels) != 1:
        raise FileNotFoundError(
            f"Expected one xView_train.geojson below {root}, found {len(labels)}"
        )
    return images, labels[0]


def _load_boxes(
    geojson_path: Path,
    images: dict[str, Path],
) -> dict[str, list[tuple[str, int, float, float, float, float]]]:
    document = json.loads(geojson_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[str, int, float, float, float, float]]] = defaultdict(list)
    for index, feature in enumerate(document["features"]):
        properties = feature["properties"]
        image_name = str(properties["image_id"])
        bounds = properties.get("bounds_imcoords")
        if not bounds or image_name not in images:
            continue
        class_id = int(properties["type_id"])
        if class_id not in XVIEW_CLASSES:
            continue
        coordinates = [float(value) for value in str(bounds).split(",")]
        if len(coordinates) != 4:
            raise ValueError(f"Invalid xView bounds_imcoords: {bounds!r}")
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            continue
        instance_id = str(properties.get("feature_id", feature.get("id", index)))
        grouped[image_name].append((instance_id, class_id, x1, y1, x2, y2))
    return dict(grouped)


def _balanced_source_splits(
    boxes_by_image: dict[str, list[tuple[str, int, float, float, float, float]]],
    seed: int,
) -> dict[str, str]:
    names = sorted(boxes_by_image)
    target_sizes = {
        "train": len(names) - 2 * round(len(names) * 0.1),
        "val": round(len(names) * 0.1),
        "test": round(len(names) * 0.1),
    }
    image_classes = {
        name: {box[1] for box in boxes}
        for name, boxes in boxes_by_image.items()
    }
    frequencies = Counter(
        class_id for classes in image_classes.values() for class_id in classes
    )
    split_names = ("train", "val", "test")
    target_classes = {split: {} for split in split_names}
    for class_id, frequency in frequencies.items():
        held_out = max(1, round(frequency * 0.1)) if frequency >= 3 else 0
        target_classes["val"][class_id] = held_out
        target_classes["test"][class_id] = held_out
        target_classes["train"][class_id] = frequency - 2 * held_out
    assigned_sizes = Counter()
    assigned_classes = {split: Counter() for split in split_names}

    def order_key(name: str) -> tuple[float, str]:
        rarity = sum(1.0 / frequencies[class_id] for class_id in image_classes[name])
        tie = hashlib.sha1(f"{seed}:{name}".encode()).hexdigest()
        return -rarity, tie

    assignments = {}
    for name in sorted(names, key=order_key):
        available = [
            split for split in split_names if assigned_sizes[split] < target_sizes[split]
        ]

        def need(split: str) -> tuple[float, float]:
            class_need = sum(
                max(target_classes[split][class_id] - assigned_classes[split][class_id], 0)
                / max(target_classes[split][class_id], 1)
                for class_id in image_classes[name]
            )
            size_need = target_sizes[split] - assigned_sizes[split]
            return class_need, size_need / max(target_sizes[split], 1)

        split = max(available, key=need)
        assignments[name] = split
        assigned_sizes[split] += 1
        assigned_classes[split].update(image_classes[name])
    return assignments


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def _chip_source(task: tuple[Any, ...]) -> dict[str, Any]:
    image_name, image_path, boxes, split, image_dir, annotation_dir, tile_size, stride = task
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    clipped_boxes = []
    for instance_id, class_id, x1, y1, x2, y2 in boxes:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 > x1 and y2 > y1:
            clipped_boxes.append((instance_id, class_id, x1, y1, x2, y2))
    boxes = clipped_boxes
    windows = [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in _positions(height, tile_size, stride)
        for x in _positions(width, tile_size, stride)
    ]
    active_windows: set[int] = set()
    for box in boxes:
        _, _, x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        def score(index: int) -> tuple[float, float]:
            wx1, wy1, wx2, wy2 = windows[index]
            visible = max(0.0, min(x2, wx2) - max(x1, wx1)) * max(
                0.0, min(y2, wy2) - max(y1, wy1)
            )
            distance = abs(center_x - (wx1 + wx2) / 2) + abs(center_y - (wy1 + wy2) / 2)
            return visible, -distance

        candidates = [
            index
            for index, (wx1, wy1, wx2, wy2) in enumerate(windows)
            if wx1 <= center_x <= wx2 and wy1 <= center_y <= wy2
        ]
        active_windows.add(max(candidates, key=score))

    sample_ids = []
    instances = 0
    for index in sorted(active_windows):
        x0, y0, x2, y2 = windows[index]
        sample_id = f"{Path(image_name).stem}__x{x0}_y{y0}"
        chip = image.crop((x0, y0, x2, y2))
        if chip.size != (tile_size, tile_size):
            canvas = Image.new("RGB", (tile_size, tile_size))
            canvas.paste(chip, (0, 0))
            chip = canvas
        chip.save(image_dir / f"{sample_id}.png", compress_level=3)
        annotations = []
        for instance_id, class_id, bx1, by1, bx2, by2 in boxes:
            clipped = [
                max(0.0, bx1 - x0),
                max(0.0, by1 - y0),
                min(float(tile_size), bx2 - x0),
                min(float(tile_size), by2 - y0),
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            annotations.append(
                {
                    "instance_id": instance_id,
                    "dataset_class_id": class_id,
                    "class_text": XVIEW_CLASSES[class_id],
                    "bbox_xyxy": [round(value, 2) for value in clipped],
                }
            )
        record = {
            "format_version": "ovrs-detection-annotations-v1",
            "sample_id": sample_id,
            "source_image": image_name,
            "image_size": [tile_size, tile_size],
            "annotations": annotations,
        }
        (annotation_dir / f"{sample_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        sample_ids.append(sample_id)
        instances += len(annotations)
    return {"split": split, "sample_ids": sample_ids, "instances": instances}


def prepare_xview(
    raw_root: Path,
    output_dir: Path,
    *,
    tile_size: int = 512,
    stride: int = 384,
    workers: int = 8,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    if overwrite:
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(annotation_dir, ignore_errors=True)
    image_dir.mkdir(exist_ok=True)
    annotation_dir.mkdir(exist_ok=True)

    images, geojson = _find_raw_inputs(raw_root)
    if not images:
        raise FileNotFoundError(f"No xView images found below {raw_root}")
    boxes_by_image = _load_boxes(geojson, images)
    if not boxes_by_image:
        raise ValueError("No xView annotations match the extracted training images")
    assignments = _balanced_source_splits(boxes_by_image, seed)
    tasks = [
        (
            name,
            images[name],
            boxes,
            assignments[name],
            image_dir,
            annotation_dir,
            tile_size,
            stride,
        )
        for name, boxes in sorted(boxes_by_image.items())
    ]
    if workers == 1:
        results = list(
            tqdm(
                map(_chip_source, tasks),
                total=len(tasks),
                desc="xView source images",
                unit="image",
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                tqdm(
                    executor.map(_chip_source, tasks),
                    total=len(tasks),
                    desc="xView source images",
                    unit="image",
                )
            )

    splits = {split: [] for split in ("train", "val", "test")}
    for result in results:
        splits[result["split"]].extend(result["sample_ids"])
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for split, sample_ids in splits.items():
        sample_ids.sort()
        (split_dir / f"{split}.txt").write_text(
            "\n".join(sample_ids) + "\n", encoding="utf-8"
        )

    native_classes = {str(class_id): name for class_id, name in XVIEW_CLASSES.items()}
    (output_dir / "classes.before-text-prompts.json").write_text(
        json.dumps(native_classes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_class_catalog(
        output_dir / "classes.json",
        "xView",
        XVIEW_CLASSES,
        default_semantic_level="object",
        background_id=None,
    )
    metadata = {
        "format_version": "ovrs-semantic-v1",
        "dataset_name": "xView",
        "native_supervision": "object_detection",
        "semantic_level": "object",
        "supervision_level": "box",
        "image_dir": "images",
        "annotations_dir": "annotations",
        "classes_file": "classes.json",
        "classes_format": "ovrs-classes-v1",
        "background_id": None,
        "ignore_id": 255,
        "image_size": [tile_size, tile_size],
        "tile_stride": stride,
        "source_images": len(tasks),
        "source_instance_count": sum(len(boxes) for boxes in boxes_by_image.values()),
        "sample_count": sum(len(values) for values in splits.values()),
        "chip_instance_count": sum(result["instances"] for result in results),
        "splits": {split: len(values) for split, values in splits.items()},
        "split_unit": "source_image",
        "license": "CC BY-NC-SA 4.0",
        "source": "https://challenge.xviewdataset.org",
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare xView for OVRS Object training"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/Datasets/RSOD/xView"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--train-images-archive", type=Path)
    parser.add_argument("--train-labels-archive", type=Path)
    parser.add_argument("--train-images-url")
    parser.add_argument("--train-labels-url")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--pack-lmdb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-prepared-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-work", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_license:
        raise SystemExit("Pass --accept-license after accepting xView CC BY-NC-SA 4.0")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else work / "raw"
    source_mode = "extracted_directory" if args.raw_root else "local_archives"
    if args.raw_root is None:
        downloads = work / "downloads"
        image_archive = args.train_images_archive
        label_archive = args.train_labels_archive
        if args.train_images_url:
            source_mode = "official_personalized_urls"
            image_archive = downloads / "train_images.zip"
            _download(args.train_images_url, image_archive)
        if args.train_labels_url:
            source_mode = "official_personalized_urls"
            label_archive = downloads / "train_labels.zip"
            _download(args.train_labels_url, label_archive)
        if image_archive is None or label_archive is None:
            raise SystemExit(
                "Provide --raw-root, both local archives, or both official download URLs"
            )
        _extract(image_archive.expanduser().resolve(), raw_root)
        _extract(label_archive.expanduser().resolve(), raw_root)

    metadata = prepare_xview(
        raw_root,
        output,
        tile_size=args.tile_size,
        stride=args.stride,
        workers=args.workers,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    if args.pack_lmdb:
        metadata["lmdb"] = pack_dataset(
            output,
            overwrite=args.overwrite,
            remove_source_files=args.remove_prepared_files,
        )
    if args.remove_work and work.is_dir():
        shutil.rmtree(work)
    (output / "download_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "xView",
                "source": "https://challenge.xviewdataset.org",
                "source_mode": source_mode,
                "license": "CC BY-NC-SA 4.0",
                "train_images": True,
                "train_labels": True,
                "unlabeled_official_validation_images": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
