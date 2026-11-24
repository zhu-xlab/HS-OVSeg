#!/usr/bin/env python3
"""Convert supported native segmentation datasets to ovrs-semantic-v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import _bootstrap  # noqa: F401

from ovrs_segmentation.datasets import write_class_catalog
from ovrs_segmentation.visualization import render_semantic_overlay


Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _palette(hex_colors: list[str]) -> dict[tuple[int, int, int], int]:
    return {
        tuple(bytes.fromhex(color.lstrip("#"))): index
        for index, color in enumerate(hex_colors)
    }


ADAPTERS = {
    "openearthmap": {
        "name": "OpenEarthMap",
        "classes": {
            1: "bareland", 2: "rangeland", 3: "developed space", 4: "road",
            5: "tree", 6: "water", 7: "agriculture land", 8: "building",
        },
        # OpenEarthMap's native labels are 0=unknown and 1..8=land-cover
        # classes. Keep the official foreground IDs and normalize unknown to
        # the unified ignore ID. RGB labels use the same codes via this map.
        "palette": {
            (0, 0, 0): 255,
            **{
                color: class_id + 1
                for color, class_id in _palette([
                    "800000", "00FF24", "949494", "FFFFFF",
                    "226126", "0045FF", "4BB549", "DE1F07",
                ]).items()
            },
        },
        "value_remap": {0: 255},
        "background_id": None,
        "split_ratios": (0.7, 0.15, 0.15),
        "source": "https://zenodo.org/records/7223446",
        "resolution": "0.25-0.5 m",
    },
    "loveda": {
        "name": "LoveDA",
        "classes": {1: "background", 2: "building", 3: "road", 4: "water", 5: "barren", 6: "forest", 7: "agriculture"},
        "value_remap": {0: 255},
        "background_id": 1,
        "split_ratios": (0.8, 0.2, 0.0),
        "source": "https://zenodo.org/records/5706578",
        "resolution": "0.3 m",
    },
    "isaid": {
        "name": "iSAID",
        "classes": {
            0: "background", 1: "ship", 2: "storage tank", 3: "baseball diamond",
            4: "tennis court", 5: "basketball court", 6: "ground track field",
            7: "bridge", 8: "large vehicle", 9: "small vehicle", 10: "helicopter",
            11: "swimming pool", 12: "roundabout", 13: "soccer ball field",
            14: "plane", 15: "harbor",
        },
        "palette": {
            color: index for index, color in enumerate([
                (0, 0, 0), (0, 0, 63), (0, 63, 63), (0, 63, 0),
                (0, 63, 127), (0, 63, 191), (0, 63, 255), (0, 127, 63),
                (0, 127, 127), (0, 0, 127), (0, 0, 191), (0, 0, 255),
                (0, 191, 127), (0, 127, 191), (0, 127, 255), (0, 100, 155),
            ])
        },
        "background_id": 0,
        # Harbor is a large coastal region; the remaining foreground labels
        # are discrete targets/facilities in the iSAID ontology.
        "object_class_ids": list(range(1, 15)),
        "split_ratios": (0.75, 0.25, 0.0),
        "source": "https://captain-whu.github.io/iSAID/dataset.html",
    },
    "isprs-potsdam": {
        "name": "ISPRS-Potsdam",
        "classes": {0: "impervious surface", 1: "building", 2: "low vegetation", 3: "tree", 4: "car", 5: "clutter background"},
        "palette": {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5},
        "background_id": 5,
        "split_ratios": (0.8, 0.2, 0.0),
        "source": "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx",
        "stored_channels": ["red", "green", "blue"],
        "require_native_color_channels": True,
        "object_class_ids": [4],
    },
    "isprs-vaihingen": {
        "name": "ISPRS-Vaihingen",
        "classes": {0: "impervious surface", 1: "building", 2: "low vegetation", 3: "tree", 4: "car", 5: "clutter background"},
        "palette": {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5},
        "background_id": 5,
        "split_ratios": (0.8, 0.2, 0.0),
        "source": "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx",
        "stored_channels": ["near infrared", "red", "green"],
        "object_class_ids": [4],
    },
    "landcoverai": {
        "name": "LandCover.ai",
        "classes": {0: "background", 1: "building", 2: "woodland", 3: "water", 4: "road"},
        "background_id": 0,
        "split_ratios": (0.7, 0.15, 0.15),
        "source": "https://landcover.ai/",
        "resolution": "0.25/0.5 m",
    },
    "uavid": {
        "name": "UAVid",
        "classes": {0: "building", 1: "road", 2: "tree", 3: "low vegetation", 4: "moving car", 5: "static car", 6: "human", 7: "background clutter"},
        "palette": {(128, 0, 0): 0, (128, 64, 128): 1, (0, 128, 0): 2, (128, 128, 0): 3, (64, 0, 128): 4, (192, 0, 192): 5, (64, 64, 0): 6, (0, 0, 0): 7},
        "background_id": 7,
        "split_ratios": (0.75, 0.25, 0.0),
        "source": "https://uavid.nl/",
        "object_class_ids": [4, 5, 6],
    },
    "skyscapes": {
        "name": "SkyScapes",
        "dataset_variant": "Dense",
        # Official SS_Dense grayscale IDs. The 12 lane-marking subtypes are
        # merged into the dense semantic class ``lane marking``.
        "classes": {
            1: "low vegetation", 2: "paved road", 3: "non-paved road",
            4: "paved parking place", 5: "non-paved parking place", 6: "bikeway",
            7: "sidewalk", 8: "entrance exit", 9: "danger area",
            10: "lane marking", 11: "building", 12: "car", 13: "trailer",
            14: "van", 15: "truck", 16: "large truck", 17: "bus",
            18: "clutter", 19: "impervious surface", 20: "tree",
        },
        "palette": {
            (244, 35, 232): 1, (102, 102, 156): 2, (190, 153, 153): 3,
            (153, 153, 153): 4, (250, 170, 30): 5, (152, 251, 152): 6,
            (70, 130, 180): 7, (107, 142, 35): 8, (220, 220, 0): 9,
            (255, 0, 0): 10, (220, 20, 60): 11, (125, 0, 142): 12,
            (170, 200, 40): 13, (200, 60, 100): 14, (150, 18, 80): 15,
            (81, 180, 81): 16, (190, 241, 21): 17, (11, 119, 32): 18,
            (120, 240, 120): 19, (70, 70, 70): 20,
        },
        "background_id": None,
        "split_ratios": (0.8, 0.2, 0.0),
        "source": "https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes",
        "resolution": "0.13 m",
        "object_class_ids": [12, 13, 14, 15, 16, 17],
        "excluded_class_ids": [18],
    },
    "semantic-drone": {
        "name": "Semantic-Drone",
        # Preserve official train IDs 1..22. Unlabeled=0 and conflicting=23
        # are normalized to the shared ignore ID instead of becoming classes.
        "classes": {
            1: "paved area", 2: "dirt", 3: "grass", 4: "gravel", 5: "water",
            6: "rocks", 7: "pool", 8: "other vegetation", 9: "roof", 10: "wall",
            11: "window", 12: "door", 13: "fence", 14: "fence pole",
            15: "person", 16: "dog", 17: "car", 18: "bicycle", 19: "tree",
            20: "bald tree", 21: "AR marker", 22: "obstacle",
        },
        "palette": {
            (0, 0, 0): 255, (128, 64, 128): 1, (130, 76, 0): 2,
            (0, 102, 0): 3, (112, 103, 87): 4, (28, 42, 168): 5,
            (48, 41, 30): 6, (0, 50, 89): 7, (107, 142, 35): 8,
            (70, 70, 70): 9, (102, 102, 156): 10, (254, 228, 12): 11,
            (254, 148, 12): 12, (190, 153, 153): 13, (153, 153, 153): 14,
            (255, 22, 96): 15, (102, 51, 0): 16, (9, 143, 150): 17,
            (119, 11, 32): 18, (51, 51, 0): 19, (190, 250, 190): 20,
            (112, 150, 146): 21, (2, 135, 115): 22, (255, 0, 0): 255,
        },
        "value_remap": {0: 255, 23: 255},
        "background_id": None,
        "split_ratios": (0.8, 0.2, 0.0),
        "source": "https://ivc.tugraz.at/research-project/semantic-drone-dataset/",
        # Paved/vegetation/ground classes remain LULC; bounded structures,
        # living targets, vehicles and markers use object-level text queries.
        "object_class_ids": [7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
    },
    "gid15": {
        "name": "GID-15",
        "classes": {
            0: "background", 1: "industrial land", 2: "urban residential",
            3: "rural residential", 4: "traffic land", 5: "paddy field",
            6: "irrigated land", 7: "dry cropland", 8: "garden plot",
            9: "arbor woodland", 10: "shrub land", 11: "natural grassland",
            12: "artificial grassland", 13: "river", 14: "lake", 15: "pond",
        },
        # The official release provides both grayscale PNG labels and RGB
        # TIFF labels. Keep the official class ordering for both encodings.
        "palette": {
            (0, 0, 0): 0,
            (200, 0, 0): 1,
            (250, 0, 150): 2,
            (200, 150, 150): 3,
            (250, 150, 150): 4,
            (0, 200, 0): 5,
            (150, 250, 0): 6,
            (150, 200, 150): 7,
            (200, 0, 200): 8,
            (150, 0, 250): 9,
            (150, 150, 250): 10,
            (250, 200, 0): 11,
            (200, 200, 0): 12,
            (0, 0, 200): 13,
            (0, 150, 200): 14,
            (0, 200, 250): 15,
        },
        "background_id": 0,
        "split_ratios": (0.8, 0.1, 0.1),
        "source": "https://captain-whu.github.io/GID15/",
        "resolution": "3 m",
    },
    "floodnet": {
        "name": "FloodNet",
        "classes": {
            0: "background", 1: "building flooded", 2: "building non-flooded",
            3: "road flooded", 4: "road non-flooded", 5: "water",
            6: "tree", 7: "vehicle", 8: "pool", 9: "grass",
        },
        "palette": {
            (0, 0, 0): 0, (255, 0, 0): 1, (180, 120, 120): 2,
            (160, 150, 20): 3, (140, 140, 140): 4, (61, 230, 250): 5,
            # Both blue encodings occur in the released ColorMasks archive
            # and denote tree. Keep these as exact aliases rather than
            # weakening palette validation with a large global tolerance.
            (0, 4, 255): 6, (0, 82, 255): 6,
            (255, 0, 245): 7, (255, 235, 0): 8,
            (4, 250, 7): 9,
        },
        "palette_tolerance": 12,
        # A few released masks contain isolated colors not listed in the
        # supplied palette. Some are equidistant from several semantic
        # classes, so guessing would corrupt supervision; preserve them as
        # the unified ignore ID and report their counts.
        "unknown_color_policy": "ignore",
        "background_id": 0,
        "split_ratios": (0.6, 0.2, 0.2),
        # The official bundle keeps the 4000x3000/3000x4000 source JPEGs,
        # while its semantic annotations are authored on a 1024x1024 grid.
        # Align RGB to that label grid before paired 512x512 tiling; never
        # resample the categorical mask.
        "image_alignment": "resize_to_mask",
        "source": "https://github.com/BinaLab/FloodNet-Supervised_v1.0",
        "object_class_ids": [1, 2, 7, 8],
    },
    "rescuenet": {
        "name": "RescueNet",
        "classes": {
            0: "background", 1: "water", 2: "building no damage",
            3: "building minor damage", 4: "building major damage",
            5: "building total destruction", 6: "vehicle", 7: "road clear",
            8: "road blocked", 9: "tree", 10: "pool",
        },
        "palette": {
            (0, 0, 0): 0, (61, 230, 250): 1, (180, 120, 120): 2,
            (235, 255, 7): 3, (255, 184, 6): 4, (255, 0, 0): 5,
            (255, 0, 245): 6, (140, 140, 140): 7, (160, 150, 20): 8,
            (4, 250, 7): 9, (255, 235, 0): 10,
        },
        "palette_tolerance": 12,
        "background_id": 0,
        "split_ratios": (0.8, 0.1, 0.1),
        "source": "https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation",
        "object_class_ids": [2, 3, 4, 5, 6, 10],
    },
    "deepglobe-landcover": {
        "name": "DeepGlobe-LandCover",
        "classes": {
            0: "urban land", 1: "agriculture land", 2: "rangeland",
            3: "forest land", 4: "water", 5: "barren land",
        },
        "palette": {
            (0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
            (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5,
            (0, 0, 0): 255,
        },
        "channel_threshold": 128,
        "background_id": None,
        "split_ratios": (0.8, 0.1, 0.1),
        "source": "https://competitions.codalab.org/competitions/18468",
        "resolution": "0.5 m",
    },
}

MASK_DIR_TOKENS = {"mask", "masks", "masks_png", "label", "labels", "annotation", "annotations", "ann_dir", "semantic_masks", "gt", "ground_truth"}
IMAGE_DIR_TOKENS = {"image", "images", "images_png", "img", "img_dir", "rgb", "top"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a native dataset as ovrs-semantic-v1")
    parser.add_argument("dataset", choices=sorted(ADAPTERS))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Read an already extracted images/labels tree directly instead of DATASET_ROOT/raw",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional image-only tree used to match masks by basename; repeat for multiple "
            "roots. This lets iSAID reuse DOTA-v1 train/val images without copying them."
        ),
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-overlays", type=int, default=50)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _canonical_stem(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_instance_color_RGB", "_instance_id_RGB", "_manual1", "_noBoundary",
        "_15label", "_label", "_labels", "_mask", "_masks", "_lab", "_sat", "_class",
        "_RGBIR", "_IRRG", "_RGB",
    ):
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
    return stem.lower()


def _dataset_parts(path: Path) -> list[str]:
    """Ignore unrelated ancestors whose names may contain train/test/mask."""
    parts = [part.lower() for part in path.parts]
    raw_positions = [index for index, part in enumerate(parts) if part == "raw"]
    if raw_positions:
        return parts[raw_positions[-1] + 1:]
    return parts[-6:]


def _infer_split(path: Path) -> str | None:
    # Prefer the closest split directory. This prevents unrelated source-root
    # ancestors such as ``test_data/UAVid/train/...`` from overriding train.
    for part in reversed(_dataset_parts(path)):
        value = part.replace("-", "_")
        if value in {"train", "training", "uavid_train"}:
            return "train"
        if value in {"val", "valid", "validation", "uavid_val"}:
            return "val"
        if value in {"test", "testing", "uavid_test"}:
            return "test"
        if value.startswith("train") or value.endswith("trainset"):
            return "train"
        if value.startswith(("val", "valid")) or value.endswith(("valset", "validationset")):
            return "val"
        if value.startswith("test") or value.endswith("testset"):
            return "test"
    return None


def _pair_context(path: Path, dataset: str) -> str | None:
    """Return a dataset-specific namespace for otherwise repeated basenames."""
    if dataset == "uavid":
        for part in _dataset_parts(path):
            if re.fullmatch(r"seq\d+", part, flags=re.IGNORECASE):
                return part.lower()
    return None


def _is_mask(path: Path) -> bool:
    parts = set(_dataset_parts(path.parent))
    stem = path.stem.lower()
    mask_directory = any(
        part in MASK_DIR_TOKENS or "mask" in part or "label" in part
        for part in parts
    )
    return mask_directory or any(
        token in stem for token in ("_mask", "_label", "_lab", "_class", "instance_color_rgb")
    )


def _is_image(path: Path, dataset: str) -> bool:
    if _is_mask(path):
        return False
    lower = "/".join(_dataset_parts(path))
    excluded_parts = {
        "dsm", "ndsm", "dem", "depth", "depths", "depth_map", "depth_maps",
        "height", "height_map", "height_maps", "elevation",
    }
    if any(part in excluded_parts for part in _dataset_parts(path)):
        return False
    if any(
        token in lower
        for token in (
            "dsm", "ndsm", "boundary", "depthmap", "depth_map",
            "heightmap", "height_map",
        )
    ):
        return False
    if dataset == "isprs-potsdam" and "irrg" in lower:
        return False
    return True


def _mask_priority(path: Path) -> tuple[int, int]:
    """Prefer complete, non-eroded truth when an archive contains label variants."""
    lower = path.as_posix().lower()
    score = 0
    score += 8 if "complete" in lower else 0
    score += 4 if "full" in lower else 0
    score -= 8 if "participant" in lower else 0
    score -= 6 if "noboundary" in lower or "no_boundary" in lower else 0
    score -= 6 if "eroded" in lower else 0
    return score, -len(path.parts)


def _split_lists(raw_dir: Path) -> dict[str, set[str]]:
    result = {}
    for name in ("train", "val", "test"):
        values = set()
        for path in raw_dir.rglob(f"{name}.txt"):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                value = line.strip()
                if value:
                    values.add(_canonical_stem(Path(value)))
        result[name] = values
    return result


def discover_pairs(
    dataset: str,
    root: Path,
    source_root: Path | None = None,
    image_roots: list[Path] | None = None,
) -> list[dict]:
    raw_dir = (
        source_root.expanduser().resolve()
        if source_root is not None
        else root.expanduser().resolve() / "raw"
    )
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory is absent: {raw_dir}")
    files = [
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    masks = [path for path in files if _is_mask(path)]
    if dataset == "isaid":
        # Official iSAID exports both per-instance IDs and class-color semantic
        # masks with matching stems. Only the latter belongs in a class-ID
        # semantic segmentation dataset.
        masks = [
            path for path in masks
            if "semantic_masks" in {part.lower() for part in path.parts}
        ]
    elif dataset == "skyscapes":
        # The archive includes equivalent RGB and grayscale labels. Prefer the
        # lossless official class-ID masks and avoid duplicate pairs. The same
        # outer archive also contains the separate SS_Multi_Lane benchmark;
        # never mix its 0..12 IDs into the 1..20 SS_Dense ontology.
        grayscale = [
            path for path in masks
            if "grayscale" in _dataset_parts(path)
            and "ss_dense" in _dataset_parts(path)
            and "ss_multi_lane" not in _dataset_parts(path)
        ]
        if grayscale:
            masks = grayscale
    elif dataset == "semantic-drone":
        semantic_masks = [
            path for path in masks
            if "/gt/semantic/label_images/" in f"/{'/'.join(_dataset_parts(path))}/"
        ]
        if not semantic_masks:
            raise FileNotFoundError(
                "Semantic Drone archive contains no training_set/gt/semantic/label_images masks"
            )
        masks = semantic_masks
    images = [path for path in files if _is_image(path, dataset)]
    if dataset == "skyscapes":
        images = [
            path for path in images
            if "ss_dense" in _dataset_parts(path)
            and "ss_multi_lane" not in _dataset_parts(path)
        ]
    resolved_image_roots = []
    for image_root in image_roots or []:
        image_root = image_root.expanduser().resolve()
        if not image_root.is_dir():
            raise FileNotFoundError(f"Additional image directory is absent: {image_root}")
        resolved_image_roots.append(image_root)
        images.extend(
            path for path in image_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and _is_image(path, dataset)
        )
    images = list(dict.fromkeys(images))
    if not masks or not images:
        raise FileNotFoundError(
            f"Could not discover native images/masks below {raw_dir}: "
            f"images={len(images)}, masks={len(masks)}, additional_image_roots="
            f"{[str(path) for path in resolved_image_roots]}"
        )
    split_lists = _split_lists(raw_dir)
    image_index: dict[tuple[str | None, str | None, str], list[Path]] = {}
    for image in images:
        key = (_infer_split(image), _pair_context(image, dataset), _canonical_stem(image))
        image_index.setdefault(key, []).append(image)
        image_index.setdefault((None, key[1], key[2]), []).append(image)
    records = []
    for mask in masks:
        stem, split = _canonical_stem(mask), _infer_split(mask)
        context = _pair_context(mask, dataset)
        candidates = (
            image_index.get((split, context, stem), [])
            or image_index.get((None, context, stem), [])
        )
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            continue
        mask_parts = set(part.lower() for part in mask.parts)
        image = max(candidates, key=lambda path: len(mask_parts & {part.lower() for part in path.parts}))
        if split is None:
            split = _infer_split(image)
        if split is None:
            matches = [name for name, values in split_lists.items() if stem in values]
            if len(matches) == 1:
                split = matches[0]
        sample_stem = f"{context}_{stem}" if context is not None else stem
        records.append({"stem": sample_stem, "split": split, "image": image, "mask": mask})
    if not records:
        raise ValueError(f"No image/mask pairs could be matched below {raw_dir}")
    # Official benchmark bundles sometimes include participant, complete and
    # eroded labels together.  A physical image must contribute exactly once.
    best_by_image = {}
    for record in records:
        image_key = str(record["image"].resolve()).lower()
        previous = best_by_image.get(image_key)
        if previous is None or _mask_priority(record["mask"]) > _mask_priority(previous["mask"]):
            best_by_image[image_key] = record
    records = list(best_by_image.values())
    counts = {}
    for record in records:
        counts[record["stem"]] = counts.get(record["stem"], 0) + 1
    for record in records:
        sample_id = record["stem"]
        if counts[sample_id] > 1:
            image_identity = record["image"].resolve().as_posix().lower()
            digest = hashlib.sha1(image_identity.encode()).hexdigest()[:8]
            sample_id = f"{sample_id}__{digest}"
        record["sample_id"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    unique = {}
    for record in records:
        previous = unique.get(record["sample_id"])
        if previous is not None and previous["mask"] != record["mask"]:
            raise ValueError(f"Duplicate sample ID after normalization: {record['sample_id']}")
        unique[record["sample_id"]] = record
    return sorted(unique.values(), key=lambda row: row["sample_id"])


def _decode_mask(
    path: Path, adapter: dict, return_unknown_colors: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(Image.open(path))
    unknown_color_counts: dict[str, int] = {}
    if raw.ndim == 3:
        raw = raw[..., :3]
        if "channel_threshold" in adapter:
            raw = np.where(raw >= int(adapter["channel_threshold"]), 255, 0).astype(np.uint8)
        output = np.full(raw.shape[:2], 255, dtype=np.uint8)
        palette = adapter.get("palette", {})
        known = np.zeros(raw.shape[:2], dtype=bool)
        for color, class_id in palette.items():
            active = np.all(raw == np.asarray(color, dtype=raw.dtype), axis=2)
            output[active] = int(class_id)
            known |= active
        tolerance = int(adapter.get("palette_tolerance", 0))
        if not known.all() and tolerance > 0:
            unknown_pixels = raw[~known].astype(np.int32)
            colors = np.asarray(list(palette), dtype=np.int32)
            ids = np.asarray(list(palette.values()), dtype=np.uint8)
            distances = ((unknown_pixels[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
            nearest = distances.argmin(axis=1)
            accepted = distances[np.arange(len(nearest)), nearest] <= tolerance ** 2
            mapped = output[~known]
            mapped[accepted] = ids[nearest[accepted]]
            output[~known] = mapped
            known_positions = np.flatnonzero(~known)
            known.flat[known_positions[accepted]] = True
        if not known.all():
            unknown, frequencies = np.unique(
                raw[~known].reshape(-1, 3), axis=0, return_counts=True
            )
            if adapter.get("unknown_color_policy") == "ignore":
                unknown_color_counts = {
                    ",".join(map(str, color.tolist())): int(count)
                    for color, count in zip(unknown, frequencies)
                }
            else:
                raise ValueError(
                    f"Mask {path} contains unknown RGB colors: {unknown[:10].tolist()}"
                )
    elif raw.ndim == 2:
        output = raw.astype(np.uint8, copy=True)
    else:
        raise ValueError(f"Unsupported mask shape {raw.shape}: {path}")
    for source, target in adapter.get("value_remap", {}).items():
        output[output == int(source)] = int(target)
    valid = set(adapter["classes"]) | {255}
    unknown_ids = set(map(int, np.unique(output))) - valid
    if unknown_ids:
        raise ValueError(f"Mask {path} contains unknown IDs: {sorted(unknown_ids)}")
    if return_unknown_colors:
        return output, unknown_color_counts
    return output


def _tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, stride))
    if values[-1] != length - tile_size:
        values.append(length - tile_size)
    return values


def _valid_prepared_pair(image_path: Path, mask_path: Path, tile_size: int) -> bool:
    """Return whether an existing converted sample is complete."""
    try:
        with Image.open(image_path) as stream:
            stream.verify()
        with Image.open(mask_path) as stream:
            stream.verify()
        with Image.open(image_path) as stream:
            image_size, image_mode = stream.size, stream.mode
        with Image.open(mask_path) as stream:
            mask_size, mask_mode = stream.size, stream.mode
        return (
            image_size == (tile_size, tile_size)
            and mask_size == image_size
            and image_mode == "RGB"
            and mask_mode in {"L", "I", "I;16"}
        )
    except (OSError, SyntaxError):
        return False


def _save_png_atomic(array: np.ndarray, mode: str, path: Path, **save_kwargs) -> None:
    """Never leave a partially written final PNG after interruption."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(array, mode=mode).save(
            temporary, format="PNG", **save_kwargs
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _convert_task(task: tuple) -> dict:
    record, root_text, adapter, overwrite, tile_size, stride = task
    root = Path(root_text)
    with Image.open(record["image"]) as stream:
        native = np.asarray(stream)
        if adapter.get("require_native_color_channels"):
            insufficient_channels = native.ndim != 3 or native.shape[2] < 3
            replicated_grayscale = False
            if not insufficient_channels:
                replicated_grayscale = bool(
                    np.mean(
                        (native[..., 0] == native[..., 1])
                        & (native[..., 1] == native[..., 2])
                    ) >= 0.995
                )
            if insufficient_channels or replicated_grayscale:
                raise ValueError(
                    f"{adapter['name']} requires a native RGB/RGBIR image, but "
                    f"{record['image']} has shape {native.shape} and "
                    f"replicated_grayscale={replicated_grayscale}. "
                    "This is likely a DSM/height map."
                )
        image = stream.convert("RGB")
        image_array = np.asarray(image)
    mask, unknown_colors = _decode_mask(
        record["mask"], adapter, return_unknown_colors=True
    )
    image_aligned_to_mask = False
    if image_array.shape[:2] != mask.shape:
        if adapter.get("image_alignment") == "resize_to_mask":
            resampling = getattr(Image, "Resampling", Image)
            image_array = np.asarray(
                Image.fromarray(image_array, mode="RGB").resize(
                    (mask.shape[1], mask.shape[0]), resample=resampling.LANCZOS
                )
            )
            image_aligned_to_mask = True
        else:
            raise ValueError(
                f"Image/mask mismatch for {record['sample_id']}: "
                f"{image_array.shape[:2]} vs {mask.shape}"
            )
    converted = skipped = 0
    sample_ids = []
    height, width = mask.shape
    for y in _tile_origins(height, tile_size, stride):
        for x in _tile_origins(width, tile_size, stride):
            sample_id = f"{record['sample_id']}__x{x}_y{y}"
            image_out = root / "images" / f"{sample_id}.png"
            mask_out = root / "masks" / f"{sample_id}.png"
            sample_ids.append(sample_id)
            if image_out.is_file() and mask_out.is_file() and not overwrite:
                if _valid_prepared_pair(image_out, mask_out, tile_size):
                    skipped += 1
                    continue
            image_tile = image_array[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            mask_tile = mask[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            if image_tile.shape[:2] != (tile_size, tile_size):
                padded_image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_mask = np.full((tile_size, tile_size), 255, dtype=np.uint8)
                padded_image[:image_tile.shape[0], :image_tile.shape[1]] = image_tile
                padded_mask[:mask_tile.shape[0], :mask_tile.shape[1]] = mask_tile
                image_tile, mask_tile = padded_image, padded_mask
            image_out.parent.mkdir(parents=True, exist_ok=True)
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            _save_png_atomic(image_tile, "RGB", image_out, compress_level=3)
            _save_png_atomic(mask_tile, "L", mask_out)
            converted += 1
    return {
        "sample_ids": sample_ids,
        "split": record["split"],
        "converted": converted,
        "skipped": skipped,
        "image_aligned_to_mask": int(image_aligned_to_mask),
        "unknown_colors_ignored": unknown_colors,
    }


def _assign_missing_splits(records: list[dict], ratios: tuple[float, float, float], seed: int) -> None:
    missing = [record for record in records if record["split"] is None]
    random.Random(seed).shuffle(missing)
    train_end = round(len(missing) * ratios[0])
    val_end = train_end + round(len(missing) * ratios[1])
    for index, record in enumerate(missing):
        record["split"] = "train" if index < train_end else "val" if index < val_end else "test"


def _split_deepglobe_test_for_validation(records: list[dict], seed: int) -> None:
    if any(record["split"] == "val" for record in records):
        return
    test_records = [record for record in records if record["split"] == "test"]
    random.Random(seed).shuffle(test_records)
    for record in test_records[: len(test_records) // 2]:
        record["split"] = "val"


def _write_overlays(
    root: Path, overlay_dir: Path, sample_ids: list[str], adapter: dict, count: int, seed: int
) -> list[str]:
    selected = sorted(random.Random(seed).sample(sample_ids, min(count, len(sample_ids))))
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for old in overlay_dir.glob("*.png"):
        if old.stem not in set(selected):
            old.unlink()
    for sample_id in selected:
        image = np.asarray(Image.open(root / "images" / f"{sample_id}.png").convert("RGB"))
        mask = np.asarray(Image.open(root / "masks" / f"{sample_id}.png"))
        fused = render_semantic_overlay(
            image,
            mask,
            adapter["classes"],
            background_id=adapter["background_id"],
            ignore_id=255,
            alpha=0.62,
            image_brightness=0.82,
        )
        Image.fromarray(fused, mode="RGB").save(overlay_dir / f"{sample_id}.png")
    return selected


def prepare(
    dataset: str,
    root: Path,
    workers: int,
    seed: int,
    num_overlays: int,
    overlay_dir: Path,
    overwrite: bool,
    source_root: Path | None = None,
    tile_size: int = 512,
    stride: int = 512,
    image_roots: list[Path] | None = None,
) -> dict:
    if workers < 1 or tile_size < 1 or stride < 1:
        raise ValueError("--workers, --tile-size and --stride must be positive")
    if num_overlays < 0:
        raise ValueError("--num-overlays cannot be negative")
    adapter = ADAPTERS[dataset]
    root = root.expanduser().resolve()
    source_root = source_root.expanduser().resolve() if source_root is not None else None
    resolved_image_roots = [path.expanduser().resolve() for path in (image_roots or [])]
    records = discover_pairs(dataset, root, source_root, resolved_image_roots)
    _assign_missing_splits(records, adapter["split_ratios"], seed)
    if dataset == "deepglobe-landcover":
        _split_deepglobe_test_for_validation(records, seed)
    tasks = [(record, str(root), adapter, overwrite, tile_size, stride) for record in records]
    counts = {"converted": 0, "skipped": 0, "images_aligned_to_mask": 0}
    unknown_colors_ignored: Counter[str] = Counter()
    tile_rows = []
    if workers == 1:
        results = map(_convert_task, tasks)
        iterator = tqdm(results, total=len(tasks), desc=f"Preparing {adapter['name']}")
        for result in iterator:
            counts["converted"] += result["converted"]
            counts["skipped"] += result["skipped"]
            counts["images_aligned_to_mask"] += result["image_aligned_to_mask"]
            unknown_colors_ignored.update(result["unknown_colors_ignored"])
            tile_rows.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_convert_task, tasks, chunksize=4)
            for result in tqdm(results, total=len(tasks), desc=f"Preparing {adapter['name']}"):
                counts["converted"] += result["converted"]
                counts["skipped"] += result["skipped"]
                counts["images_aligned_to_mask"] += result["image_aligned_to_mask"]
                unknown_colors_ignored.update(result["unknown_colors_ignored"])
                tile_rows.append(result)
    splits = {name: [] for name in ("train", "val", "test")}
    for row in tile_rows:
        splits.setdefault(row["split"], []).extend(row["sample_ids"])
    all_sample_ids = sorted(value for values in splits.values() for value in values)
    if overwrite:
        active = set(all_sample_ids)
        for directory in (root / "images", root / "masks"):
            for old in directory.glob("*.png"):
                if old.stem not in active:
                    old.unlink()
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        values.sort()
        (split_dir / f"{name}.txt").write_text(
            "".join(f"{value}\n" for value in values), encoding="utf-8"
        )
    query_scope_by_class = {
        class_id: (
            "object" if class_id in set(adapter.get("object_class_ids", [])) else "lulc"
        )
        for class_id in adapter["classes"]
    }
    write_class_catalog(
        root / "classes.json",
        adapter["name"],
        adapter["classes"],
        default_semantic_level="lulc",
        scope_by_class=query_scope_by_class,
        background_id=adapter["background_id"],
        excluded_class_ids=adapter.get("excluded_class_ids", []),
        exclusion_reason="undefined or non-semantic dataset label",
    )
    overlay_ids = _write_overlays(
        root, overlay_dir, all_sample_ids, adapter, num_overlays, seed
    )
    metadata = {
        "format_version": "ovrs-semantic-v1",
        "dataset_name": adapter["name"],
        "image_dir": "images",
        "mask_dir": "masks",
        "classes_file": "classes.json",
        "classes_format": "ovrs-classes-v1",
        "background_id": adapter["background_id"],
        "ignore_id": 255,
        "stored_image_channels": adapter.get("stored_channels", ["red", "green", "blue"]),
        "native_masks_preserved_semantically": True,
        "image_alignment": adapter.get("image_alignment", "native_dimensions_must_match"),
        "unknown_color_policy": adapter.get("unknown_color_policy", "error"),
        "unknown_colors_ignored": dict(sorted(unknown_colors_ignored.items())),
        "unknown_color_pixels_ignored": sum(unknown_colors_ignored.values()),
        "supervision_level": "pixel",
        "optimization_by_level": {
            "lulc": ["region_text_alignment", "mask_localization"],
            "object": ["instance_region_attention_bce"],
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
            str(class_id): scope for class_id, scope in query_scope_by_class.items()
        },
        "source": adapter["source"],
        "prepared_from": str(source_root if source_root is not None else root / "raw"),
        "additional_image_roots": [str(path) for path in resolved_image_roots],
        "spatial_resolution": adapter.get("resolution"),
        "source_sample_count": len(records),
        "sample_count": len(all_sample_ids),
        "tile_size": tile_size,
        "stride": stride,
        "split_unit": "source_image",
        "splits": {name: len(values) for name, values in splits.items()},
        "overlay_dir": str(overlay_dir),
        "overlay_sample_ids": overlay_ids,
    }
    if "dataset_variant" in adapter:
        metadata["dataset_variant"] = adapter["dataset_variant"]
    (root / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "dataset": adapter["name"],
        "root": str(root),
        "samples": len(all_sample_ids),
        **counts,
        "unknown_color_pixels_ignored": sum(unknown_colors_ignored.values()),
        "unknown_colors_ignored": dict(sorted(unknown_colors_ignored.items())),
        "splits": metadata["splits"],
    }


def main() -> None:
    args = parse_args()
    overlay_dir = args.overlay_dir or Path("outputs") / args.dataset / "overlays"
    summary = prepare(
        args.dataset,
        args.dataset_root,
        args.workers,
        args.seed,
        args.num_overlays,
        overlay_dir.expanduser().resolve(),
        args.overwrite,
        args.source_root,
        args.tile_size,
        args.stride,
        args.image_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
