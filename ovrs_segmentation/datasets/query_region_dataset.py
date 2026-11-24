from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from ovrs_segmentation.inference.tiling import minimum_size
from PIL import Image, ImageDraw, UnidentifiedImageError
from torch.nn import functional as F

from .class_prompts import (
    class_specs_by_id,
    combined_prompt_texts,
)
from .semantic_dataset import (
    CorruptSampleError,
    SemanticSegmentationDataset,
    load_class_texts,
)
from .lmdb_store import open_lmdb_store


SEMANTIC_LEVELS = {"scene", "lulc", "object"}
SUPERVISION_LEVELS = {"image", "box", "pixel"}
ANNOTATION_CONTRACTS = {("scene", "image"), ("object", "box"), ("lulc", "pixel")}


def _dataset_vocabulary(
    *,
    dataset_id: str,
    dataset_name: str,
    class_specs: dict[int, dict[str, Any]],
    level: str,
    default_level: str,
    ignore_id: int,
    background_id: int | None = None,
    scope_by_class: dict[int, str] | None = None,
    single_label: bool = True,
) -> dict[str, Any]:
    """Build one contiguous, dataset-local vocabulary for one semantic level."""
    classes: list[dict[str, Any]] = []
    for raw_id, spec in sorted(class_specs.items()):
        class_text = str(spec["class_text"])
        is_ignore = bool(spec.get("is_ignore")) or raw_id == ignore_id
        is_background = bool(spec.get("is_background")) or raw_id == background_id
        is_excluded = bool(spec.get("is_excluded"))
        queryable = bool(
            spec.get("queryable", not (is_ignore or is_background or is_excluded))
        )
        scope = (
            (scope_by_class or {}).get(raw_id)
            or spec.get("semantic_level")
            or default_level
        )
        include = (
            not is_ignore
            and not is_background
            and not is_excluded
            and scope == level
            and queryable
        )
        if not include:
            continue
        prompt_groups = combined_prompt_texts(class_text, level, dataset_name)
        prompt_texts = prompt_groups["prompt_texts"]
        classes.append(
            {
                "local_class_id": len(classes),
                "raw_class_id": int(raw_id),
                "class_name": class_text,
                "prompt_text": prompt_texts[0],
                **prompt_groups,
                "is_background": is_background,
            }
        )
    if not classes:
        raise ValueError(f"Dataset {dataset_name!r} has no {level} vocabulary")
    return {
        "dataset_id": str(dataset_id),
        "dataset_name": str(dataset_name),
        "level": level,
        "class_names": [row["class_name"] for row in classes],
        "classes": classes,
        "raw_to_local": {
            str(row["raw_class_id"]): row["local_class_id"] for row in classes
        },
        "local_to_raw": {
            str(row["local_class_id"]): row["raw_class_id"] for row in classes
        },
        "num_classes": len(classes),
        "single_label": bool(single_label),
    }


def _local_class_id(vocabulary: dict[str, Any], raw_class_id: int) -> int:
    try:
        return int(vocabulary["raw_to_local"][str(int(raw_class_id))])
    except KeyError as error:
        raise ValueError(
            f"Raw class ID {raw_class_id} is absent from the {vocabulary['level']} "
            f"vocabulary of {vocabulary['dataset_name']}"
        ) from error


def _local_pixel_mask(
    raw_mask: np.ndarray,
    vocabulary: dict[str, Any],
    ignore_id: int,
) -> np.ndarray:
    local = np.full(raw_mask.shape, ignore_id, dtype=np.uint8)
    for raw_id, local_id in vocabulary["raw_to_local"].items():
        local[raw_mask == int(raw_id)] = int(local_id)
    return local


def _local_pixel_valid_mask(
    raw_mask: np.ndarray,
    vocabulary: dict[str, Any],
    ignore_id: int,
    background_ids: Sequence[int],
) -> np.ndarray:
    valid = np.zeros(raw_mask.shape, dtype=bool)
    for raw_id in vocabulary["raw_to_local"]:
        valid |= raw_mask == int(raw_id)
    for raw_id in background_ids:
        valid |= raw_mask == int(raw_id)
    return valid & (raw_mask != ignore_id)


def _read_pil_array(path: Path, *, rgb: bool, storage=None) -> np.ndarray:
    try:
        source = BytesIO(storage.read(path)) if storage else path
        with Image.open(source) as stream:
            return np.asarray(stream.convert("RGB") if rgb else stream)
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        size = path.stat().st_size if path.is_file() else None
        kind = "RGB image" if rgb else "pixel target"
        raise CorruptSampleError(
            f"Unreadable {kind}: {path} (size={size} bytes). "
            "The file is truncated or is not a valid image."
        ) from error


def _read_jsonl(path: Path, storage=None) -> list[dict[str, Any]]:
    text = storage.read(path).decode("utf-8") if storage else path.read_text(encoding="utf-8")
    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def _quality(raw: Any, *, source: str) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"label": raw, "source": source, "confidence": None}
    if isinstance(raw, dict):
        result = dict(raw)
        result.setdefault("source", source)
        result.setdefault("confidence", None)
        return result
    return {
        "label": "native",
        "source": source,
        "confidence": 1.0,
    }


def _annotation_contract(raw: dict[str, Any]) -> tuple[str, str]:
    semantic = str(raw.get("semantic_level", raw.get("scope", "scene")))
    supervision = str(
        raw.get("supervision_level", "image" if semantic == "scene" else "pixel")
    )
    if (semantic, supervision) not in ANNOTATION_CONTRACTS:
        raise ValueError(
            "scene requires image supervision, object requires box supervision, "
            "and LULC requires pixel supervision"
        )

    return semantic, supervision


def _annotation_target(
    raw: dict[str, Any], supervision_level: str, class_id: int
) -> dict[str, Any] | None:
    target = raw.get("target")
    if target is not None:
        return target
    if supervision_level == "image":
        return {"type": "presence", "present": bool(raw.get("present", True))}
    if supervision_level == "box":
        boxes = raw.get("boxes_xyxy")
        if boxes is None and raw.get("bbox_xyxy") is not None:
            boxes = [raw["bbox_xyxy"]]
        return {
            "type": "detection_geometry",
            "boxes": boxes or [],
            "polygons": raw.get("polygons_xy", []),
        }
    if "region_ref" in raw:
        region = raw["region_ref"]
        return {
            "type": raw.get("region_encoding", "class_id_mask"),
            "path": region.get("mask_path"),
            "class_id": int(region.get("class_id", class_id)),
        }
    if raw.get("mask"):
        return {
            "type": "binary_mask",
            "path": str(raw["mask"]),
            "positive_values": [1, 255],
        }
    return None


def _box_geometry(
    raw: dict[str, Any], target: dict[str, Any] | None, annotation_id: str
) -> dict[str, Any]:
    boxes = []
    for box in (target or {}).get("boxes", raw.get("boxes_xyxy", [])) or []:
        coordinates = np.asarray(box, dtype=float)
        if (
            coordinates.shape != (4,)
            or not np.isfinite(coordinates).all()
            or coordinates[2] <= coordinates[0]
            or coordinates[3] <= coordinates[1]
        ):
            raise ValueError(f"Invalid xyxy box in {annotation_id}: {box!r}")
        boxes.append(coordinates.tolist())
    if not boxes:
        raise ValueError(f"Box annotation {annotation_id} contains no boxes")

    polygons = []
    for polygon in (target or {}).get("polygons", raw.get("polygons_xy", [])) or []:
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"Invalid polygon in {annotation_id}: {polygon!r}")
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite polygon in {annotation_id}")
        polygons.append(points.tolist())
    result = {"boxes_xyxy": boxes}
    if polygons:
        result["polygons_xy"] = polygons
    return result


def _canonical_annotation(
    raw: dict[str, Any],
    *,
    dataset: str,
    class_text_by_id: dict[int, str],
    annotation_id: str,
    default_ignore_id: int = 255,
) -> dict[str, Any]:
    class_id = int(raw.get("dataset_class_id", raw.get("class_id")))
    if class_id not in class_text_by_id:
        raise ValueError(f"Unknown dataset class ID {class_id} in {dataset}")
    class_text = class_text_by_id[class_id]
    semantic_level, supervision_level = _annotation_contract(raw)
    prompt_groups = combined_prompt_texts(class_text, semantic_level, dataset)
    prompts = prompt_groups["prompt_texts"]
    target_ref = _annotation_target(raw, supervision_level, class_id)
    default_supervision_source = (
        "native_image_annotation"
        if supervision_level == "image"
        else "native_pixel_annotation"
    )
    annotation_quality = _quality(
        raw.get("annotation_quality"),
        source=str(raw.get("supervision_source") or default_supervision_source),
    )
    supervision_source = str(
        raw.get("supervision_source")
        or annotation_quality.get("source")
        or default_supervision_source
    )
    result = {
        "annotation_id": str(
            raw.get("annotation_id", raw.get("query_id", annotation_id))
        ),
        "dataset": dataset,
        "dataset_class_id": class_id,
        "class_text": class_text,
        "prompt_texts": prompts,
        "native_prompt_texts": prompt_groups["native_prompt_texts"],
        "canonical_prompt_texts": prompt_groups["canonical_prompt_texts"],
        "variant_prompt_texts": prompt_groups["variant_prompt_texts"],
        "semantic_level": semantic_level,
        "supervision_level": supervision_level,
        "target_ref": target_ref,
        "ignore_id": int(raw.get("ignore_id", default_ignore_id)),
        "supervision_source": supervision_source,
        "exhaustive_pixel_labels": supervision_level == "pixel",
        "annotation_quality": annotation_quality,
    }
    if supervision_level == "box":
        result.update(_box_geometry(raw, target_ref, annotation_id))
    result["present"] = bool(
        (target_ref or {}).get("present", raw.get("present", True))
    )
    return result


def _pixel_target(
    class_id_mask: np.ndarray,
    class_id: int,
    ignore_id: int,
) -> np.ndarray:
    positive = class_id_mask == class_id
    valid = class_id_mask != ignore_id
    target = np.full(class_id_mask.shape, ignore_id, dtype=np.uint8)
    target[valid] = 0
    target[positive & valid] = 1
    return target


class SceneTextPromptDataset(Sequence[dict[str, Any]]):
    """Read image-level Scene text annotations."""

    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.metadata = json.loads(
            (self.root / "dataset.json").read_text(encoding="utf-8")
        )
        self.storage = open_lmdb_store(self.root, self.metadata)
        if (
            self.metadata.get("semantic_level") != "scene"
            or self.metadata.get("supervision_level") != "image"
        ):
            raise ValueError(
                "Scene datasets require semantic_level=scene and supervision_level=image"
            )
        self.dataset_name = str(self.metadata["dataset_name"])
        self.dataset_id = str(self.metadata.get("dataset_id", self.dataset_name))
        classes_path = self.root / self.metadata.get("classes_file", "classes.json")
        self.class_text_by_id = load_class_texts(classes_path)
        self.class_specs_by_id = class_specs_by_id(classes_path)
        self.dataset_vocabularies = {
            "scene": _dataset_vocabulary(
                dataset_id=self.dataset_id,
                dataset_name=self.dataset_name,
                class_specs=self.class_specs_by_id,
                level="scene",
                default_level="scene",
                ignore_id=int(self.metadata.get("ignore_id", 255)),
                background_id=self.metadata.get("background_id"),
                single_label=bool(self.metadata.get("single_label", True)),
            )
        }
        annotation_path = self.root / self.metadata.get(
            "annotations_file", "annotations.jsonl"
        )
        rows = {str(row["sample_id"]): row for row in _read_jsonl(annotation_path, self.storage)}
        if split is None:
            self.sample_ids = sorted(rows)
        else:
            split_path = self.root / "splits" / f"{split}.txt"
            if not split_path.is_file():
                raise FileNotFoundError(split_path)
            self.sample_ids = [
                line.strip()
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        missing = [sample_id for sample_id in self.sample_ids if sample_id not in rows]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} split samples are absent from {annotation_path}"
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.sample_ids)

    def class_ids_for_index(self, index: int, level: str) -> set[int]:
        """Return local labels without loading the image."""
        if level != "scene":
            return set()
        row = self.rows[self.sample_ids[index]]
        vocabulary = self.dataset_vocabularies["scene"]
        labels = set()
        for position, raw in enumerate(row.get("annotations", row.get("queries", []))):
            annotation = _canonical_annotation(
                raw,
                dataset=self.dataset_name,
                class_text_by_id=self.class_text_by_id,
                annotation_id=f"{self.sample_ids[index]}:{position}",
                default_ignore_id=int(self.metadata.get("ignore_id", 255)),
            )
            if annotation["present"] and annotation["semantic_level"] == "scene":
                labels.add(
                    _local_class_id(vocabulary, int(annotation["dataset_class_id"]))
                )
        return labels

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        row = self.rows[sample_id]
        image_path = self.root / row.get("image", row.get("image_path"))
        image = _read_pil_array(image_path, rgb=True, storage=self.storage)
        valid_image_mask = np.ones(image.shape[:2], dtype=bool)
        if row.get("source_size"):
            source_width, source_height = map(int, row["source_size"])
            scale = min(image.shape[1] / source_width, image.shape[0] / source_height)
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            left = (image.shape[1] - resized_width) // 2
            top = (image.shape[0] - resized_height) // 2
            valid_image_mask[:] = False
            valid_image_mask[
                top : top + resized_height, left : left + resized_width
            ] = True
        raw_annotations = row.get("annotations", row.get("queries", []))
        annotations = [
            _canonical_annotation(
                raw,
                dataset=self.dataset_name,
                class_text_by_id=self.class_text_by_id,
                annotation_id=f"{sample_id}:{position}",
                default_ignore_id=int(self.metadata.get("ignore_id", 255)),
            )
            for position, raw in enumerate(raw_annotations)
        ]
        annotations = [
            annotation for annotation in annotations if annotation["present"]
        ]
        vocabulary = self.dataset_vocabularies["scene"]
        for annotation in annotations:
            raw_class_id = int(annotation["dataset_class_id"])
            annotation["raw_dataset_class_id"] = raw_class_id
            annotation["dataset_class_id"] = _local_class_id(vocabulary, raw_class_id)
        for annotation in annotations:
            self._load_target(annotation)
        return {
            "sample_id": sample_id,
            "dataset": self.dataset_name,
            "dataset_name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "image": image,
            "image_path": image_path,
            "valid_image_mask": valid_image_mask,
            "annotations": annotations,
            "class_text_by_id": self.class_text_by_id,
            "dataset_vocabularies": self.dataset_vocabularies,
            "scene_local_class_ids": sorted(
                {
                    int(annotation["dataset_class_id"])
                    for annotation in annotations
                    if annotation["semantic_level"] == "scene"
                }
            ),
            "scene_label": (
                int(annotations[0]["dataset_class_id"])
                if len(annotations) == 1
                else None
            ),
            "dataset_local_pixel_mask": None,
            "dataset_local_pixel_valid_mask": None,
        }

    def _load_target(self, annotation: dict[str, Any]) -> None:
        supervision = annotation["supervision_level"]
        if supervision == "image":
            annotation["target"] = annotation["present"]
            return
        if supervision == "box":
            return

        target_ref = annotation["target_ref"]
        target_path = Path(target_ref["path"])
        if not target_path.is_absolute():
            target_path = self.root / target_path
        target = _read_pil_array(target_path, rgb=False, storage=self.storage)
        target_type = target_ref.get("type", "class_id_mask")
        if target_type == "class_id_mask":
            annotation["target"] = _pixel_target(
                target,
                int(target_ref.get("class_id", annotation["dataset_class_id"])),
                annotation["ignore_id"],
            )
        elif target_type == "binary_mask":
            positives = list(map(int, target_ref.get("positive_values", [1])))
            annotation["target"] = np.isin(target, positives).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported target type {target_type!r}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


class SemanticTextPromptDataset(Sequence[dict[str, Any]]):
    """Expose ovrs-semantic-v1 masks through text-conditioned annotations."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        *,
        validate_masks: bool = True,
        level: str | None = None,
    ) -> None:
        self.semantic = SemanticSegmentationDataset(
            root, split=split, validate_masks=validate_masks
        )
        self.dataset_id = str(
            self.semantic.metadata.get("dataset_id", self.semantic.dataset_name)
        )
        default_scope = (
            "object"
            if str(self.semantic.metadata.get("supervision_level", "")).lower()
            == "box"
            else "lulc"
        )
        dataset_level = level or default_scope
        if dataset_level not in {"lulc", "object"}:
            raise ValueError(f"Unsupported semantic level: {dataset_level}")
        self.scope_by_class = {
            class_id: dataset_level
            for class_id in self.semantic.class_text_by_id
        }
        self.background_class_ids = [
            class_id
            for class_id, spec in self.semantic.class_specs_by_id.items()
            if bool(spec.get("is_background"))
            or class_id == self.semantic.background_id
        ]
        self.dataset_vocabularies = {
            level: _dataset_vocabulary(
                dataset_id=self.dataset_id,
                dataset_name=self.semantic.dataset_name,
                class_specs=self.semantic.class_specs_by_id,
                level=level,
                default_level=default_scope,
                ignore_id=self.semantic.ignore_id,
                background_id=self.semantic.background_id,
                scope_by_class=self.scope_by_class,
                single_label=False,
            )
            for level in sorted(set(self.scope_by_class.values()))
        }
        self.exhaustive = True
        annotations_dir = self.semantic.metadata.get("annotations_dir")
        self.detection_annotations_dir = (
            None if not annotations_dir else self.semantic.root / str(annotations_dir)
        )
        if self.detection_annotations_dir is not None and not self.detection_annotations_dir.is_dir() and not (self.semantic.storage and self.semantic.storage.has_prefix(self.detection_annotations_dir)):
            raise FileNotFoundError(self.detection_annotations_dir)
        self.box_only = bool(
            self.detection_annotations_dir is not None
            and str(self.semantic.metadata.get("supervision_level", "")).lower()
            == "box"
        )

    def _detection_geometry_for_sample(self, sample_id: str) -> tuple[
        dict[int, list[list[float]]],
        dict[int, list[list[list[float]]]],
    ]:
        if self.detection_annotations_dir is None:
            return {}, {}
        path = self.detection_annotations_dir / f"{sample_id}.json"
        if not path.is_file() and not self.semantic.storage:
            raise FileNotFoundError(
                f"Image {sample_id!r} has no corresponding annotation JSON: {path}"
            )
        text = self.semantic.storage.read(path).decode("utf-8") if self.semantic.storage else path.read_text(encoding="utf-8")
        row = json.loads(text)
        if str(row.get("sample_id")) != sample_id:
            raise ValueError(
                f"Annotation sample_id mismatch in {path}: {row.get('sample_id')!r}"
            )
        grouped: dict[int, list[list[float]]] = {}
        polygons: dict[int, list[list[list[float]]]] = {}
        for annotation in row.get("annotations", []):
            class_id = int(annotation["dataset_class_id"])
            box = annotation.get("bbox_xyxy")
            if box is None or len(box) != 4:
                raise ValueError(f"Invalid bbox_xyxy in {path}: {box!r}")
            coordinates = [float(value) for value in box]
            if not np.isfinite(coordinates).all():
                raise ValueError(f"Non-finite bbox_xyxy in {path}: {box!r}")
            if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                continue
            grouped.setdefault(class_id, []).append(coordinates)
            polygon = annotation.get("oriented_bbox_xy")
            if polygon:
                points = [[float(point[0]), float(point[1])] for point in polygon]
                if len(points) < 3 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid oriented_bbox_xy in {path}")
                polygons.setdefault(class_id, []).append(points)
        return grouped, polygons

    def __len__(self) -> int:
        return len(self.semantic)

    def class_ids_for_index(self, index: int, level: str) -> set[int]:
        """Return local labels while avoiding RGB image decoding."""
        vocabulary = self.dataset_vocabularies.get(level)
        if vocabulary is None:
            return set()
        sample_id = self.semantic.sample_ids[index]
        if level == "object":
            raw_ids = self._detection_geometry_for_sample(sample_id)[0]
        elif level == "lulc" and not self.box_only:
            mask = self.semantic._read_mask(self.semantic._masks[sample_id])
            raw_ids = np.unique(mask).tolist()
        else:
            return set()
        raw_to_local = vocabulary["raw_to_local"]
        return {
            int(raw_to_local[str(raw_id)])
            for raw_id in raw_ids
            if str(raw_id) in raw_to_local
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = (
            self.semantic.image_only_sample(index)
            if self.box_only
            else self.semantic[index]
        )
        annotations = []
        sample_detection_boxes, sample_detection_polygons = (
            self._detection_geometry_for_sample(sample["sample_id"])
        )
        present_class_ids = set(
            sample_detection_boxes if self.box_only else sample["present_class_ids"]
        )
        present_class_ids.update(sample_detection_boxes)
        # Only concepts present in this image become SAM3 text queries. The
        # binary target still treats every other valid pixel as negative.
        for class_id in sorted(present_class_ids):
            if class_id in {sample["background_id"], sample["ignore_id"]}:
                continue
            if class_id not in self.semantic.class_specs_by_id:
                # Per-image files preserve the original detector annotation,
                # including classes intentionally excluded from the active
                # training catalog. They remain on disk but are not queried.
                continue
            class_spec = self.semantic.class_specs_by_id[class_id]
            if class_spec.get("queryable") is False:
                continue
            semantic_level = self.scope_by_class[class_id]
            box_supervision = semantic_level == "object"
            object_boxes = sample_detection_boxes.get(class_id)
            object_polygons = sample_detection_polygons.get(class_id, [])
            if box_supervision and not object_boxes:
                continue
            class_text = sample["class_text_by_id"][class_id]
            annotation = _canonical_annotation(
                {
                    "annotation_id": f"{sample['sample_id']}:{class_id}",
                    "dataset_class_id": class_id,
                    "class_text": class_text,
                    "semantic_level": semantic_level,
                    "supervision_level": "box" if box_supervision else "pixel",
                    "target": (
                        {
                            "type": "detection_geometry",
                            "boxes": object_boxes,
                            "polygons": object_polygons,
                        }
                        if box_supervision
                        else {
                            "type": "class_id_mask",
                            "path": str(sample["mask_path"]),
                            "class_id": class_id,
                            "exhaustive": True,
                        }
                    ),
                    "present": class_id in present_class_ids,
                    "ignore_id": sample["ignore_id"],
                    "annotation_quality": (
                        {
                            "label": "weak",
                            "source": "native_detection_box",
                            "confidence": 1.0,
                        }
                        if box_supervision
                        else self.semantic.metadata.get("annotation_quality")
                    ),
                    "supervision_source": (
                        "native_detection_box"
                        if box_supervision
                        else "native_pixel_annotation"
                    ),
                },
                dataset=sample["dataset_name"],
                class_text_by_id=sample["class_text_by_id"],
                annotation_id=f"{sample['sample_id']}:{class_id}",
                default_ignore_id=sample["ignore_id"],
            )
            if not box_supervision:
                annotation["target"] = _pixel_target(
                    sample["class_id_mask"],
                    class_id,
                    sample["ignore_id"],
                )
                annotation["exhaustive_pixel_labels"] = True
            raw_class_id = int(annotation["dataset_class_id"])
            annotation["raw_dataset_class_id"] = raw_class_id
            annotation["dataset_class_id"] = _local_class_id(
                self.dataset_vocabularies[semantic_level], raw_class_id
            )
            annotations.append(annotation)
        lulc_vocabulary = self.dataset_vocabularies.get("lulc")
        local_pixel_mask = (
            None
            if self.box_only or lulc_vocabulary is None
            else _local_pixel_mask(
                sample["class_id_mask"], lulc_vocabulary, sample["ignore_id"]
            )
        )
        local_pixel_valid_mask = (
            None
            if self.box_only or lulc_vocabulary is None
            else _local_pixel_valid_mask(
                sample["class_id_mask"],
                lulc_vocabulary,
                sample["ignore_id"],
                self.background_class_ids,
            )
        )
        return {
            **sample,
            "dataset": sample["dataset_name"],
            "dataset_id": self.dataset_id,
            "dataset_vocabularies": self.dataset_vocabularies,
            "scene_local_class_ids": [],
            "scene_label": None,
            "dataset_local_pixel_mask": local_pixel_mask,
            "dataset_local_pixel_valid_mask": local_pixel_valid_mask,
            "valid_image_mask": np.ones(sample["image"].shape[:2], dtype=bool),
            "annotations": annotations,
        }


def open_text_prompt_dataset(
    root: str | Path,
    split: str | None = None,
    *,
    validate_masks: bool = True,
    level: str | None = None,
):
    source = Path(root).expanduser().resolve()
    metadata = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    if (
        metadata.get("semantic_level") == "scene"
        and metadata.get("supervision_level") == "image"
    ):
        return SceneTextPromptDataset(source, split)
    return SemanticTextPromptDataset(
        source, split, validate_masks=validate_masks, level=level
    )


def _spatial_size(value: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    size = (value, value) if isinstance(value, int) else tuple(map(int, value))
    if len(size) != 2 or min(size) < 1:
        raise ValueError("spatial_size must contain two positive dimensions")
    return size


def _resize_mask(mask: Any, size: tuple[int, int], dtype: torch.dtype, *, parameters=None, fill=255) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(mask, copy=True))[None, None].float()
    if parameters is not None:
        return _apply_spatial_augmentation(tensor[0, 0], parameters, mode="nearest", fill=fill).to(dtype)
    if tuple(tensor.shape[-2:]) != size:
        tensor = F.interpolate(tensor, size=size, mode="nearest")
    return tensor[0, 0].to(dtype)


def _collate_images(
    samples: Sequence[dict[str, Any]], target_size: tuple[int, int] | None, parameters=None
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    images, valid_masks, original_sizes = [], [], []
    for sample_index, sample in enumerate(samples):
        image = np.asarray(sample["image"])
        source_size = tuple(map(int, image.shape[:2]))
        original_sizes.append(source_size)
        image_tensor = (
            torch.from_numpy(np.array(image, copy=True))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        valid = sample.get("valid_image_mask", np.ones(source_size, dtype=bool))
        spatial = parameters[sample_index] if parameters is not None else None
        valid_tensor = _resize_mask(valid, target_size or source_size, torch.bool, parameters=spatial, fill=0)
        if spatial is not None:
            image_tensor = _apply_spatial_augmentation(image_tensor, spatial, mode="bilinear", fill=0.5)
        elif target_size is not None and source_size != target_size:
            image_tensor = F.interpolate(
                image_tensor[None],
                size=target_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        images.append(image_tensor)
        valid_masks.append(valid_tensor)
    return torch.stack(images), torch.stack(valid_masks), original_sizes


def _hash_index(token: str, count: int) -> int:
    return int(hashlib.sha1(token.encode()).hexdigest()[:8], 16) % count


def _select_prompt(annotation: dict[str, Any], strategy: str, seed: int) -> str:
    if strategy == "first":
        return annotation["prompt_texts"][0]
    groups = [
        values
        for values in (
            annotation["native_prompt_texts"],
            annotation["canonical_prompt_texts"],
            annotation["variant_prompt_texts"],
        )
        if values
    ]
    annotation_id = annotation["annotation_id"]
    variants = groups[_hash_index(f"{seed}:{annotation_id}:group", len(groups))]
    return variants[_hash_index(f"{seed}:{annotation_id}:prompt", len(variants))]


def _normalized_boxes(
    annotation: dict[str, Any], source_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    xyxy = torch.tensor(annotation.get("boxes_xyxy", []), dtype=torch.float32).reshape(
        -1, 4
    )
    if not len(xyxy):
        return xyxy, xyxy.clone()
    source_height, source_width = source_size
    xyxy = (
        xyxy / torch.tensor([source_width, source_height, source_width, source_height])
    ).clamp_(0.0, 1.0)
    cxcywh = torch.stack(
        (
            (xyxy[:, 0] + xyxy[:, 2]) * 0.5,
            (xyxy[:, 1] + xyxy[:, 3]) * 0.5,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1],
        ),
        dim=1,
    )
    return xyxy, cxcywh


def _rasterize_boxes(
    annotation: dict[str, Any],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    parameters=None,
) -> torch.Tensor:
    height, width = target_size
    source_height, source_width = source_size
    scale_x, scale_y = width / source_width, height / source_height
    offset_x = offset_y = 0
    if parameters is not None:
        scale_x, scale_y = parameters["scaled_width"] / source_width, parameters["scaled_height"] / source_height
        offset_x = parameters["pad_left"] - parameters["crop_left"]
        offset_y = parameters["pad_top"] - parameters["crop_top"]
    instances = []
    polygons = annotation.get("polygons_xy", [])
    for polygon in polygons:
        raster = Image.new("L", (width, height), 0)
        ImageDraw.Draw(raster).polygon(
            [(float(x) * scale_x + offset_x, float(y) * scale_y + offset_y) for x, y in polygon], fill=1
        )
        instance = torch.from_numpy(np.asarray(raster, dtype=np.uint8).copy()).bool()
        if instance.any():
            instances.append(instance)
    if not polygons:
        for x1, y1, x2, y2 in annotation.get("boxes_xyxy", []):
            left = max(0, min(width, int(np.floor(x1 * scale_x + offset_x))))
            top = max(0, min(height, int(np.floor(y1 * scale_y + offset_y))))
            right = max(0, min(width, int(np.ceil(x2 * scale_x + offset_x))))
            bottom = max(0, min(height, int(np.ceil(y2 * scale_y + offset_y))))
            if right > left and bottom > top:
                instance = torch.zeros(target_size, dtype=torch.bool)
                instance[top:bottom, left:right] = True
                instances.append(instance)
    if not instances:
        if parameters is not None:
            return torch.empty(0, height, width, dtype=torch.bool)
        raise ValueError(f"Empty box supervision in {annotation['annotation_id']}")
    masks = torch.stack(instances)
    return _orient_spatial(masks, parameters) if parameters is not None else masks


def _collate_annotations(
    samples: Sequence[dict[str, Any]],
    original_sizes: Sequence[tuple[int, int]],
    output_size: tuple[int, int],
    prompt_strategy: str,
    seed: int,
    parameters=None,
) -> dict[str, Any]:
    rows = {
        "annotations": [],
        "sample_indices": [],
        "prompts": [],
        "pixel_targets": [],
        "scene_targets": [],
        "boxes_xyxy": [],
        "boxes": [],
        "instance_box_masks": [],
    }
    ignore_target = torch.full(output_size, 255, dtype=torch.uint8)

    for sample_index, sample in enumerate(samples):
        source_size = original_sizes[sample_index]
        spatial = parameters[sample_index] if parameters is not None else None
        for annotation in sample["annotations"]:
            supervision = annotation["supervision_level"]
            instances = torch.empty(0, *output_size, dtype=torch.bool)
            xyxy, cxcywh = _normalized_boxes(annotation, source_size)
            if supervision == "box":
                instances = _rasterize_boxes(annotation, source_size, output_size, spatial)
                if spatial is not None:
                    if not len(instances):
                        continue
                    xyxy, cxcywh = _boxes_from_masks(instances)
            rows["annotations"].append(annotation)
            rows["sample_indices"].append(sample_index)
            rows["prompts"].append(_select_prompt(annotation, prompt_strategy, seed))

            rows["boxes_xyxy"].append(xyxy)
            rows["boxes"].append(cxcywh)

            if supervision == "pixel":
                rows["pixel_targets"].append(
                    _resize_mask(annotation["target"], output_size, torch.uint8, parameters=spatial)
                )
                rows["scene_targets"].append(float("nan"))
            elif supervision == "box":
                rows["pixel_targets"].append(ignore_target)
                rows["scene_targets"].append(float("nan"))
            else:
                instances = torch.empty(0, *output_size, dtype=torch.bool)
                rows["pixel_targets"].append(ignore_target)
                rows["scene_targets"].append(float(bool(annotation["target"])))
            rows["instance_box_masks"].append(instances)
    return rows


def _spatial_augmentation_parameters(
    sample_id: str,
    seed: int,
    size: tuple[int, int],
    scale_range: tuple[float, float],
    output_size: tuple[int, int],
) -> dict[str, int]:
    height, width = size
    generator = torch.Generator().manual_seed(
        int(hashlib.sha1(f"{seed}:{sample_id}:spatial".encode()).hexdigest()[:15], 16)
    )
    minimum, maximum = scale_range
    scale = minimum + (maximum - minimum) * float(torch.rand((), generator=generator))
    base_height, base_width = minimum_size((height, width), output_size)
    scale = max(scale, output_size[0] / base_height, output_size[1] / base_width)
    scaled_height = max(output_size[0], round(base_height * scale))
    scaled_width = max(output_size[1], round(base_width * scale))
    output_height, output_width = output_size
    crop_top = int(torch.randint(max(1, scaled_height - output_height + 1), (), generator=generator))
    crop_left = int(torch.randint(max(1, scaled_width - output_width + 1), (), generator=generator))
    pad_top = int(torch.randint(max(1, output_height - scaled_height + 1), (), generator=generator))
    pad_left = int(torch.randint(max(1, output_width - scaled_width + 1), (), generator=generator))
    return {
        "output_height": output_height, "output_width": output_width,
        "scaled_height": scaled_height,
        "scaled_width": scaled_width,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "horizontal_flip": int(torch.rand((), generator=generator) < 0.5),
        "vertical_flip": int(torch.rand((), generator=generator) < 0.5),
        "rotation": int(torch.randint(4, (), generator=generator)),
    }


def _apply_spatial_augmentation(
    tensor: torch.Tensor,
    parameters: dict[str, int],
    *,
    mode: str,
    fill: int | float,
) -> torch.Tensor:
    if not tensor.numel():
        return tensor
    dtype = tensor.dtype
    source_height, source_width = tensor.shape[-2:]
    height, width = parameters["output_height"], parameters["output_width"]
    leading_shape = tensor.shape[:-2]
    resized = F.interpolate(
        tensor.reshape(-1, 1, source_height, source_width).float(),
        size=(parameters["scaled_height"], parameters["scaled_width"]),
        mode=mode,
        **({"align_corners": False, "antialias": True} if mode == "bilinear" else {}),
    ).reshape(*leading_shape, parameters["scaled_height"], parameters["scaled_width"])
    cropped = resized[
        ...,
        parameters["crop_top"] : parameters["crop_top"] + height,
        parameters["crop_left"] : parameters["crop_left"] + width,
    ]
    output = torch.full((*leading_shape, height, width), fill, dtype=resized.dtype)
    paste_height = min(height - parameters["pad_top"], cropped.shape[-2])
    paste_width = min(width - parameters["pad_left"], cropped.shape[-1])
    output[
        ...,
        parameters["pad_top"] : parameters["pad_top"] + paste_height,
        parameters["pad_left"] : parameters["pad_left"] + paste_width,
    ] = cropped[..., :paste_height, :paste_width]
    output = _orient_spatial(output, parameters)
    return output.gt(0.5) if dtype == torch.bool else output.to(dtype)


def _orient_spatial(output, parameters):
    if parameters["horizontal_flip"]:
        output = output.flip(-1)
    if parameters["vertical_flip"]:
        output = output.flip(-2)
    output = torch.rot90(output, parameters["rotation"], (-2, -1))
    return output


def _boxes_from_masks(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = masks.shape[-2:]
    xyxy = []
    for mask in masks:
        y, x = mask.nonzero(as_tuple=True)
        xyxy.append(
            torch.tensor(
                [x.min() / width, y.min() / height, (x.max() + 1) / width, (y.max() + 1) / height]
            )
        )
    normalized_xyxy = torch.stack(xyxy) if xyxy else torch.empty(0, 4)
    if not len(normalized_xyxy):
        return normalized_xyxy, normalized_xyxy.clone()
    x1, y1, x2, y2 = normalized_xyxy.unbind(1)
    cxcywh = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), 1)
    return normalized_xyxy, cxcywh


def text_prompt_collate(
    samples: Sequence[dict[str, Any]],
    *,
    prompt_strategy: str = "random",
    seed: int = 0,
    spatial_size: int | tuple[int, int] | None = None,
    spatial_augmentation: bool = False,
    scale_range: tuple[float, float] = (0.75, 1.50),
) -> dict[str, Any]:
    if prompt_strategy not in {"first", "random"}:
        raise ValueError("prompt_strategy must be 'first' or 'random'")
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    target_size = _spatial_size(spatial_size)
    parameters = None
    if spatial_augmentation:
        if target_size is None or target_size[0] != target_size[1]:
            raise ValueError("Scale/crop augmentation requires a square spatial_size")
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError("Invalid augmentation scale range")
        parameters = [
            _spatial_augmentation_parameters(
                str(sample["sample_id"]), seed, tuple(sample["image"].shape[:2]), scale_range, target_size
            )
            for sample in samples
        ]
    images, valid_image_masks, original_sizes = _collate_images(samples, target_size, parameters)
    output_size = tuple(map(int, images.shape[-2:]))
    rows = _collate_annotations(
        samples, original_sizes, output_size, prompt_strategy, seed, parameters
    )
    flat = rows["annotations"]
    sample_indices = rows["sample_indices"]
    pixel_targets = (
        torch.stack(rows["pixel_targets"])
        if flat
        else torch.empty(0, *output_size, dtype=torch.uint8)
    )
    scene_targets = torch.tensor(rows["scene_targets"], dtype=torch.float32)
    height, width = output_size
    dataset_local_pixel_masks = torch.stack(
        [
            (
                torch.full(output_size, 255, dtype=torch.uint8)
                if sample.get("dataset_local_pixel_mask") is None
                else _resize_mask(
                    sample["dataset_local_pixel_mask"], output_size, torch.uint8,
                    parameters=parameters[index] if parameters is not None else None,
                )
            )
            for index, sample in enumerate(samples)
        ]
    )
    dataset_local_pixel_valid_masks = torch.stack(
        [
            (
                torch.zeros(output_size, dtype=torch.bool)
                if sample.get("dataset_local_pixel_valid_mask") is None
                else _resize_mask(
                    sample["dataset_local_pixel_valid_mask"], output_size, torch.bool,
                    parameters=parameters[index] if parameters is not None else None, fill=0,
                )
            )
            for index, sample in enumerate(samples)
        ]
    )
    semantic_levels = [annotation["semantic_level"] for annotation in flat]
    task_type = (
        semantic_levels[0]
        if semantic_levels and len(set(semantic_levels)) == 1
        else None
    )
    batch = {
        "images": images,
        "valid_image_masks": valid_image_masks,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "datasets": [sample["dataset"] for sample in samples],
        "dataset_vocabularies": [
            sample.get("dataset_vocabularies", {}) for sample in samples
        ],
        "scene_local_class_ids": [
            list(map(int, sample.get("scene_local_class_ids", [])))
            for sample in samples
        ],
        "dataset_local_pixel_masks": dataset_local_pixel_masks,
        "dataset_local_pixel_valid_masks": dataset_local_pixel_valid_masks,
        "sample_levels": [str(sample.get("level", "")) for sample in samples],
        "annotations": flat,
        "prompt_texts": rows["prompts"],
        "task_type": task_type,
        "class_ids": torch.tensor(
            [int(annotation["dataset_class_id"]) for annotation in flat],
            dtype=torch.long,
        ),
        "annotation_sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "pixel_targets": pixel_targets,
        "valid_mask": (
            torch.stack(
                [
                    valid_image_masks[sample_index]
                    & (
                        pixel_targets[index].ne(255)
                        if flat[index]["supervision_level"] == "pixel"
                        else torch.ones((height, width), dtype=torch.bool)
                    )
                    for index, sample_index in enumerate(sample_indices)
                ]
            )
            if flat
            else torch.empty(0, height, width, dtype=torch.bool)
        ),
        "boxes": rows["boxes"],
        "boxes_xyxy": rows["boxes_xyxy"],
        "instance_box_masks": rows["instance_box_masks"],
        "scene_targets": scene_targets,
        "semantic_levels": semantic_levels,
        "supervision_levels": [annotation["supervision_level"] for annotation in flat],
        "supervision_sources": [
            str(
                annotation.get("supervision_source")
                or annotation.get("annotation_quality", {}).get("source")
                or (
                    "native_image_annotation"
                    if annotation["supervision_level"] == "image"
                    else "native_pixel_annotation"
                )
            )
            for annotation in flat
        ],
    }
    return batch
