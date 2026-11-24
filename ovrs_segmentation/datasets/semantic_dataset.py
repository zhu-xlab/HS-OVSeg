from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from .class_prompts import class_specs_by_id
from .lmdb_store import open_lmdb_store


FORMAT_VERSION = "ovrs-semantic-v1"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MASK_SUFFIXES = (".png", ".tif", ".tiff")


class CorruptSampleError(OSError):
    """A prepared image or mask exists but cannot be decoded completely."""


@dataclass(frozen=True)
class SemanticClass:
    class_id: int
    class_text: str


def load_class_texts(path: str | Path) -> dict[int, str]:
    """Load a dataset-local class-id to class-text mapping.

    The representation is the structured ``ovrs-classes-v1`` catalog.
    """
    source = Path(path)
    mapping = {
        class_id: row["class_text"]
        for class_id, row in class_specs_by_id(source).items()
    }
    if not mapping or any(not text for text in mapping.values()):
        raise ValueError(f"Class map contains an empty class text: {source}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Class texts must be unique inside one dataset: {source}")
    return dict(sorted(mapping.items()))


class SemanticSegmentationDataset(Sequence[dict[str, Any]]):
    """Unified reader for converted detection and native segmentation datasets.

    Class IDs are deliberately dataset-local. Consumers align datasets through
    ``class_text_by_id`` instead of assuming that numeric IDs match globally.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        *,
        validate_masks: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / "dataset.json"
        with metadata_path.open("r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)
        self.storage = open_lmdb_store(self.root, self.metadata)
        self.dataset_name = str(self.metadata["dataset_name"])
        self.validate_masks = bool(validate_masks)
        self.format_version = str(self.metadata.get("format_version", FORMAT_VERSION))
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported semantic dataset format {self.format_version!r}; expected {FORMAT_VERSION!r}"
            )
        self.ignore_id = int(self.metadata.get("ignore_id", 255))
        raw_background_id = self.metadata.get("background_id", 0)
        self.background_id = None if raw_background_id is None else int(raw_background_id)
        self.image_dir = self.root / self.metadata.get("image_dir", "images")
        self.mask_dir = self.root / self.metadata.get("mask_dir", "masks")
        self.box_only = bool(
            self.metadata.get("annotations_dir")
            and str(self.metadata.get("supervision_level", "")).lower() == "box"
        )
        self.class_text_by_id = load_class_texts(
            self.root / self.metadata.get("classes_file", "classes.json")
        )
        self.class_specs_by_id = class_specs_by_id(
            self.root / self.metadata.get("classes_file", "classes.json")
        )
        self.classes = tuple(
            SemanticClass(class_id, class_text)
            for class_id, class_text in self.class_text_by_id.items()
        )
        if self.background_id is not None and self.background_id not in self.class_text_by_id:
            raise ValueError(f"background_id {self.background_id} is absent from classes.json")
        self.sample_ids = self._load_sample_ids(split)
        self._images = self._index_files(self.image_dir, IMAGE_SUFFIXES)
        self._masks = (
            self._index_files(self.mask_dir, MASK_SUFFIXES)
            if self.mask_dir.is_dir() or (self.storage and self.storage.has_prefix(self.mask_dir))
            else {}
        )
        self.optional_modalities: dict[str, dict[str, Any]] = {}
        for name, config in self.metadata.get("optional_modalities", {}).items():
            directory = self.root / str(config["directory"])
            indexed = self._index_files(directory, IMAGE_SUFFIXES) if directory.is_dir() or (self.storage and self.storage.has_prefix(directory)) else {}
            self.optional_modalities[str(name)] = {
                "files": indexed,
            }
        missing_images = [sample_id for sample_id in self.sample_ids if sample_id not in self._images]
        missing_masks = (
            []
            if self.box_only
            else [
                sample_id
                for sample_id in self.sample_ids
                if sample_id not in self._masks
            ]
        )
        if missing_images or missing_masks:
            raise FileNotFoundError(
                f"Dataset {self.dataset_name} has {len(missing_images)} missing images and "
                f"{len(missing_masks)} missing masks"
            )
        if self.validate_masks and not self.box_only:
            for sample_id in self.sample_ids:
                self._validate_mask(sample_id, self._read_mask(self._masks[sample_id]))

    def _load_sample_ids(self, split: str | None) -> list[str]:
        if split is not None:
            split_path = self.root / "splits" / f"{split}.txt"
            if not split_path.is_file():
                raise FileNotFoundError(f"Unknown split {split!r}: {split_path}")
            values = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
            return [value for value in values if value]
        if self.box_only:
            return sorted(self._index_files(self.image_dir, IMAGE_SUFFIXES))
        return sorted(self._index_files(self.mask_dir, MASK_SUFFIXES))

    def _index_files(self, directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
        if self.storage is not None:
            return self.storage.index_files(directory, suffixes)
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        indexed: dict[str, Path] = {}
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            sample_id = path.relative_to(directory).with_suffix("").as_posix()
            if sample_id in indexed:
                raise ValueError(f"Duplicate sample ID {sample_id!r} below {directory}")
            indexed[sample_id] = path
        return indexed

    def _read_mask(self, path: Path) -> np.ndarray:
        try:
            source = BytesIO(self.storage.read(path)) if self.storage else path
            with Image.open(source) as stream:
                mask = np.asarray(stream)
        except (UnidentifiedImageError, OSError, SyntaxError) as error:
            size = path.stat().st_size if path.is_file() else None
            raise CorruptSampleError(
                f"Unreadable class-ID mask: {path} (size={size} bytes). "
                "The prepared file is truncated or is not a valid PNG/TIFF; "
                "rerun its dataset preparation command with --overwrite."
            ) from error
        if mask.ndim != 2:
            raise ValueError(f"Class-ID mask must be single-channel, got shape {mask.shape}: {path}")
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Class-ID mask must use an integer dtype, got {mask.dtype}: {path}")
        return mask

    def _read_rgb(self, path: Path) -> np.ndarray:
        try:
            image: np.ndarray
            source = BytesIO(self.storage.read(path)) if self.storage else path
            if path.suffix.lower() in {".tif", ".tiff"}:
                try:
                    import tifffile

                    image = np.asarray(tifffile.imread(source))
                except (ImportError, ValueError, OSError):
                    if self.storage:
                        source.seek(0)
                    with Image.open(source) as stream:
                        image = np.asarray(stream)
            else:
                with Image.open(source) as stream:
                    image = np.asarray(stream)
        except (UnidentifiedImageError, OSError, SyntaxError) as error:
            size = path.stat().st_size if path.is_file() else None
            raise CorruptSampleError(
                f"Unreadable RGB image: {path} (size={size} bytes). "
                "The file is truncated or is not a valid image; restore or "
                "regenerate this prepared sample."
            ) from error
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        rgb = image
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"Unified image must already be stored as three-channel RGB, got {rgb.shape}: {path}"
            )
        return rgb

    def _validate_mask(self, sample_id: str, mask: np.ndarray) -> None:
        valid_ids = set(self.class_text_by_id) | {self.ignore_id}
        unknown = set(map(int, np.unique(mask))) - valid_ids
        if unknown:
            raise ValueError(f"Mask {sample_id!r} contains IDs absent from classes.json: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def image_only_sample(self, index: int) -> dict[str, Any]:
        """Load RGB and metadata for a native-box Object sample."""
        sample_id = self.sample_ids[index]
        image_path = self._images[sample_id]
        image = self._read_rgb(image_path)
        modalities: dict[str, np.ndarray] = {}
        modality_paths: dict[str, Path] = {}
        for name, config in self.optional_modalities.items():
            modality_path = config["files"].get(sample_id)
            if modality_path is None:
                continue
            modalities[name] = self._read_rgb(modality_path)
            modality_paths[name] = modality_path
        return {
            "sample_id": sample_id,
            "dataset_name": self.dataset_name,
            "image": image,
            "class_text_by_id": self.class_text_by_id,
            "background_id": self.background_id,
            "ignore_id": self.ignore_id,
            "image_path": image_path,
            "modalities": modalities,
            "modality_paths": modality_paths,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        image_path, mask_path = self._images[sample_id], self._masks[sample_id]
        image = self._read_rgb(image_path)
        mask = self._read_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {sample_id}: {image.shape[:2]} versus {mask.shape}"
            )
        if self.validate_masks:
            self._validate_mask(sample_id, mask)
        present_ids = [
            int(class_id)
            for class_id in np.unique(mask)
            if int(class_id) != self.ignore_id and int(class_id) in self.class_text_by_id
        ]
        modalities: dict[str, np.ndarray] = {}
        modality_paths: dict[str, Path] = {}
        for name, config in self.optional_modalities.items():
            modality_path = config["files"].get(sample_id)
            if modality_path is None:
                continue
            modalities[name] = self._read_rgb(modality_path)
            modality_paths[name] = modality_path
        return {
            "sample_id": sample_id,
            "dataset_name": self.dataset_name,
            "image": image,
            "class_id_mask": mask,
            "class_text_by_id": self.class_text_by_id,
            "present_class_ids": present_ids,
            "present_class_texts": [self.class_text_by_id[class_id] for class_id in present_ids],
            "background_id": self.background_id,
            "ignore_id": self.ignore_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "modalities": modalities,
            "modality_paths": modality_paths,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]
