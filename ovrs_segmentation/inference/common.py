"""Small shared helpers for the native and fine-tuned inference CLIs."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import torch

from ovrs_segmentation.datasets.class_prompts import (
    canonical_prompt_text,
    class_specs_by_id,
    global_alias_prompt_texts,
    is_level_prompt,
    normalize_class_text,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LEVELS = {"scene", "object", "lulc"}
Q = TypeVar("Q")


def device(choice: str) -> torch.device:
    cuda = choice == "cuda" or (choice == "auto" and torch.cuda.is_available())
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if cuda else "cpu")


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda" and precision == "fp16":
        raise ValueError("fp16 inference requires CUDA")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype)


def discover_images(inputs: Sequence[Path]) -> list[Path]:
    images: list[Path] = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.is_dir():
            images.extend(
                child.resolve()
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            raise ValueError(f"Unsupported image extension: {path}")
        else:
            raise FileNotFoundError(path)
    images = list(dict.fromkeys(sorted(images)))
    if not images:
        raise FileNotFoundError("No supported images were found")
    return images


def _dataset_metadata(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.expanduser().resolve()
    path = root / "dataset.json"
    return root, json.loads(path.read_text(encoding="utf-8"))


def _split_ids(root: Path, split: str) -> list[str] | None:
    if split == "all":
        return None
    path = root / "splits" / f"{split}.txt"
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _annotation_rows(root: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / metadata.get("annotations_file", "annotations.jsonl")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset_split_images(dataset_root: Path, split: str) -> list[Path]:
    root, metadata = _dataset_metadata(dataset_root)
    rows_path = root / metadata.get("annotations_file", "annotations.jsonl")
    if rows_path.is_file():
        images = {}
        for row in _annotation_rows(root, metadata):
            value = row.get("image", row.get("image_path"))
            path = Path(value)
            images[str(row["sample_id"])] = (
                path if path.is_absolute() else root / path
            ).resolve()
    else:
        paths = discover_images([root / metadata.get("image_dir", "images")])
        images = {path.stem: path for path in paths}
    selected_ids = _split_ids(root, split)
    sample_ids = sorted(images) if selected_ids is None else selected_ids
    selected = [images[sample_id] for sample_id in sample_ids]
    if not selected:
        raise ValueError(f"Dataset split is empty: {root} ({split})")
    return selected


def usable_texts(
    spec: dict[str, Any], policy: str, dataset_name: str
) -> tuple[str, ...]:
    native = normalize_class_text(str(spec["class_text"]))
    values = [native]
    if policy == "all":
        values.append(canonical_prompt_text(native, dataset_name))
        values.extend(global_alias_prompt_texts(str(spec["class_text"]), dataset_name))
    return tuple(
        value
        for value in dict.fromkeys(
            str(value).strip() for value in values if str(value).strip()
        )
        if not is_level_prompt(value, str(spec["class_text"]), dataset_name)
    )


def _catalog_context(path: Path) -> tuple[dict[str, Any], str, int | None, int]:
    metadata_path = path.parent / "dataset.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    default_level = str(metadata.get("semantic_level", ""))
    if default_level not in LEVELS:
        if str(metadata.get("supervision_level", "")).lower() == "box":
            default_level = "object"
        else:
            default_level = "lulc"
    background = metadata.get("background_id")
    background_id = None if background is None else int(background)
    return metadata, default_level, background_id, int(metadata.get("ignore_id", 255))


def _queryable_class(
    source_id: int,
    spec: dict[str, Any],
    background_id: int | None,
    ignore_id: int,
) -> bool:
    is_background = bool(spec.get("is_background")) or (
        background_id is not None and source_id == background_id
    )
    is_ignore = bool(spec.get("is_ignore")) or source_id == ignore_id
    default = not (is_background or is_ignore or bool(spec.get("is_excluded")))
    return bool(spec.get("queryable", default)) and default


def load_queries(
    args: Any,
    make_query: Callable[[int, str, tuple[str, ...]], Q],
) -> list[Q]:
    if args.prompt:
        prompts = list(
            dict.fromkeys(text.strip() for text in args.prompt if text.strip())
        )
        return [
            make_query(index, prompt, (prompt,))
            for index, prompt in enumerate(prompts, 1)
        ]

    paths = (
        [args.classes_json]
        if isinstance(args.classes_json, Path)
        else list(args.classes_json or [])
    )
    queries: list[Q] = []
    index_by_text: dict[str, int] = {}
    used_ids: set[int] = set()
    for path in paths:
        metadata, default_level, background_id, ignore_id = _catalog_context(path)
        dataset_name = str(metadata.get("dataset_name") or path.parent.name)
        for source_id, spec in class_specs_by_id(path).items():
            level = default_level
            requested_level = getattr(args, "level", "auto")
            if (
                not _queryable_class(source_id, spec, background_id, ignore_id)
                or (requested_level != "auto" and level != requested_level)
            ):
                continue
            text = str(spec["class_text"])
            key = normalize_class_text(text)
            prompts = usable_texts(spec, args.prompt_policy, dataset_name)
            if key in index_by_text:
                index = index_by_text[key]
                old = queries[index]
                merged = tuple(dict.fromkeys((*old.prompt_texts, *prompts)))
                queries[index] = make_query(old.class_id, old.class_text, merged)
                continue
            class_id = int(source_id)
            if class_id < 0 or class_id in used_ids:
                class_id = 0
                while class_id in used_ids:
                    class_id += 1
            used_ids.add(class_id)
            index_by_text[key] = len(queries)
            queries.append(make_query(class_id, text, prompts))
    if not queries:
        raise ValueError(
            f"No queryable classes for level={getattr(args, 'level', 'auto')!r} were found"
        )
    return queries


def load_gt_samples(
    dataset_root: Path,
    split: str,
    prompt_policy: str,
    level: str,
    make_query: Callable[[int, str, tuple[str, ...]], Q],
) -> dict[Path, list[Q]]:
    """Map prepared images to their positive ground-truth prompts."""
    root, metadata = _dataset_metadata(dataset_root)
    classes_path = root / metadata.get("classes_file", "classes.json")
    specs = class_specs_by_id(classes_path)
    _, dataset_level, _, _ = _catalog_context(classes_path)
    if level != dataset_level:
        raise ValueError(
            f"Dataset {root.name!r} provides {dataset_level} supervision, not {level}"
        )
    ids_by_text = {
        normalize_class_text(str(spec["class_text"])): key
        for key, spec in specs.items()
    }
    split_ids = _split_ids(root, split)
    selected = None if split_ids is None else set(split_ids)
    samples: dict[Path, list[Q]] = {}
    for row in _annotation_rows(root, metadata):
        if selected is not None and str(row["sample_id"]) not in selected:
            continue
        queries: dict[int, Q] = {}
        for annotation in row.get("annotations", row.get("queries", [])):
            target = annotation.get("target", {})
            present = (
                target.get("present", annotation.get("present", True))
                if isinstance(target, dict)
                else annotation.get("present", target)
            )
            if not present:
                continue
            class_id = annotation.get(
                "dataset_class_id",
                annotation.get(
                    "class_id", row.get("dataset_class_id", row.get("class_id"))
                ),
            )
            if class_id is None:
                text = annotation.get(
                    "class_text",
                    annotation.get(
                        "label", row.get("class_text", row.get("label", ""))
                    ),
                )
                class_id = ids_by_text.get(normalize_class_text(str(text)))
            if class_id is None or int(class_id) not in specs:
                continue
            class_id = int(class_id)
            spec = {
                **specs[class_id],
                **{
                    key: value for key, value in annotation.items() if value is not None
                },
            }
            text = str(specs[class_id]["class_text"])
            prompts = usable_texts(
                {**spec, "class_text": text}, prompt_policy, str(metadata.get("dataset_name") or root.name)
            )
            queries[class_id] = make_query(class_id, text, prompts)
        if queries:
            path = Path(row.get("image", row.get("image_path")))
            samples[(path if path.is_absolute() else root / path).resolve()] = list(
                queries.values()
            )
    if selected is not None and level == "scene" and len(samples) != len(selected):
        raise ValueError(
            f"Not every sample in split={split!r} has a positive Scene label"
        )
    if not samples:
        raise ValueError(f"No GT {level} samples found below {root} for split={split}")
    return samples


def select_images(args: Any) -> list[Path]:
    if args.dataset_root is None:
        if not args.input:
            raise ValueError("--input is required unless --dataset-root is used")
        return discover_images(args.input)
    images = load_dataset_split_images(args.dataset_root, args.split)
    if args.input:
        requested = set(discover_images(args.input))
        images = [path for path in images if path in requested]
        if not images:
            raise ValueError("--input does not select an image in the requested split")
    return images


def image_ids(paths: Sequence[Path]) -> dict[Path, str]:
    counts = {
        stem: sum(path.stem == stem for path in paths)
        for stem in {path.stem for path in paths}
    }
    return {
        path: (
            path.stem
            if counts[path.stem] == 1
            else f"{path.stem}-{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
        )
        for path in paths
    }


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "class"


def query_source_suffix(args: Any) -> str:
    if not args.classes_json:
        return "prompts"
    labels = []
    for raw in args.classes_json:
        path = raw.expanduser()
        label = path.parent.name if path.name.lower() == "classes.json" else path.stem
        value = slug(label)
        if value not in labels:
            labels.append(value)
    return "__".join(labels)


def empty_semantic_map(queries: Sequence[Any], shape: tuple[int, int]) -> np.ndarray:
    """Create a mask canvas without reserving a valid class ID as background."""
    class_ids = {int(query.class_id) for query in queries}
    dtype = np.uint8 if max(query.class_id for query in queries) < 255 else np.uint16
    return np.full(shape, 255 if 0 in class_ids else 0, dtype=dtype)


def semantic_background_id(queries: Sequence[Any]) -> int | None:
    return None if any(int(query.class_id) == 0 for query in queries) else 0
