from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CLASS_CATALOG_FORMAT_VERSION = "ovrs-classes-v1"
SEMANTIC_LEVELS = {"scene", "lulc", "object"}
GENERAL_PROMPT_FORMATS = {
    "scene": ("{class_text}", "{class_text} scene"),
    "lulc": ("{class_text}", "{class_text} area"),
    "object": ("{class_text}",),
}
PROMPT_MODE = "sam3_noun_phrase"

PROMPT_VOCABULARY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "text_prompt_vocabulary.yaml"
)
NATIVE_CLASSES_BACKUP = "classes.before-text-prompts.json"


def normalize_class_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


@lru_cache(maxsize=1)
def _global_vocabulary() -> dict[str, dict[str, tuple[str, tuple[str, ...]]]]:
    vocabulary = yaml.safe_load(PROMPT_VOCABULARY_PATH.read_text(encoding="utf-8"))
    by_dataset = {}
    for dataset, rows in vocabulary["datasets"].items():
        dataset_rows = {}
        for native, row in rows.items():
            native = normalize_class_text(native)
            canonical = normalize_class_text(row["canonical_name"])
            aliases = tuple(
                dict.fromkeys(normalize_class_text(value) for value in row.get("aliases", []))
            )
            entry = (canonical, aliases)
            dataset_rows[native] = entry
        by_dataset[normalize_class_text(dataset)] = dataset_rows
    return by_dataset


def _vocabulary_entry(class_text: str, dataset_name: str) -> tuple[str, tuple[str, ...]]:
    normalized = normalize_class_text(class_text)
    return _global_vocabulary().get(normalize_class_text(dataset_name), {}).get(
        normalized, (normalized, ())
    )


def canonical_prompt_text(class_text: str, dataset_name: str) -> str:
    """Return the canonical text assigned to one dataset-native class."""
    return _vocabulary_entry(class_text, dataset_name)[0]


def global_alias_prompt_texts(class_text: str, dataset_name: str) -> list[str]:
    """Return reviewed aliases assigned to one dataset-native class."""
    return list(_vocabulary_entry(class_text, dataset_name)[1])


def is_level_prompt(value: str, class_text: str, dataset_name: str) -> bool:
    """Identify automatic ``scene``/``area`` training variants."""
    prompt = normalize_class_text(value)
    canonical = canonical_prompt_text(class_text, dataset_name)
    return prompt in {f"{canonical} scene", f"{canonical} area"}


def default_prompt_texts(
    class_text: str, semantic_level: str, dataset_name: str
) -> list[str]:
    """Return the canonical noun phrase followed by training-only variants."""
    text = canonical_prompt_text(class_text, dataset_name)
    if semantic_level not in GENERAL_PROMPT_FORMATS:
        raise ValueError(f"Unsupported semantic level: {semantic_level!r}")
    values = [
        template.format(class_text=text)
        for template in GENERAL_PROMPT_FORMATS[semantic_level]
    ]
    return list(dict.fromkeys(values))


def combined_prompt_texts(
    class_text: str,
    semantic_level: str,
    dataset_name: str,
) -> dict[str, list[str]]:
    native = normalize_class_text(class_text)
    canonical = canonical_prompt_text(class_text, dataset_name)
    canonical_variants = list(
        default_prompt_texts(class_text, semantic_level, dataset_name)
    )[1:]
    aliases = global_alias_prompt_texts(class_text, dataset_name)
    variants = list(
        dict.fromkeys(
            value
            for value in (*canonical_variants, *aliases)
            if value not in {native, canonical}
        )
    )
    return {
        "native_prompt_texts": [native],
        "canonical_prompt_texts": [canonical],
        "variant_prompt_texts": variants,
        "prompt_texts": list(dict.fromkeys((native, canonical, *variants))),
    }


def _optimization_targets(semantic_level: str | None) -> list[str]:
    if semantic_level is None:
        return []
    if semantic_level == "scene":
        return ["region_text_alignment", "image_presence"]
    if semantic_level == "object":
        return ["instance_region_attention_bce"]
    return ["region_text_alignment", "mask_localization"]


def build_class_catalog(
    dataset_name: str,
    classes: Mapping[int | str, str],
    *,
    default_semantic_level: str,
    scope_by_class: Mapping[int | str, str] | None = None,
    background_id: int | None = None,
    ignore_id: int = 255,
    excluded_class_ids: Sequence[int | str] | None = None,
    exclusion_reason: str = "excluded from text-prompt supervision",
) -> dict[str, Any]:
    """Build a dataset-local text-prompt catalog.

    Numeric IDs remain local to ``dataset_name``. Background and ignore labels
    are retained for mask decoding but are deliberately not text queries.
    """
    if default_semantic_level not in SEMANTIC_LEVELS:
        raise ValueError(f"Unsupported semantic level: {default_semantic_level!r}")
    scopes = {int(key): str(value) for key, value in (scope_by_class or {}).items()}
    excluded = {int(value) for value in (excluded_class_ids or [])}
    rows = []
    for raw_id, raw_text in sorted(classes.items(), key=lambda item: int(item[0])):
        class_id = int(raw_id)
        class_text = normalize_class_text(raw_text)
        is_background = background_id is not None and class_id == int(background_id)
        is_ignore = class_id == int(ignore_id) or class_text in {"ignore", "ignored", "void"}
        is_excluded = class_id in excluded
        queryable = not (is_background or is_ignore or is_excluded)
        semantic_level = scopes.get(class_id, default_semantic_level) if queryable else None
        if semantic_level is not None and semantic_level not in SEMANTIC_LEVELS:
            raise ValueError(
                f"Unsupported semantic level {semantic_level!r} for class {class_id}"
            )
        prompt_groups = {
            "native_prompt_texts": [],
            "canonical_prompt_texts": [],
            "variant_prompt_texts": [],
            "prompt_texts": [],
        }
        if queryable:
            prompt_groups = combined_prompt_texts(
                class_text,
                str(semantic_level),
                dataset_name,
            )
        rows.append({
            "class_id": class_id,
            "class_text": class_text,
            **prompt_groups,
            "semantic_level": semantic_level,
            "supervision_level": (
                None
                if semantic_level is None
                else {"scene": "image", "object": "box", "lulc": "pixel"}[semantic_level]
            ),
            "optimization_targets": _optimization_targets(semantic_level),
            "queryable": queryable,
            "is_background": is_background,
            "is_ignore": is_ignore,
            "is_excluded": is_excluded,
            "exclusion_reason": exclusion_reason if is_excluded else None,
        })
    return {
        "format_version": CLASS_CATALOG_FORMAT_VERSION,
        "dataset": str(dataset_name),
        "class_ids_are_dataset_local": True,
        "prompt_mode": PROMPT_MODE,
        "classes": rows,
    }


def write_class_catalog(
    path: str | Path,
    dataset_name: str,
    classes: Mapping[int | str, str],
    **kwargs: Any,
) -> dict[str, Any]:
    catalog = build_class_catalog(dataset_name, classes, **kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def load_class_catalog(path: str | Path) -> dict[str, Any]:
    """Load one structured class catalog."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format_version") != CLASS_CATALOG_FORMAT_VERSION:
        raise ValueError(f"Expected {CLASS_CATALOG_FORMAT_VERSION}: {source}")
    if not isinstance(raw.get("classes"), list):
        raise ValueError(f"Class catalog has no classes list: {source}")
    return raw


def _native_class_texts(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("classes") if isinstance(raw, dict) else None
    if isinstance(rows, list):
        return {int(row["class_id"]): str(row["class_text"]).strip() for row in rows}
    if isinstance(raw, dict):
        return {int(class_id): str(text).strip() for class_id, text in raw.items()}
    raise ValueError(f"Invalid native class-name backup: {path}")


def class_specs_by_id(path: str | Path) -> dict[int, dict[str, Any]]:
    source = Path(path)
    catalog = load_class_catalog(source)
    backup_path = source.with_name(NATIVE_CLASSES_BACKUP)
    native_texts = _native_class_texts(backup_path) if backup_path.is_file() else {}
    result: dict[int, dict[str, Any]] = {}
    for raw in catalog["classes"]:
        row = dict(raw)
        class_id = int(row["class_id"])
        class_text = native_texts.get(
            class_id,
            str(row["class_text"]).strip(),
        )
        if not class_text:
            raise ValueError(f"Class {class_id} contains an empty class text: {path}")
        row["class_id"] = class_id
        row["class_text"] = class_text
        if class_id in result:
            raise ValueError(f"Duplicate class ID {class_id}: {path}")
        result[class_id] = row
    if not result:
        raise ValueError(f"Class catalog is empty: {path}")
    return dict(sorted(result.items()))
