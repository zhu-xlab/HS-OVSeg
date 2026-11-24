from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ovrs_segmentation.datasets.lmdb_store import META_KEY
from ovrs_segmentation.datasets.class_prompts import (
    CLASS_CATALOG_FORMAT_VERSION,
    NATIVE_CLASSES_BACKUP,
    build_class_catalog,
)


DATASET_VARIANTS = {
    "FLAIR-HUB": "CoSIA",
    "Global-LoveDA": "semantic segmentation",
    "SkyScapes": "Dense",
}


def _write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _update_lmdb_metadata(path: Path, dataset_name: str) -> None:
    import lmdb

    environment = lmdb.open(str(path), subdir=True, readonly=False, lock=True)
    info = environment.info()
    used_bytes = (info["last_pgno"] + 1) * environment.stat()["psize"]
    environment.set_mapsize(max(info["map_size"], used_bytes + 64 * 1024 * 1024))
    with environment.begin(write=True) as transaction:
        raw = transaction.get(META_KEY)
        metadata = json.loads(raw.decode("utf-8")) if raw else {}
        metadata["dataset_name"] = dataset_name
        transaction.put(META_KEY, json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
    environment.close()


def _normalize_classes(root: Path, metadata: dict) -> None:
    path = root / str(metadata.get("classes_file", "classes.json"))
    catalog = json.loads(path.read_text(encoding="utf-8"))
    rows = catalog.get("classes") if isinstance(catalog, dict) else None
    if isinstance(rows, list):
        classes = {int(row["class_id"]): str(row["class_text"]) for row in rows}
        scopes = {
            int(row["class_id"]): str(row["semantic_level"])
            for row in rows
            if row.get("semantic_level")
        }
        excluded = [int(row["class_id"]) for row in rows if row.get("is_excluded")]
    else:
        classes = catalog
        scopes = {}
        excluded = list(metadata.get("ignored_native_class_ids", []))
        backup = path.with_name(NATIVE_CLASSES_BACKUP)
        if not backup.exists():
            _write_json(backup, catalog)
    backup = path.with_name(NATIVE_CLASSES_BACKUP)
    if backup.is_file():
        native = json.loads(backup.read_text(encoding="utf-8"))
        native_rows = native.get("classes") if isinstance(native, dict) else None
        classes = (
            {int(row["class_id"]): str(row["class_text"]) for row in native_rows}
            if isinstance(native_rows, list)
            else {int(class_id): str(text) for class_id, text in native.items()}
        )
    default_level = (
        "scene"
        if metadata.get("native_supervision") == "scene_classification"
        else "object"
        if str(metadata.get("supervision_level", "")).lower() == "box"
        else "lulc"
    )
    scopes.update(
        {int(class_id): str(level) for class_id, level in metadata.get("query_scope_by_class", {}).items()}
    )
    catalog = build_class_catalog(
        root.name,
        classes,
        default_semantic_level=default_level,
        scope_by_class=scopes,
        background_id=metadata.get("background_id"),
        ignore_id=int(metadata.get("ignore_id", 255)),
        excluded_class_ids=excluded,
    )
    _write_json(path, catalog)
    metadata["classes_format"] = CLASS_CATALOG_FORMAT_VERSION


def normalize_dataset_name(root: Path) -> dict[str, str]:
    root = root.resolve()
    metadata_path = root / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous = str(metadata.get("dataset_name") or root.name)
    metadata["dataset_name"] = root.name
    if metadata.get("dataset_id") == previous:
        metadata["dataset_id"] = root.name
    variant = DATASET_VARIANTS.get(root.name)
    if variant:
        metadata["dataset_variant"] = variant
    if metadata.get("native_supervision") == "scene_classification":
        metadata.pop("format_version", None)
        metadata.update(
            {
                "semantic_level": "scene",
                "supervision_level": "image",
            }
        )
    _normalize_classes(root, metadata)
    _write_json(metadata_path, metadata)

    storage = metadata.get("storage", {})
    if storage.get("backend") == "lmdb":
        _update_lmdb_metadata(root / str(storage.get("path", "data.lmdb")), root.name)
    return {"directory": root.name, "previous": previous, "dataset_name": root.name}


def main() -> None:
    parser = argparse.ArgumentParser(description="Use dataset directory names as canonical dataset identifiers")
    parser.add_argument("--parent-root", type=Path, required=True)
    args = parser.parse_args()
    roots = sorted(path for path in args.parent_root.iterdir() if (path / "dataset.json").is_file())
    for root in roots:
        print(json.dumps(normalize_dataset_name(root), ensure_ascii=False))


if __name__ == "__main__":
    main()
