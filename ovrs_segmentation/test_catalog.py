"""One location-independent catalog for source and out-of-domain test sets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "configs/segmentation_test.yaml"
SCOPES = ("all", "training_source", "out_of_domain")


def load_suite(path, data_root, test_root=None, *, scope="all"):
    if scope not in SCOPES:
        raise ValueError(f"Unknown evaluation scope: {scope}")
    data_root = Path(data_root)
    test_root = Path(test_root) if test_root is not None else data_root / "segmentation_test"
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["datasets"]
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Test suite contains duplicate dataset names")
    for entry in entries:
        if entry.get("evaluation_scope", "training_source") not in SCOPES[1:]:
            raise ValueError(f"Invalid evaluation scope for {entry['name']}")
    return [{**entry, "root": Path(str(entry["root"]).format(data_root=data_root, test_root=test_root))}
            for entry in entries if scope == "all" or entry.get("evaluation_scope", scope) == scope]


def canonicalize_entries(entries):
    """Map persisted benchmark inventories to the shared /datasets test mount."""
    catalog = {row["name"]: row for row in load_suite(DEFAULT_CATALOG, Path("/datasets"))}
    result = []
    for entry in entries:
        target = catalog.get(entry["name"])
        result.append({**entry, "root": str(target["root"]), "directory": target["root"].name}
                      if target else dict(entry))
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def dataset_identity(root):
    """Preserve proven test identity when only storage and directory change."""
    root = Path(root)
    current = {name: digest(root / name) for name in ("dataset.json", "classes.json", "splits/test.txt")}
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    migration = metadata.get("test_collection")
    if not migration:
        return current
    original = migration["original_identity"]
    source = root / "source_dataset.json"
    if digest(source) != original["dataset.json"]:
        raise ValueError(f"Changed source provenance: {root}")
    prior = json.loads(source.read_text(encoding="utf-8"))
    for key in ("format_version", "dataset_name", "background_id", "ignore_id"):
        if metadata.get(key) != prior.get(key):
            raise ValueError(f"Changed semantic metadata {key}: {root}")
    if any(current[key] != original[key] for key in ("classes.json", "splits/test.txt")):
        raise ValueError(f"Changed vocabulary or test IDs: {root}")
    if digest(root / "pairs.jsonl") != migration["pairs_sha256"]:
        raise ValueError(f"Changed test-pair inventory: {root}")
    return original
