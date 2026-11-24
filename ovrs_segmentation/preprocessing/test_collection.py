"""Materialize immutable selected test pairs beside manually reviewed test sets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ovrs_segmentation.test_catalog import DEFAULT_CATALOG, dataset_identity, digest, load_suite


def export_test(source, destination):
    from ovrs_segmentation.datasets.semantic_dataset import SemanticSegmentationDataset
    source, destination = Path(source).resolve(), Path(destination).resolve()
    dataset = SemanticSegmentationDataset(source, "test", validate_masks=False)
    if dataset.box_only or not len(dataset):
        raise ValueError(f"Pixel test GT required: {source}")
    if len(dataset) > 2000:
        raise ValueError(f"Keep the existing <=2000 selected test IDs: {source}")
    if source == destination:
        return dict(samples=len(dataset), original_identity=dataset_identity(source), already_centralized=True)
    identity = dataset_identity(source)
    if destination.exists():
        if dataset_identity(destination) != identity:
            raise ValueError(f"Conflicting destination: {destination}")
        return dict(samples=len(dataset), original_identity=identity, already_exported=True)
    pending = destination.with_name(destination.name + ".partial")
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "splits").mkdir(exist_ok=True)
    rows, total_bytes = [], 0
    for sample_id in dataset.sample_ids:
        row = dict(sample_id=sample_id)
        for kind, indexed in (("image", dataset._images), ("mask", dataset._masks)):
            original = indexed[sample_id]
            relative = Path("images" if kind == "image" else "masks") / (sample_id + original.suffix)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe sample ID: {sample_id}")
            target = pending / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = dataset.storage.read(original) if dataset.storage else original.read_bytes()
            expected = hashlib.sha256(data).hexdigest()
            if not target.exists() or digest(target) != expected:
                target.write_bytes(data)
            if digest(target) != expected:
                raise OSError(f"Copy verification failed: {target}")
            row[kind] = relative.as_posix()
            row[kind + "_sha256"] = expected
            row[kind + "_bytes"] = len(data)
            total_bytes += len(data)
        rows.append(row)
    shutil.copy2(source / "dataset.json", pending / "source_dataset.json")
    shutil.copy2(source / dataset.metadata.get("classes_file", "classes.json"), pending / "classes.json")
    shutil.copy2(source / "splits/test.txt", pending / "splits/test.txt")
    pairs = pending / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    metadata = dict(dataset.metadata)
    metadata.pop("storage", None)
    metadata.pop("optional_modalities", None)
    metadata.update(image_dir="images", mask_dir="masks", classes_file="classes.json",
        splits={"test": len(dataset)}, evaluation_only=True,
        test_collection=dict(original_identity=identity, pairs_sha256=digest(pairs),
                             source_dataset=str(source), pixels_and_encoding_unchanged=True))
    (pending / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    copied = SemanticSegmentationDataset(pending, "test", validate_masks=False)
    if copied.sample_ids != dataset.sample_ids or dataset_identity(pending) != identity:
        raise ValueError(f"Test identity changed during export: {source}")
    pending.rename(destination)
    return dict(samples=len(dataset), bytes=total_bytes, original_identity=identity, copied=True)


def prepare(data_root, test_root, catalog=DEFAULT_CATALOG):
    data_root, test_root = Path(data_root).resolve(), Path(test_root).resolve()
    test_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in load_suite(catalog, data_root, test_root):
        destination = entry["root"]
        source = destination if destination.exists() else data_root / entry["name"]
        result = export_test(source, destination)
        row = dict(name=entry["name"], directory=destination.name, source_task=entry["source_task"],
                   evaluation_scope=entry["evaluation_scope"], **result)
        rows.append(row)
        print(json.dumps(row), flush=True)
        temporary = test_root / "registry.json.partial"
        temporary.write_text(json.dumps(dict(complete=len(rows) == len(load_suite(catalog, data_root, test_root)), datasets=rows), indent=2) + "\n")
        temporary.replace(test_root / "registry.json")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    prepare(args.data_root, args.test_root or args.data_root / "segmentation_test", args.catalog)


if __name__ == "__main__":
    main()
