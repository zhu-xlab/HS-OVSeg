"""Export only train/validation and source-domain test samples as loose files."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
from pathlib import Path, PureWindowsPath
import tarfile

from ovrs_segmentation.datasets import open_text_prompt_dataset
from ovrs_segmentation.datasets.semantic_dataset import SemanticSegmentationDataset
from ovrs_segmentation.preprocessing.huggingface_data import (
    TRAIN_SUITE, dataset_entries, validate_datasets,
)
from ovrs_segmentation.test_catalog import DEFAULT_CATALOG


METADATA_KEYS = {
    "format_version", "dataset_name", "dataset_id", "semantic_level",
    "supervision_level", "background_id", "ignore_id", "single_label",
    "annotation_quality",
}


def archive_dataset(dataset_root, relative, archive_root, *, shard_bytes=4 * 1024**3):
    """Package one filtered dataset into bounded tar shards with portable paths."""
    root, relative = Path(dataset_root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Archive dataset paths must be relative")
    if shard_bytes < 1024:
        raise ValueError("Shard size must be at least 1024 bytes")
    folder = Path(archive_root) / relative
    folder.mkdir(parents=True, exist_ok=True)
    results, archive, current, current_bytes = [], None, None, 0

    def finish():
        if archive is None:
            return
        archive.close()
        complete = current.with_suffix("")
        current.replace(complete)
        digest = hashlib.sha256()
        with complete.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        results.append(dict(path=str(complete), bytes=complete.stat().st_size,
                            sha256=digest.hexdigest()))

    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError(f"Export contains a symlink: {path}")
            size = path.stat().st_size
            estimate = 512 + ((size + 511) // 512) * 512
            if archive is not None and current_bytes + estimate > shard_bytes:
                finish(); archive = None
            if archive is None:
                current = folder / f"part-{len(results):05d}.tar.partial"
                archive = tarfile.open(current, "w")
                current_bytes = 0
            archive.add(path, arcname=(relative / path.relative_to(root)).as_posix(), recursive=False)
            current_bytes += estimate
        finish(); archive = None
    finally:
        if archive is not None:
            archive.close()
    return results


def export_dataset(source, destination, splits, *, reader, level, progress=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Export destination must be separate from the source dataset")
    if destination.exists():
        raise FileExistsError(f"Use a new, empty export destination: {destination}")
    pending = destination.with_name(destination.name + ".partial")
    if pending.exists():
        raise FileExistsError(f"Unfinished export exists: {pending}")
    datasets = []
    try:
        for split in splits:
            dataset = (open_text_prompt_dataset(source, split, validate_masks=False, level=level)
                       if reader == "training" else
                       SemanticSegmentationDataset(source, split, validate_masks=False))
            datasets.append(dataset)
        primary = datasets[0]
        base = getattr(primary, "semantic", primary)
        metadata = base.metadata
        storage = getattr(base, "storage", None)
        selected = sorted(set().union(*(set(getattr(d, "semantic", d).sample_ids) for d in datasets)))
        pending.mkdir(parents=True)
        copied, total = set(), 0

        def materialize(path, *, metadata_file=False):
            nonlocal total
            path = Path(path)
            if PureWindowsPath(str(path)).drive and not path.is_absolute():
                raise ValueError(f"Nonportable source path: {path}")
            original = (source / path).resolve() if not path.is_absolute() else path.resolve()
            relative = original.relative_to(source)
            if relative in copied:
                return relative.as_posix()
            data = storage.read(original) if storage and not metadata_file else original.read_bytes()
            target = pending / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            copied.add(relative)
            total += len(data)
            if progress is not None and len(copied) % 1000 == 0:
                progress(files=len(copied), bytes=total)
            return relative.as_posix()

        # Catalog and only the selected split files are public data dependencies.
        classes = materialize(metadata.get("classes_file", "classes.json"), metadata_file=True)
        (pending / "splits").mkdir(exist_ok=True)
        counts = {}
        for split, dataset in zip(splits, datasets):
            ids = getattr(dataset, "semantic", dataset).sample_ids
            counts[split] = len(ids)
            (pending / "splits" / f"{split}.txt").write_text(
                "".join(sample + "\n" for sample in ids), encoding="utf-8")

        cleaned = {key: metadata[key] for key in METADATA_KEYS if key in metadata}
        cleaned.update(classes_file=classes, splits=counts)
        if hasattr(primary, "rows"):
            rows = []
            for sample_id in selected:
                row = copy.deepcopy(primary.rows[sample_id])
                row["image"] = materialize(row.get("image", row.get("image_path")))
                row.pop("image_path", None)
                for annotation in row.get("annotations", row.get("queries", [])):
                    target = annotation.get("target")
                    if isinstance(target, dict) and target.get("path"):
                        target["path"] = materialize(target["path"])
                rows.append(row)
            (pending / "annotations.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            cleaned["annotations_file"] = "annotations.jsonl"
        else:
            cleaned["image_dir"] = metadata.get("image_dir", "images")
            if not base.box_only:
                cleaned["mask_dir"] = metadata.get("mask_dir", "masks")
            # Only object training consumes detection boxes; semantic tests need masks.
            annotations = metadata.get("annotations_dir") if reader == "training" else None
            if annotations:
                cleaned["annotations_dir"] = annotations
            for sample_id in selected:
                materialize(base._images[sample_id])
                if not base.box_only:
                    materialize(base._masks[sample_id])
                if annotations:
                    materialize(Path(annotations) / f"{sample_id}.json")
        (pending / "dataset.json").write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
        entries = [dict(name=cleaned["dataset_name"], source_task=level, root=pending,
                        split=split, reader=reader) for split in splits]
        validate_datasets(entries)
        pending.rename(destination)
        return dict(dataset=cleaned["dataset_name"], splits=counts,
                    unique_samples=len(selected), sample_bytes=total)
    finally:
        for dataset in datasets:
            storage = getattr(getattr(dataset, "semantic", dataset), "storage", None)
            if storage:
                storage.close()


def export_datasets(data_root, output_root, *, mode="all", train_suite=TRAIN_SUITE,
                    test_suite=DEFAULT_CATALOG):
    source, output = Path(data_root).resolve(), Path(output_root).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Export root must be separate from the source data tree")
    if output.exists():
        raise FileExistsError(f"Use a new export root: {output}")
    entries = dataset_entries(source, mode, train_suite=train_suite,
                              test_suite=test_suite, scope="training_source")
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["root"], []).append(entry)
    output.mkdir(parents=True)
    results = []
    for root, rows in grouped.items():
        result = export_dataset(root, output / root.relative_to(source),
            [row["split"] for row in rows], reader=rows[0]["reader"], level=rows[0]["source_task"])
        results.append({"directory": root.relative_to(source).as_posix(), **result})
        print(json.dumps(results[-1]), flush=True)
    report = dict(complete=True, scope="training_source", mode=mode, datasets=results)
    (output / "export-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "train", "test"), default="all")
    parser.add_argument("--train-suite", type=Path, default=TRAIN_SUITE)
    parser.add_argument("--test-suite", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    export_datasets(args.data_root, args.output_root, mode=args.mode,
                    train_suite=args.train_suite, test_suite=args.test_suite)


if __name__ == "__main__":
    main()
