"""Download prepared HS-OVSeg datasets and check them with the real readers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tarfile
import zipfile

import yaml

from ovrs_segmentation.datasets import open_text_prompt_dataset
from ovrs_segmentation.datasets.semantic_dataset import SemanticSegmentationDataset
from ovrs_segmentation.test_catalog import DEFAULT_CATALOG, SCOPES, load_suite


TRAIN_SUITE = Path(__file__).resolve().parents[2] / "configs/inference_val.yaml"
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")


def dataset_entries(data_root, mode, *, train_suite=TRAIN_SUITE,
                    test_suite=DEFAULT_CATALOG, scope="training_source"):
    root = Path(data_root).expanduser().resolve()
    if mode not in ("all", "train", "test"):
        raise ValueError(f"Unknown download mode: {mode}")
    entries = []
    if mode in ("all", "train"):
        rows = yaml.safe_load(Path(train_suite).read_text(encoding="utf-8"))["datasets"]
        for row in rows:
            path = Path(row["root"].format(data_root=root)).resolve()
            for split in ("train", "val"):
                entries.append({**row, "root": path, "split": split, "reader": "training"})
    if mode in ("all", "test"):
        entries.extend({**row, "reader": "semantic"} for row in
                       load_suite(test_suite, root, scope=scope))
    if not entries:
        raise ValueError("The selected dataset suites contain no datasets")
    for entry in entries:
        entry["root"] = Path(entry["root"]).resolve()
        entry["root"].relative_to(root)
    return entries


def validate_datasets(entries):
    """Resolve every split and decode its first sample without scanning all masks."""
    results, errors = [], []
    for entry in entries:
        dataset = None
        try:
            root = entry["root"]
            metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
            paths = [metadata.get(k) for k in
                     ("image_dir", "mask_dir", "classes_file", "annotations_dir",
                      "annotations_file", "labels_file")]
            paths.append((metadata.get("storage") or {}).get("path"))
            for path in paths:
                if path and (PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
                             or ".." in PurePosixPath(path.replace("\\", "/")).parts):
                    raise ValueError(f"Metadata paths must be dataset-relative: {path}")
            reader = (open_text_prompt_dataset if entry["reader"] == "training"
                      else SemanticSegmentationDataset)
            kwargs = {"validate_masks": False}
            if entry["reader"] == "training":
                kwargs["level"] = entry["source_task"]
            dataset = reader(root, split=entry["split"], **kwargs)
            if not len(dataset):
                raise ValueError("Split is empty")
            dataset[0]
            results.append({"dataset": entry["name"], "split": entry["split"],
                            "samples": len(dataset)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{entry['name']} ({entry['split']}): {error}")
        finally:
            storage = getattr(getattr(dataset, "semantic", dataset), "storage", None)
            if storage is not None:
                storage.close()
    if errors:
        raise ValueError("Downloaded data is not ready:\n  " + "\n  ".join(errors))
    return results


def _archive_target(root, name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    target = (root / path).resolve()
    target.relative_to(root)
    return target


def extract_archive(archive, data_root):
    """Extract a prepared-data archive, rejecting links and escaping paths."""
    root = Path(data_root).resolve()
    archive = Path(archive)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as stream:
            members = stream.infolist()
            targets = [_archive_target(root, member.filename) for member in members]
            if any((member.external_attr >> 16) & 0o170000 == 0o120000 for member in members):
                raise ValueError("Archive links are not supported")
            for member, target in zip(members, targets):
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            targets = [_archive_target(root, member.name) for member in members]
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise ValueError("Only regular files and directories are supported in archives")
            for member, target in zip(members, targets):
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)


def download_datasets(repo_id, data_root, *, revision="main", mode="all",
                      train_suite=TRAIN_SUITE, test_suite=DEFAULT_CATALOG,
                      scope="training_source", extract=False, validate=True, workers=8):
    from huggingface_hub import snapshot_download

    root = Path(data_root).expanduser().resolve()
    entries = dataset_entries(root, mode, train_suite=train_suite,
                              test_suite=test_suite, scope=scope)
    directories = sorted({entry["root"].relative_to(root).as_posix() for entry in entries})
    patterns = [f"{directory}/**" for directory in directories]
    archives = [f"{directory}{suffix}" for directory in directories for suffix in ARCHIVE_SUFFIXES]
    if extract:
        patterns += archives + [f"archives/{directory}/*{suffix}"
            for directory in directories for suffix in ARCHIVE_SUFFIXES]
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      local_dir=str(root), allow_patterns=patterns, max_workers=workers)
    if extract:
        archives += [path.relative_to(root).as_posix() for directory in directories
                     for path in sorted((root / "archives" / directory).glob("*"))
                     if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES)]
        for name in archives:
            archive = root / name
            if archive.is_file():
                extract_archive(archive, root)
    report = {"repo_id": repo_id, "revision": revision, "data_root": str(root),
              "mode": mode, "validated": validate,
              "splits": validate_datasets(entries) if validate else []}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("OVRS_HF_DATASET_REPO"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or commit")
    parser.add_argument("--mode", choices=("all", "train", "test"), default="all")
    parser.add_argument("--train-suite", type=Path, default=TRAIN_SUITE)
    parser.add_argument("--test-suite", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--scope", choices=SCOPES, default="training_source")
    parser.add_argument("--extract-archives", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("Set --repo-id OWNER/DATASET or OVRS_HF_DATASET_REPO")
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = download_datasets(args.repo_id, args.data_root, revision=args.revision,
        mode=args.mode, train_suite=args.train_suite, test_suite=args.test_suite,
        scope=args.scope, extract=args.extract_archives,
        validate=not args.skip_validation, workers=args.workers)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
