from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

from ovrs_segmentation.datasets.lmdb_store import FORMAT_VERSION, INDEX_KEY, META_KEY


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _referenced_paths(root: Path, metadata: dict) -> list[Path]:
    paths: set[Path] = set()
    directories = [metadata.get("image_dir", "images"), metadata.get("mask_dir", "masks"), metadata.get("annotations_dir")]
    directories.extend(config.get("directory") for config in metadata.get("optional_modalities", {}).values())
    for value in directories:
        directory = root / str(value) if value else None
        if directory and directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())

    annotations_file = metadata.get("annotations_file")
    if annotations_file:
        annotation_path = root / str(annotations_file)
        paths.add(annotation_path)
        for row in _jsonl(annotation_path):
            image = row.get("image", row.get("image_path"))
            if image:
                paths.add(root / str(image))
            for annotation in row.get("annotations", row.get("queries", [])):
                target = annotation.get("target") or {}
                if target.get("path"):
                    paths.add(root / str(target["path"]))

    return sorted(path.resolve() for path in paths if path.is_file())


def _write_metadata(root: Path, metadata: dict, lmdb_name: str) -> None:
    metadata = dict(metadata)
    metadata["storage"] = {"backend": "lmdb", "path": lmdb_name, "format_version": FORMAT_VERSION}
    destination = root / "dataset.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def pack_dataset(root: str | Path, *, output_name: str = "data.lmdb", overwrite: bool = False, remove_source_files: bool = False) -> dict:
    import lmdb

    root = Path(root).expanduser().resolve()
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    destination = root / output_name
    if metadata.get("storage", {}).get("backend") == "lmdb" and destination.is_dir() and not overwrite:
        return {"dataset": metadata["dataset_name"], "lmdb": str(destination), "status": "already_packed"}
    files = _referenced_paths(root, metadata)
    if not files:
        raise FileNotFoundError(f"No sample files found below {root}")

    temporary = root / f".{output_name}.tmp"
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    if temporary.exists():
        shutil.rmtree(temporary)

    total_bytes = sum(path.stat().st_size for path in files)
    environment = lmdb.open(str(temporary), subdir=True, map_size=int(total_bytes * 1.15) + 1_073_741_824, lock=True, meminit=False)
    relative_paths = [path.relative_to(root).as_posix() for path in files]
    transaction = environment.begin(write=True)
    for index, (path, relative) in enumerate(tqdm(zip(files, relative_paths), total=len(files), desc=root.name, unit="file"), 1):
        transaction.put(f"file:{relative}".encode("utf-8"), path.read_bytes())
        if index % 512 == 0:
            transaction.commit()
            transaction = environment.begin(write=True)
    transaction.put(INDEX_KEY, json.dumps(relative_paths, separators=(",", ":")).encode("utf-8"))
    transaction.put(META_KEY, json.dumps({"format_version": FORMAT_VERSION, "dataset_name": metadata["dataset_name"], "file_count": len(files), "source_bytes": total_bytes}, separators=(",", ":")).encode("utf-8"))
    transaction.commit()
    environment.sync()
    environment.close()
    os.replace(temporary, destination)
    _write_metadata(root, metadata, output_name)

    if remove_source_files:
        for path in tqdm(files, desc="Removing source files", unit="file"):
            path.unlink()
        for directory in sorted({path.parent for path in files}, key=lambda path: len(path.parts), reverse=True):
            if directory != root:
                try:
                    directory.rmdir()
                except OSError:
                    pass
    return {"dataset": metadata["dataset_name"], "lmdb": str(destination), "files": len(files), "source_bytes": total_bytes, "source_files_removed": remove_source_files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack prepared OVRS datasets into LMDB.")
    parser.add_argument("--dataset-root", type=Path, nargs="*", default=[])
    parser.add_argument("--parent-root", type=Path, nargs="*", default=[])
    parser.add_argument("--output-name", default="data.lmdb")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remove-source-files", action="store_true")
    return parser.parse_args()


def discover_dataset_roots(dataset_roots: list[Path], parent_roots: list[Path]) -> list[Path]:
    roots = list(dataset_roots)
    for parent in parent_roots:
        roots.extend(path for path in sorted(parent.expanduser().iterdir()) if path.is_dir() and (path / "dataset.json").is_file())
    return list(dict.fromkeys(path.expanduser().resolve() for path in roots))


def main() -> None:
    args = parse_args()
    roots = discover_dataset_roots(args.dataset_root, args.parent_root)
    if not roots:
        raise ValueError("Provide --dataset-root or a --parent-root containing prepared datasets")
    failures = []
    for root in roots:
        try:
            result = pack_dataset(root, output_name=args.output_name, overwrite=args.overwrite, remove_source_files=args.remove_source_files)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as error:
            failures.append(root)
            print(json.dumps({"dataset_root": str(root), "status": "failed", "error": str(error)}, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"LMDB packing failed for {len(failures)} dataset(s)")


if __name__ == "__main__":
    main()
