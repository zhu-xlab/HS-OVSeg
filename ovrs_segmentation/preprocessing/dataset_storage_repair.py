from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from ovrs_segmentation.datasets.lmdb_store import INDEX_KEY, META_KEY
from ovrs_segmentation.preprocessing.dataset_storage_audit import SAMPLE_SUFFIXES, audit_parent_root


TILE_SUFFIX = re.compile(r"__x\d+_y\d+$")


def _active_lmdb(root: Path) -> Path:
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    return root / str(metadata.get("storage", {}).get("path", "data.lmdb"))


def archive_loose_samples(root: Path) -> int:
    import lmdb

    lmdb_path = _active_lmdb(root)
    loose = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SAMPLE_SUFFIXES
        and not any(parent.suffix.lower() == ".lmdb" for parent in path.parents)
    )
    if not loose:
        return 0

    relative_paths = [path.relative_to(root).as_posix() for path in loose]
    added_bytes = sum(path.stat().st_size for path in loose)
    current_bytes = (lmdb_path / "data.mdb").stat().st_size
    environment = lmdb.open(str(lmdb_path), subdir=True, map_size=current_bytes + added_bytes * 2 + 1_073_741_824, lock=True, meminit=False)
    with environment.begin(write=True) as transaction:
        index = json.loads(transaction.get(INDEX_KEY).decode("utf-8"))
        indexed = set(index)
        for path, relative in zip(loose, relative_paths):
            key = f"file:{relative}".encode("utf-8")
            if transaction.get(key) is None:
                transaction.put(key, path.read_bytes())
            if relative not in indexed:
                index.append(relative)
                indexed.add(relative)
        metadata = json.loads(transaction.get(META_KEY).decode("utf-8"))
        metadata["file_count"] = len(index)
        metadata["source_bytes"] = int(metadata.get("source_bytes", 0)) + added_bytes
        transaction.put(INDEX_KEY, json.dumps(sorted(index), separators=(",", ":")).encode("utf-8"))
        transaction.put(META_KEY, json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
    environment.sync()

    with environment.begin() as transaction:
        missing = [relative for relative in relative_paths if transaction.get(f"file:{relative}".encode("utf-8")) is None]
    environment.close()
    if missing:
        raise RuntimeError(f"LMDB verification failed for {len(missing)} files below {root}")

    for path in loose:
        path.unlink()
    for directory in sorted({path.parent for path in loose}, key=lambda path: len(path.parts), reverse=True):
        if directory != root:
            try:
                directory.rmdir()
            except OSError:
                pass
    return len(loose)


def _sample_group(sample_id: str) -> str:
    return TILE_SUFFIX.sub("", sample_id)


def split_val_for_empty_test(root: Path, seed: int = 42) -> tuple[int, int] | None:
    split_dir = root / "splits"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    test_ids = test_path.read_text(encoding="utf-8").splitlines() if test_path.is_file() else []
    if any(value.strip() for value in test_ids):
        return None

    val_ids = [value.strip() for value in val_path.read_text(encoding="utf-8").splitlines() if value.strip()]
    groups: dict[str, list[str]] = {}
    for sample_id in val_ids:
        groups.setdefault(_sample_group(sample_id), []).append(sample_id)
    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    test_groups = set(group_names[: len(group_names) // 2])
    new_val = sorted(sample_id for group, ids in groups.items() if group not in test_groups for sample_id in ids)
    new_test = sorted(sample_id for group, ids in groups.items() if group in test_groups for sample_id in ids)
    if not new_val or not new_test:
        raise ValueError(f"Cannot split {len(val_ids)} validation samples below {root}")
    val_path.write_text("".join(f"{sample_id}\n" for sample_id in new_val), encoding="utf-8")
    test_path.write_text("".join(f"{sample_id}\n" for sample_id in new_test), encoding="utf-8")
    return len(new_val), len(new_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive loose samples and fill empty test splits")
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    roots = sorted(path for path in args.parent_root.iterdir() if (path / "dataset.json").is_file())
    for root in roots:
        archived = archive_loose_samples(root)
        split = split_val_for_empty_test(root, args.seed)
        if archived or split:
            print(json.dumps({"dataset": root.name, "archived_files": archived, "val_test": split}))

    rows = audit_parent_root(args.parent_root)
    invalid = [row["dataset"] for row in rows if not row["valid"]]
    if invalid:
        raise RuntimeError(f"Storage audit still fails: {', '.join(invalid)}")
    print(json.dumps({"valid": True, "datasets": len(rows)}))


if __name__ == "__main__":
    main()
