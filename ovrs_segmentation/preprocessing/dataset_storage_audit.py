from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ovrs_segmentation.datasets import open_text_prompt_dataset


SAMPLE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def audit_dataset(root: Path) -> dict[str, Any]:
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    storage = metadata.get("storage", {})
    active_lmdb = root / str(storage.get("path", "data.lmdb"))
    lmdb_directories = sorted(path for path in root.rglob("*.lmdb") if path.is_dir())
    loose_files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SAMPLE_SUFFIXES
        and not any(parent.suffix.lower() == ".lmdb" for parent in path.parents)
    )
    splits = {}
    for split in ("val", "test"):
        path = root / "splits" / f"{split}.txt"
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
        error = None
        if ids:
            try:
                dataset = open_text_prompt_dataset(root, split=split, validate_masks=False)
                if len(dataset) != len(ids):
                    error = f"split has {len(ids)} IDs but dataset resolves {len(dataset)} samples"
            except Exception as exception:
                error = str(exception)
        splits[split] = {"samples": len(ids), "valid": bool(ids) and error is None, "error": error}
    valid = (
        storage.get("backend") == "lmdb"
        and active_lmdb.is_dir()
        and not loose_files
        and all(row["valid"] for row in splits.values())
    )
    return {
        "dataset": root.name,
        "valid": valid,
        "active_lmdb": active_lmdb.name if active_lmdb.is_dir() else None,
        "extra_lmdb": [path.name for path in lmdb_directories if path.resolve() != active_lmdb.resolve()],
        "loose_sample_files": len(loose_files),
        "loose_examples": [path.relative_to(root).as_posix() for path in loose_files[:5]],
        "splits": splits,
    }


def audit_parent_root(parent_root: Path) -> list[dict[str, Any]]:
    return [
        audit_dataset(root)
        for root in sorted(path for path in parent_root.iterdir() if (path / "dataset.json").is_file())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LMDB-only storage and non-empty validation/test splits")
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = audit_parent_root(args.parent_root)
    report = {"valid": all(row["valid"] for row in rows), "datasets": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
