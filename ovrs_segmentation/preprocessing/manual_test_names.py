"""Name reviewed test datasets with their GT sample count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MANUAL_TEST_COUNTS = {"AID": 90, "NWPU-RESISC45": 135, "PatternNet": 114,
                      "DIOR-R": 200, "DOTA-v2": 160, "FAIR1M-2.0": 201, "xView": 100}
RENAMES = {name: f"{name}-GT{count}" for name, count in MANUAL_TEST_COUNTS.items()}


def normalize_manual_test_names(parent: Path):
    parent = parent.resolve(strict=True)
    planned = [(parent / old, parent / new) for old, new in RENAMES.items() if (parent / old).exists()]
    for source, target in planned:
        if source.is_symlink() or source.resolve().parent != parent:
            raise ValueError(f"Dataset must stay inside the named test directory: {source}")
        if target.exists():
            raise FileExistsError(f"Both old and canonical dataset names exist: {source}, {target}")
        metadata = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
        if metadata["dataset_name"] != source.name:
            raise ValueError(f"Unexpected dataset identity: {source}")
        ids = (source / "splits/test.txt").read_text(encoding="utf-8").splitlines()
        ids = [value.strip() for value in ids if value.strip()]
        count = MANUAL_TEST_COUNTS[source.name]
        if metadata.get("splits", {}).get("test") != count or len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"GT count does not match the proposed directory name: {source}")
    renamed = []
    for source, target in planned:
        source.rename(target)
        path = target / "dataset.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["dataset_id"] = target.name
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
        renamed.append({"previous": source.name, "directory": target.name, "splits": metadata.get("splits")})
    return renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=Path("data/segmentation_test"))
    args = parser.parse_args()
    print(json.dumps(normalize_manual_test_names(args.test_root)))


if __name__ == "__main__":
    main()
