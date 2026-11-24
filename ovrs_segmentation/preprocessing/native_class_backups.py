from __future__ import annotations

import argparse
import json
from pathlib import Path

from ovrs_segmentation.datasets.class_prompts import class_specs_by_id


BACKUP_NAME = "classes.before-text-prompts.json"


def export_native_class_backups(parent_root: Path, output_root: Path) -> list[dict[str, object]]:
    manifest = []
    for dataset_root in sorted(path for path in parent_root.iterdir() if (path / "classes.json").is_file()):
        classes = {
            str(class_id): spec["class_text"]
            for class_id, spec in class_specs_by_id(dataset_root / "classes.json").items()
        }
        destination = output_root / dataset_root.name / BACKUP_NAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"dataset": dataset_root.name, "classes": len(classes), "path": str(destination)})
    if output_root.resolve() != parent_root.resolve():
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export native class-name backups for prepared datasets")
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_native_class_backups(args.parent_root, args.output_root)
    print(f"datasets={len(manifest)} classes={sum(int(row['classes']) for row in manifest)} output={args.output_root}")


if __name__ == "__main__":
    main()
