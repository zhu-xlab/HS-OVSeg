from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from ovrs_segmentation.datasets.class_prompts import (
    PROMPT_VOCABULARY_PATH,
    class_specs_by_id,
    normalize_class_text,
)

def build_vocabulary(parent_root: Path, output: Path) -> dict:
    current = yaml.safe_load(output.read_text(encoding="utf-8"))
    previous = current.get("datasets", {})
    datasets = {}
    roots = sorted(path for path in parent_root.iterdir() if (path / "classes.json").is_file())
    for root in roots:
        metadata_path = root / "dataset.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        dataset_name = str(metadata.get("dataset_name") or root.name)
        previous_rows = previous.get(dataset_name, {})
        rows = {}
        for spec in class_specs_by_id(root / "classes.json").values():
            native = normalize_class_text(spec["class_text"])
            saved = previous_rows.get(native)
            if saved is None:
                canonical, aliases = native, []
            else:
                canonical = normalize_class_text(saved["canonical_name"])
                aliases = [normalize_class_text(value) for value in saved.get("aliases", [])]
            rows[native] = {
                "canonical_name": canonical,
                "aliases": list(dict.fromkeys(aliases)),
            }
        datasets[dataset_name] = dict(sorted(rows.items()))
    document = {"version": "ovrs-text-vocabulary-v4", "datasets": datasets}
    output.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset-native text prompt mappings")
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROMPT_VOCABULARY_PATH)
    args = parser.parse_args()
    vocabulary = build_vocabulary(args.parent_root, args.output)
    classes = sum(len(rows) for rows in vocabulary["datasets"].values())
    print(f"datasets={len(vocabulary['datasets'])} native_classes={classes} output={args.output}")


if __name__ == "__main__":
    main()
