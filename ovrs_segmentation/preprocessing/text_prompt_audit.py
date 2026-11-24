"""Download and audit the class prompts actually used by prepared datasets."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ovrs_segmentation.datasets.class_prompts import (
    canonical_prompt_text,
    class_specs_by_id,
    default_prompt_texts,
    global_alias_prompt_texts,
    is_level_prompt,
    normalize_class_text,
)


CATALOG_FILES = (
    "classes.json",
    "classes.before-text-prompts.json",
    "dataset.json",
)
LEVELS = {"scene", "object", "lulc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/Datasets/RSOD"), help="Local/server RSOD parent directory")
    parser.add_argument("--remote", help="SSH destination, for example user@login.host")
    parser.add_argument("--remote-root", default="/home/Datasets/RSOD", help="RSOD parent directory on the SSH host")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text_prompt_audit"))
    return parser.parse_args()


def _remote_files(remote: str, root: str) -> dict[str, bytes]:
    code = """import base64,json,sys
from pathlib import Path
root=Path(sys.argv[1])
names={'classes.json','classes.before-text-prompts.json','dataset.json'}
files={}
for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
    for name in names:
        path=dataset/name
        if path.is_file():
            files[f'{dataset.name}/{name}']=base64.b64encode(path.read_bytes()).decode('ascii')
print(json.dumps(files))
"""
    command = f"python3 -c {shlex.quote(code)} {shlex.quote(root)}"
    process = subprocess.run(
        ["ssh", remote, command], check=True, capture_output=True, text=True
    )
    return {
        relative: base64.b64decode(content)
        for relative, content in json.loads(process.stdout).items()
    }


def _local_files(root: Path) -> dict[str, bytes]:
    files = {}
    for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
        for name in CATALOG_FILES:
            path = dataset / name
            if path.is_file():
                files[f"{dataset.name}/{name}"] = path.read_bytes()
    return files


def _save_snapshot(files: dict[str, bytes], destination: Path) -> None:
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _default_level(metadata: dict[str, Any]) -> str:
    if metadata.get("supervision_level") == "box":
        return "object"
    level = str(metadata.get("semantic_level", ""))
    if level in LEVELS:
        return level
    return "lulc"


def _unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _audit_dataset(root: Path) -> dict[str, Any]:
    catalog = _load_json(root / "classes.json")
    metadata = _load_json(root / "dataset.json") if (root / "dataset.json").is_file() else {}
    dataset_name = str(metadata.get("dataset_name") or root.name)
    rows = list(class_specs_by_id(root / "classes.json").values())
    scopes = metadata.get("query_scope_by_class") or {}
    default_level = _default_level(metadata)
    box_only = bool(
        metadata.get("annotations_dir")
        and str(metadata.get("supervision_level", "")).lower() == "box"
    )
    audited = []
    for row in rows:
        class_id = int(row.get("class_id", row.get("dataset_class_id")))
        class_text = str(row.get("class_text", row.get("name", ""))).strip()
        normalized = normalize_class_text(class_text)
        canonical = canonical_prompt_text(class_text, dataset_name)
        level = str(scopes.get(str(class_id), scopes.get(class_id, row.get("semantic_level") or default_level)))
        is_background = bool(row.get("is_background")) or class_id == metadata.get("background_id")
        is_ignore = bool(row.get("is_ignore")) or class_id == int(metadata.get("ignore_id", 255))
        is_excluded = bool(row.get("is_excluded"))
        queryable = bool(row.get("queryable", not (is_background or is_ignore or is_excluded)))
        if box_only and not (is_background or is_ignore):
            level, queryable = "object", True
        candidates = []
        if queryable and level in LEVELS:
            candidates.extend(default_prompt_texts(class_text, level, dataset_name))
            candidates.extend(global_alias_prompt_texts(class_text, dataset_name))
        candidates = _unique(candidates)
        inference_all = [
            prompt for prompt in candidates
            if not is_level_prompt(prompt, class_text, dataset_name)
        ]
        inference_queryable = queryable
        audited.append({
            "class_id": class_id,
            "class_text": class_text,
            "normalized_class_text": normalized,
            "canonical_prompt": canonical,
            "semantic_level": level if queryable else None,
            "queryable": queryable,
            "inference_queryable": inference_queryable,
            "training_inference_mismatch": queryable != inference_queryable,
            "aliases": row.get("aliases") or [],
            "training_prompt_candidates": candidates,
            "inference_class_text": [canonical] if queryable else [],
            "inference_all_candidates": inference_all,
            "noun_phrase_rewritten": canonical != normalized,
            "text_augmented": len(candidates) > 1 or canonical != normalized,
        })
    active = [row for row in audited if row["queryable"]]
    return {
        "dataset": str(metadata.get("dataset_name") or catalog.get("dataset") or root.name),
        "directory": root.name,
        "catalog_format": catalog["format_version"],
        "has_original_class_backup": (root / "classes.before-text-prompts.json").is_file(),
        "level_counts": dict(Counter(row["semantic_level"] for row in active)),
        "queryable_classes": len(active),
        "augmented_classes": sum(row["text_augmented"] for row in active),
        "classes": audited,
    }


def _write_reports(snapshot: Path, output: Path) -> list[dict[str, Any]]:
    reports = [
        _audit_dataset(root)
        for root in sorted(snapshot.iterdir())
        if root.is_dir() and (root / "classes.json").is_file()
    ]
    (output / "text_prompt_audit.json").write_text(
        json.dumps({"datasets": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "text_prompt_classes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = (
            "dataset", "directory", "class_id", "class_text", "semantic_level",
            "queryable", "inference_queryable", "training_inference_mismatch",
            "canonical_prompt", "training_prompt_candidates",
            "inference_all_candidates", "aliases", "noun_phrase_rewritten", "text_augmented",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            for row in report["classes"]:
                writer.writerow({
                    **{key: report[key] for key in ("dataset", "directory")},
                    **{key: row[key] for key in columns if key in row},
                    "training_prompt_candidates": " | ".join(row["training_prompt_candidates"]),
                    "inference_all_candidates": " | ".join(row["inference_all_candidates"]),
                    "aliases": " | ".join(map(str, row["aliases"])),
                })
    lines = [
        "# Dataset text-prompt audit",
        "",
        "| Dataset | Catalog | Original backup | Scene | Object | LULC | Queryable | Augmented |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        counts = report["level_counts"]
        lines.append(
            f"| {report['directory']} | {report['catalog_format']} | "
            f"{'yes' if report['has_original_class_backup'] else 'no'} | "
            f"{counts.get('scene', 0)} | {counts.get('object', 0)} | "
            f"{counts.get('lulc', 0)} | {report['queryable_classes']} | "
            f"{report['augmented_classes']} |"
        )
    (output / "text_prompt_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reports


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    snapshot = output / "catalogs"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True)
    files = _remote_files(args.remote, args.remote_root) if args.remote else _local_files(args.source_root.expanduser().resolve())
    if not files:
        raise FileNotFoundError("No dataset classes.json files were found")
    _save_snapshot(files, snapshot)
    reports = _write_reports(snapshot, output)
    print(json.dumps({"datasets": len(reports), "downloaded_files": len(files), "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
