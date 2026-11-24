#!/usr/bin/env python3
"""Download the complete annotated DOTA-v2.0 train/val release from a mirror."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from tqdm import tqdm


DEFAULT_REPO = "isaaccorley/dota"
ARCHIVES = (
    # DOTA-v2.0 extends rather than replaces the v1.0 imagery.  The official
    # v2 annotation archives cover both old and newly added images.
    "dotav1.0_images_train.tar.gz",
    "dotav1.0_images_val.tar.gz",
    "dotav2.0_images_train.tar.gz",
    "dotav2.0_annotations_train.tar.gz",
    "dotav2.0_images_val.tar.gz",
    "dotav2.0_annotations_val.tar.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download complete DOTA-v2.0 train/val images and oriented-box annotations"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/Datasets/RSOD/DOTA-v2"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _archive_stem(filename: str) -> str:
    return filename.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, mode="r:*") as stream:
        members = stream.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in archive {archive}: {member.name}")
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in archive {archive}: {member.name}")
        stream.extractall(root, members=members)


def download(output_dir: Path, repo_id: str, revision: str, overwrite: bool) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("Install project dependencies with: python -m pip install -e .") from error
    output_dir = output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = skipped = 0
    for filename in tqdm(ARCHIVES, desc="DOTA-v2 archives", unit="archive"):
        destination = raw_dir / _archive_stem(filename)
        marker = destination / ".extraction-complete"
        if marker.is_file() and not overwrite:
            skipped += 1
            continue
        if destination.exists() and overwrite:
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        # Cache is temporary and never leaves .cache/huggingface below the dataset root.
        with tempfile.TemporaryDirectory(prefix="dota-v2-") as temporary:
            archive = Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=temporary,
            ))
            _safe_extract(archive, destination)
        marker.write_text(filename + "\n", encoding="utf-8")
        extracted += 1
    manifest = {
        "dataset": "DOTA-v2.0",
        "repo_id": repo_id,
        "revision": revision,
        "output_dir": str(output_dir),
        "archives": list(ARCHIVES),
        "extracted": extracted,
        "skipped": skipped,
        "license_notice": "Academic use only; Google Earth imagery remains subject to its terms.",
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(
        download(args.output_dir, args.repo_id, args.revision, args.overwrite),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
