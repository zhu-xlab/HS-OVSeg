#!/usr/bin/env python3
"""Download remote-sensing scene-classification datasets without creating masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASETS = {
    "resisc45": {
        "root": "NWPU-RESISC45",
        "repo": "torchgeo/resisc45",
        "revision": "a826b44d938a883185f11ebe3d512d38b464312f",
        "archive": "NWPU-RESISC45.zip",
        "md5": "75206b2e16446591afa88e2628744886",
        "extra": ["resisc45-train.txt", "resisc45-val.txt", "resisc45-test.txt"],
    },
    "patternnet": {
        "root": "PatternNet",
        "repo": "torchgeo/PatternNet",
        "revision": "2dbd901b00e301967a5c5146b25454f5d3455ad0",
        "archive": "PatternNet.zip",
        "md5": "96d54b3224c5350a98d55d5a7e6984ad",
        "extra": [],
    },
    "aid": {
        "root": "AID",
        "repo": "blanchon/AID",
        "revision": "main",
        "snapshot_patterns": ["data/**", "README.md"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a scene classification dataset")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - published archive integrity checksum
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        for member in members:
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
        with tqdm(total=sum(x.file_size for x in members), desc=f"Extracting {archive.name}", unit="B", unit_scale=True) as progress:
            for member in members:
                stream.extract(member, root)
                progress.update(member.file_size)


def download(dataset: str, output_dir: Path, overwrite: bool) -> dict:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise ImportError("Install project dependencies with: python -m pip install -e .") from error
    config = DATASETS[dataset]
    output = output_dir.expanduser().resolve()
    raw, marker = output / "raw", output / "raw" / ".download-complete"
    if marker.is_file() and not overwrite:
        return {"dataset": dataset, "output_dir": str(output), "skipped": True}
    if raw.exists() and overwrite:
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"scene-{dataset}-") as temporary:
        if "archive" in config:
            archive = Path(hf_hub_download(
                repo_id=config["repo"], filename=config["archive"], repo_type="dataset",
                revision=config["revision"], cache_dir=temporary,
            ))
            actual = _md5(archive)
            if actual != config["md5"]:
                raise ValueError(f"MD5 mismatch for {config['archive']}: {actual}")
            _extract_zip(archive, raw)
            for filename in config["extra"]:
                source = Path(hf_hub_download(
                    repo_id=config["repo"], filename=filename, repo_type="dataset",
                    revision=config["revision"], cache_dir=temporary,
                ))
                shutil.copy2(source, raw / filename)
        else:
            snapshot = Path(snapshot_download(
                repo_id=config["repo"], repo_type="dataset", revision=config["revision"],
                allow_patterns=config["snapshot_patterns"], cache_dir=temporary,
            ))
            for source in snapshot.rglob("*"):
                if source.is_file():
                    relative = source.relative_to(snapshot)
                    destination = raw / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    image_count = sum(1 for path in raw.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    if not image_count:
        raise RuntimeError(f"No images were downloaded for {dataset}")
    marker_payload = {"repo": config["repo"], "images": image_count}
    marker.write_text(json.dumps(marker_payload) + "\n", encoding="utf-8")
    summary = {
        "dataset": dataset,
        "output_dir": str(output),
        "image_count": image_count,
        "source_repo": config["repo"],
        "source_scope": "official_distribution",
        "skipped": False,
    }
    (output / "download_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config = DATASETS[args.dataset]
    output = args.output_dir or Path("/home/Datasets/RSOD") / config["root"]
    print(json.dumps(download(args.dataset, output, args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
