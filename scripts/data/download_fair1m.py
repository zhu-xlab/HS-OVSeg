#!/usr/bin/env python3
"""Download and safely extract the annotated FAIR1M-2.0 train/validation sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

from tqdm import tqdm


# FAIR1M-2.0 reuses the FAIR1M-1.0 training set and adds an annotated
# validation set. Test images have no public labels and are intentionally not
# downloaded because they cannot contribute semantic masks.
ARCHIVES = (
    ("train/part1", "images.zip", "1LWT_ybL-s88Lzg9A9wHpj0h2rJHrqrVf", "a460fe6b1b5b276bf856ce9ac72d6568"),
    ("train/part1", "labelXml.zip", "1CnOuS8oX6T9JMqQnfFsbmf7U38G6Vc8u", "80f833ff355f91445c92a0c0c1fa7414"),
    ("train/part2", "images.zip", "1cx4MRfpmh68SnGAYetNlDy68w0NgKucJ", "ad237e61dba304fcef23cd14aa6c4280"),
    ("train/part2", "labelXmls.zip", "1RFVjadTHA_bsB7BJwSZoQbiyM7KIDEUI", "5c5948e68cd0f991a0d73f10956a3b05"),
    ("validation", "images.zip", "1lSSHOD02B6_sUmr2b-R1iqhgWRQRw-S9", "dce782be65405aa381821b5f4d9eac94"),
    ("validation", "labelXmls.zip", "1sTTna1C5n3Senpfo-73PdiNilnja1AV4", "700b516a21edc9eae66ca315b72a09a1"),
)

# The original Google Drive permissions currently allow only the first archive
# for many anonymous users. These Hugging Face repositories mirror the same
# FAIR1M-1.0 training set and FAIR1M-2.0 validation set. Part 2 images are split
# into two ZIPs in the mirror but extract into the same official directory.
HF_MIRRORS = {
    "train/part1/images.zip": ("LittleCollections/FAIR1M1", ("images.zip",)),
    "train/part1/labelXml.zip": ("LittleCollections/FAIR1M1", ("labelXml.zip",)),
    "train/part2/images.zip": (
        "LittleCollections/FAIR1M1",
        ("images-1.zip", "images-2.zip"),
    ),
    "train/part2/labelXmls.zip": (
        "LittleCollections/FAIR1M1",
        ("labelXmls.zip",),
    ),
    "validation/images.zip": ("LittleCollections/FAIR1M2", ("images.zip",)),
    "validation/labelXmls.zip": (
        "LittleCollections/FAIR1M2",
        ("labelXmls.zip",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download annotated FAIR1M-2.0 train/validation")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/Datasets/RSOD/FAIR1M-2.0"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--source",
        choices=("auto", "google-drive", "huggingface"),
        default="auto",
        help="auto tries the original Drive archive once, then uses the HF mirror",
    )
    return parser.parse_args()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - official archive integrity checksum
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        for member in members:
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
        total = sum(member.file_size for member in members if not member.is_dir())
        with tqdm(total=total, desc=f"Extracting {archive.name}", unit="B", unit_scale=True) as progress:
            for member in members:
                stream.extract(member, root)
                if not member.is_dir():
                    progress.update(member.file_size)


def _indexed(root: Path, suffix: str) -> set[str]:
    return {path.stem for path in root.rglob(f"*{suffix}") if path.is_file()}


def _pair_report(raw_dir: Path) -> dict:
    report = {}
    for split, root in (("train", raw_dir / "train"), ("val", raw_dir / "validation")):
        images = _indexed(root, ".tif")
        labels = _indexed(root, ".xml")
        report[split] = {
            "images": len(images),
            "annotations": len(labels),
            "images_without_annotations": sorted(images - labels)[:20],
            "annotations_without_images": sorted(labels - images)[:20],
        }
    return report


def _download_hf_mirror(
    key: str, download_dir: Path, overwrite: bool
) -> tuple[list[Path], Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "huggingface_hub and hf_xet are required; run: python -m pip install -e ."
        ) from error
    repo_id, filenames = HF_MIRRORS[key]
    local_dir = download_dir / ("hf_" + key.replace("/", "__").replace(".zip", ""))
    local_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in filenames:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                force_download=overwrite,
            )
        )
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Hugging Face mirror file is not a ZIP archive: {path}")
        paths.append(path)
    return paths, local_dir


def download(output_dir: Path, overwrite: bool, source: str = "auto") -> dict:
    gdown = None
    if source in {"auto", "google-drive"}:
        try:
            import gdown as gdown_module
            gdown = gdown_module
        except ImportError as error:
            if source == "google-drive":
                raise ImportError(
                    "gdown is required only for --source google-drive; "
                    "install gdown or use --source huggingface"
                ) from error
            tqdm.write(
                "gdown is not installed; --source auto will use the Hugging Face mirror"
            )
    output = output_dir.expanduser().resolve()
    raw_dir, download_dir = output / "raw", output / "downloads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = extracted = skipped = 0
    failures = []
    for relative_root, filename, drive_id, expected_md5 in ARCHIVES:
        key = f"{relative_root}/{filename}"
        destination = raw_dir / relative_root
        marker = destination / f".{filename}.extraction-complete"
        archive = download_dir / f"{relative_root.replace('/', '_')}__{filename}"
        if marker.is_file() and not overwrite:
            skipped += 1
            continue
        if overwrite:
            marker.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
        last_error = None
        source_archives: list[Path] = []
        cleanup_dir: Path | None = None
        used_source = None
        if source in {"auto", "google-drive"} and gdown is not None:
            drive_attempts = 1 if source == "auto" else 4
            for attempt in range(drive_attempts):
                try:
                    if not archive.is_file():
                        result = gdown.download(
                            id=drive_id,
                            output=str(archive),
                            quiet=False,
                            use_cookies=attempt % 2 == 0,
                            resume=not overwrite,
                        )
                        if result is None:
                            raise RuntimeError("gdown returned no completed file")
                    if not zipfile.is_zipfile(archive):
                        raise ValueError("downloaded content is not a ZIP archive")
                    actual_md5 = _md5(archive)
                    if actual_md5 != expected_md5:
                        raise ValueError(
                            f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
                        )
                    source_archives = [archive]
                    used_source = "google-drive"
                    last_error = None
                    break
                except Exception as error:  # gdown uses version-specific exception classes
                    last_error = f"Google Drive: {type(error).__name__}: {error}"
                    if archive.is_file() and not zipfile.is_zipfile(archive):
                        archive.unlink()
                    if attempt + 1 < drive_attempts:
                        delay = (2, 5, 10)[attempt]
                        tqdm.write(f"Retry {attempt + 2}/{drive_attempts} in {delay}s: {key}")
                        time.sleep(delay)
        if not source_archives and source in {"auto", "huggingface"}:
            if source == "auto":
                tqdm.write(f"Google Drive unavailable; using Hugging Face mirror: {key}")
            for attempt in range(4):
                try:
                    source_archives, cleanup_dir = _download_hf_mirror(
                        key, download_dir, overwrite and attempt == 0
                    )
                    # Single-file mirror archives are byte-identical to the
                    # originals. The part2 image mirror is intentionally split
                    # and is integrity-checked by the Hub/Xet content hashes.
                    if len(source_archives) == 1:
                        actual_md5 = _md5(source_archives[0])
                        if actual_md5 != expected_md5:
                            raise ValueError(
                                f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
                            )
                    used_source = "huggingface"
                    last_error = None
                    break
                except Exception as error:
                    source_archives = []
                    last_error = f"Hugging Face: {type(error).__name__}: {error}"
                    if attempt < 3:
                        delay = (2, 5, 10)[attempt]
                        tqdm.write(f"Retry HF {attempt + 2}/4 in {delay}s: {key}")
                        time.sleep(delay)
        if source_archives:
            try:
                destination.mkdir(parents=True, exist_ok=True)
                for source_archive in source_archives:
                    _safe_extract_zip(source_archive, destination)
                marker.write_text(
                    f"source={used_source}\ndrive_id={drive_id}\nmd5={expected_md5}\n",
                    encoding="utf-8",
                )
                for source_archive in source_archives:
                    source_archive.unlink(missing_ok=True)
                if cleanup_dir is not None and cleanup_dir.is_dir():
                    shutil.rmtree(cleanup_dir)
                archive.unlink(missing_ok=True)
                downloaded += 1
                extracted += 1
                last_error = None
            except Exception as error:
                last_error = f"Extraction: {type(error).__name__}: {error}"
        if last_error is not None:
            failures.append({
                "path": f"{relative_root}/{filename}",
                "drive_id": drive_id,
                "browser_url": f"https://drive.google.com/uc?id={drive_id}",
                "error": last_error,
            })
            tqdm.write(f"WARNING: FAIR1M archive unavailable: {relative_root}/{filename}")

    pairs = _pair_report(raw_dir)
    pair_complete = all(
        values["images"] > 0
        and values["annotations"] > 0
        and not values["images_without_annotations"]
        and not values["annotations_without_images"]
        for values in pairs.values()
    )
    incomplete = output / "download_incomplete.json"
    complete = not failures and pair_complete and extracted + skipped == len(ARCHIVES)
    if complete:
        incomplete.unlink(missing_ok=True)
    else:
        incomplete.write_text(
            json.dumps({"failed_archives": failures, "pair_report": pairs}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "dataset": "FAIR1M-2.0",
        "output_dir": str(output),
        "downloaded": downloaded,
        "extracted": extracted,
        "skipped": skipped,
        "failed_archives": len(failures),
        "pair_report": pairs,
        "complete": complete,
        "incomplete_manifest": str(incomplete) if not complete else None,
        "test_images_downloaded": False,
        "source_mode": source,
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            download(args.output_dir, args.overwrite, args.source),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
