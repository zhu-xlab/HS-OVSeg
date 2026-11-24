#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401


OFFICIAL_GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official DIOR/DIOR-R release and extract images plus oriented annotations"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/Datasets/RSOD/DIOR-R"))
    parser.add_argument("--source-url", default=OFFICIAL_GOOGLE_DRIVE_FOLDER)
    parser.add_argument("--download-dir", type=Path, help="Optional persistent staging directory")
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--skip-disk-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes output directory: {member_name}")
    return destination


def _extract_archive(archive: Path, output: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                _safe_destination(output, member.filename)
            stream.extractall(output)
        return
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as stream:
            members = []
            for member in stream.getmembers():
                _safe_destination(output, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not accepted: {member.name}")
                members.append(member)
            stream.extractall(output, members=members)
        return
    raise ValueError(f"Unsupported archive: {archive}")


def _is_archive(path: Path) -> bool:
    return any(path.name.lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _copy_non_archives(staging: Path, output: Path, overwrite: bool) -> int:
    copied = 0
    for source in staging.rglob("*"):
        if not source.is_file() or _is_archive(source):
            continue
        destination = output / source.relative_to(staging)
        if destination.exists() and not overwrite:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _download_official_folder(gdown_module, source_url: str, download_root: Path):
    """Call gdown across releases with different download_folder signatures."""
    supported = inspect.signature(gdown_module.download_folder).parameters
    options = {
        "url": source_url,
        "output": str(download_root),
        "quiet": False,
        "use_cookies": False,
        "remaining_ok": True,
    }
    return gdown_module.download_folder(
        **{name: value for name, value in options.items() if name in supported}
    )


def _prepare(staging: Path, output: Path, source_url: str, overwrite: bool) -> dict:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install download support with: pip install gdown") from exc
    download_root = staging / "official_release"
    download_root.mkdir(parents=True, exist_ok=True)
    downloaded = _download_official_folder(gdown, source_url, download_root)
    if not downloaded:
        raise RuntimeError("Google Drive returned no files; verify access to the official folder")
    output.mkdir(parents=True, exist_ok=True)
    archives = sorted(path for path in download_root.rglob("*") if path.is_file() and _is_archive(path))
    for archive in archives:
        print(f"Extracting {archive.name} ...", flush=True)
        _extract_archive(archive, output)
    copied = _copy_non_archives(download_root, output, overwrite)
    oriented_xml = [
        path for path in output.rglob("*.xml")
        if "oriented" in " ".join(part.lower() for part in path.parts)
    ]
    images = [path for path in output.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not oriented_xml:
        raise FileNotFoundError("No XML below an Oriented_Bounding_Boxes directory was found")
    if not images:
        raise FileNotFoundError("No DIOR-R image files were found")
    return {
        "source": source_url,
        "output_root": str(output),
        "downloaded_files": len(downloaded),
        "archives_extracted": len(archives),
        "raw_files_copied": copied,
        "images_found": len(images),
        "oriented_annotations_found": len(oriented_xml),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if not args.skip_disk_check:
        ancestor = output
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        free_gb = shutil.disk_usage(ancestor).free / (1024**3)
        if free_gb < args.minimum_free_gb:
            raise SystemExit(f"Only {free_gb:.1f} GiB free; require {args.minimum_free_gb:.1f} GiB")
    if args.download_dir:
        staging = args.download_dir.expanduser().resolve()
        staging.mkdir(parents=True, exist_ok=True)
        manifest = _prepare(staging, output, args.source_url, args.overwrite)
    else:
        print("Using an automatically removed temporary download directory.", flush=True)
        with tempfile.TemporaryDirectory(prefix="dior-r-") as temporary:
            manifest = _prepare(Path(temporary), output, args.source_url, args.overwrite)
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
