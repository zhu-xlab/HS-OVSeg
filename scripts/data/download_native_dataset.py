#!/usr/bin/env python3
"""Download or import native remote-sensing segmentation datasets."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import inspect
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASETS = {
    "openearthmap": {
        "root": "OpenEarthMap",
        "urls": [("OpenEarthMap.zip", "https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1")],
    },
    "loveda": {
        "root": "LoveDA",
        "urls": [
            ("Train.zip", "https://zenodo.org/records/5706578/files/Train.zip?download=1"),
            ("Val.zip", "https://zenodo.org/records/5706578/files/Val.zip?download=1"),
        ],
    },
    "landcoverai": {
        "root": "LandCover.ai",
        "urls": [("landcover.ai.v1.zip", "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip")],
    },
    "skyscapes": {
        "root": "SkyScapes",
        "urls": [("SkyScapes.zip", "https://pba-freesoftware.eoc.dlr.de/SkyScapes.zip")],
    },
    "semantic-drone": {
        "root": "Semantic-Drone",
        "urls": [(
            "semantic_drone_dataset_semantics_v1.1.zip",
            "https://cloud.tugraz.at/public.php/webdav/semantic_drone_dataset_semantics_v1.1.zip",
        )],
        "download_headers": {
            "Authorization": "Basic " + base64.b64encode(b"csxYfaKmie6LyqA:").decode("ascii")
        },
        "license_acceptance_required": True,
    },
    "gid15": {
        "root": "GID-15",
        "gdrive_file_id": "1zbkCEXPEKEV6gq19OKmIbaT8bXXfWW6u",
        "filename": "gid-15.zip",
        "md5": {"gid-15.zip": "615682bf659c3ed981826c6122c10c83"},
    },
    "isaid": {
        "root": "iSAID",
        "gdrive_folders": [
            ("train", "https://drive.google.com/drive/folders/19RPVhC0dWpLF9Y_DYjxjUrwLbKUBQZ2K"),
            ("val", "https://drive.google.com/drive/folders/17MErPhWQrwr92Ca1Maf4mwiarPS5rcWM"),
        ],
        # iSAID uses the original DOTA-v1.0 RGB images. The official iSAID
        # Drive folders may contain only annotations, so fetch the matching
        # image archives explicitly and keep them inside this dataset root.
        "dota_repo": "isaaccorley/dota",
        "dota_image_archives": [
            ("train", "dotav1.0_images_train.tar.gz"),
            ("val", "dotav1.0_images_val.tar.gz"),
        ],
    },
    "isprs-potsdam": {"root": "ISPRS-Potsdam", "manual": True},
    "isprs-vaihingen": {"root": "ISPRS-Vaihingen", "manual": True},
    "uavid": {"root": "UAVid", "manual": True},
    "five-billion-pixels": {
        "root": "Five-Billion-Pixels",
        "gdrive_folder": "https://drive.google.com/drive/folders/1924VnO08Gqo3Nv7Y4KirgJ9kqqCup7f0",
    },
    "floodnet": {
        "root": "FloodNet",
        "urls": [("FloodNet.zip", "https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1")],
    },
    "rescuenet": {
        "root": "RescueNet",
        "urls": [
            ("segmentation-trainset.zip", "https://ndownloader.figshare.com/files/40581458"),
            ("segmentation-validationset.zip", "https://ndownloader.figshare.com/files/40582916"),
            ("segmentation-testset.zip", "https://ndownloader.figshare.com/files/40583084"),
        ],
        "md5": {
            "segmentation-trainset.zip": "7ab79503e940de6301098a01a035a62f",
            "segmentation-validationset.zip": "51360be8ea3f73233605a767801c9d1c",
            "segmentation-testset.zip": "6dbd095cf14d9ddfdf1312d5f1949aeb",
        },
    },
    "deepglobe-landcover": {
        "root": "DeepGlobe-LandCover",
        "urls": [("deepglobe-landcover.zip", "https://www.kaggle.com/api/v1/datasets/download/geoap96/deepglobe2018-landcover-segmentation-traindataset")],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/import a native semantic dataset")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to /home/Datasets/RSOD/<dataset> on Linux or datasets/<dataset> on Windows.",
    )
    parser.add_argument(
        "--archive",
        action="append",
        type=Path,
        default=[],
        help="Locally downloaded ZIP/TAR archive; repeat for datasets requiring manual access.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm acceptance of a dataset's non-commercial/research license when required.",
    )
    return parser.parse_args()


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        # Dropbox folder exports may contain a single directory entry named
        # "/" to represent the shared-folder root.  It has no payload and
        # should not be treated as an absolute file extraction target.
        members = [
            member
            for member in stream.infolist()
            if not (
                member.is_dir()
                and not member.filename.replace("\\", "/").strip("/")
            )
        ]
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


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.name}")
        total = sum(member.size for member in members if member.isfile())
        with tqdm(total=total, desc=f"Extracting {archive.name}", unit="B", unit_scale=True) as progress:
            for member in members:
                stream.extract(member, root)
                if member.isfile():
                    progress.update(member.size)


def _extract(archive: Path, destination: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        _safe_extract_zip(archive, destination)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        _safe_extract_tar(archive, destination)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")


def _validate_archive(archive: Path, expected_md5: str | None = None) -> None:
    """Reject provider HTML/error pages and corrupted downloads before extraction."""
    if archive.name.lower().endswith(".zip") and not zipfile.is_zipfile(archive):
        size = archive.stat().st_size if archive.is_file() else 0
        raise ValueError(
            f"Downloaded file is not a ZIP archive: {archive} ({size} bytes). "
            "The provider likely returned an HTML error/share page."
        )
    if expected_md5:
        digest = hashlib.md5()  # nosec B324 - official integrity checksum, not cryptography
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual.lower() != expected_md5.lower():
            raise ValueError(
                f"Checksum mismatch for {archive}: expected MD5 {expected_md5}, got {actual}. "
                "Rerun with --overwrite to download a clean copy."
            )


def _extraction_directory(raw_dir: Path, archive: Path) -> Path:
    name = archive.name
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return raw_dir / name


def _download(url: str, destination: Path, request_headers: dict[str, str] | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "OVRS-Segmentation/0.1", **(request_headers or {})}
    if "kaggle.com/api/" in url:
        username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
        config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        credential_file = config_dir / "kaggle.json"
        if (not username or not key) and credential_file.is_file():
            credentials = json.loads(credential_file.read_text(encoding="utf-8"))
            username, key = credentials.get("username"), credentials.get("key")
        if username and key:
            token = base64.b64encode(f"{username}:{key}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
    request = urllib.request.Request(url, headers=headers)
    if start:
        request.add_header("Range", f"bytes={start}-")
    print(
        f"Connecting: {url}\n"
        f"Destination: {destination}\n"
        f"Existing partial download: {start / (1024 ** 3):.2f} GiB",
        flush=True,
    )
    try:
        response_context = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403} and "kaggle.com/api/" in url:
            raise RuntimeError(
                "Kaggle rejected the DeepGlobe download. Set KAGGLE_USERNAME/KAGGLE_KEY "
                "or place kaggle.json in ~/.kaggle, then rerun the same command."
            ) from error
        raise
    with response_context as response:
        append = start > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if append else "wb"
        initial = start if append else 0
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            total = int(match.group(1))
        else:
            remaining = int(response.headers.get("Content-Length", 0) or 0)
            total = initial + remaining if remaining else None
        with partial.open(mode) as stream, tqdm(
            total=total,
            initial=initial,
            desc=destination.name,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                progress.update(len(block))
    partial.replace(destination)
    print(f"Download complete: {destination}", flush=True)


def _extract_nested_archives(root: Path) -> int:
    count = 0
    for archive in sorted(path for path in root.rglob("*") if path.is_file()):
        lower = archive.name.lower()
        if not lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            continue
        destination = archive.parent / (archive.name.split(".tar", 1)[0] if ".tar" in lower else archive.stem)
        marker = destination / ".extraction-complete"
        if marker.is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        _extract(archive, destination)
        marker.write_text(str(archive) + "\n", encoding="utf-8")
        count += 1
    return count


def _contains_rasters(root: Path) -> bool:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return root.is_dir() and any(
        path.is_file() and path.suffix.lower() in suffixes for path in root.rglob("*")
    )


def _select_isaid_semantic_archive(entries):
    matches = []
    for entry in entries or []:
        relative = str(getattr(entry, "path", "")).replace("\\", "/").lower().strip("/")
        if relative.endswith("semantic_masks/images.zip"):
            matches.append(entry)
    if len(matches) != 1:
        available = [str(getattr(entry, "path", "")) for entry in entries or []]
        raise RuntimeError(
            "Expected exactly one official iSAID Semantic_masks/images.zip archive; "
            f"found {len(matches)}. Google Drive entries: {available}"
        )
    return matches[0]


def _download_isaid(raw_dir: Path, overwrite: bool) -> dict:
    """Download semantic labels from iSAID and matching RGB images from DOTA-v1."""
    try:
        import gdown
    except ImportError as error:
        raise ImportError("gdown is required; run: python -m pip install -e .") from error

    downloaded = 0
    semantic_archives_extracted = 0
    for split, url in DATASETS["isaid"]["gdrive_folders"]:
        destination = raw_dir / split / "Semantic_masks"
        marker = destination / ".download-complete"
        if not overwrite and (marker.is_file() or _contains_rasters(destination)):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(url + "\n", encoding="utf-8")
            continue
        if overwrite and destination.exists():
            shutil.rmtree(destination)
        entries = gdown.download_folder(
            url=url,
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )
        entry = _select_isaid_semantic_archive(entries)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"isaid-labels-{split}-") as temporary:
            archive = Path(temporary) / f"isaid_{split}_semantic_masks.zip"
            result = gdown.download(
                id=entry.id,
                output=str(archive),
                quiet=False,
                use_cookies=False,
                resume=not overwrite,
            )
            if result is None or not archive.is_file():
                raise RuntimeError(f"Google Drive failed to download iSAID {split} semantic masks")
            _validate_archive(archive)
            _safe_extract_zip(archive, destination)
        marker.write_text(url + "\n", encoding="utf-8")
        downloaded += 1
        semantic_archives_extracted += 1

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("huggingface_hub is required; run: python -m pip install -e .") from error
    image_archives_extracted = 0
    for split, filename in DATASETS["isaid"]["dota_image_archives"]:
        destination = raw_dir / "dota_images" / split
        marker = destination / ".extraction-complete"
        if marker.is_file() and not overwrite:
            continue
        if destination.exists() and overwrite:
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"isaid-dota-{split}-") as temporary:
            archive = Path(hf_hub_download(
                repo_id=DATASETS["isaid"]["dota_repo"],
                filename=filename,
                repo_type="dataset",
                cache_dir=temporary,
            ))
            _safe_extract_tar(archive, destination)
        marker.write_text(filename + "\n", encoding="utf-8")
        image_archives_extracted += 1

    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = [
        path for path in (raw_dir / "dota_images").rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    ]
    semantic_masks = [
        path for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_suffixes
        and "semantic_masks" in {part.lower() for part in path.parts}
    ]

    def canonical(path: Path) -> str:
        stem = path.stem.lower()
        for suffix in ("_instance_color_rgb", "_instance_id_rgb", "_rgb"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        return stem

    image_ids = {canonical(path) for path in image_files}
    mask_ids = {canonical(path) for path in semantic_masks}
    missing_images = sorted(mask_ids - image_ids)
    matched = image_ids & mask_ids
    if not image_files or not semantic_masks or not matched or missing_images:
        raise RuntimeError(
            "Incomplete iSAID download after extraction: "
            f"images={len(image_files)}, semantic_masks={len(semantic_masks)}, "
            f"matched={len(matched)}, masks_without_images={len(missing_images)}. "
            f"Examples: {missing_images[:10]}"
        )
    complete_marker = raw_dir / ".isaid-download-complete"
    complete_marker.write_text(
        json.dumps({
            "images": len(image_files),
            "semantic_masks": len(semantic_masks),
            "matched": len(matched),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "downloaded_semantic_archives": downloaded,
        "semantic_archives_extracted": semantic_archives_extracted,
        "image_archives_extracted": image_archives_extracted,
        "image_files": len(image_files),
        "semantic_mask_files": len(semantic_masks),
        "matched_samples": len(matched),
        "rgb_source": "DOTA-v1.0 train/val",
        "instance_masks_downloaded": False,
        "complete": True,
    }


def _download_gdrive_dataset(dataset: str, raw_dir: Path, overwrite: bool) -> dict:
    try:
        import gdown
    except ImportError as error:
        raise ImportError("gdown is required; run: python -m pip install -e .") from error
    config = DATASETS[dataset]
    destination = raw_dir / config["root"]
    marker = destination / ".download-complete"
    if marker.is_file() and not overwrite:
        return {"downloaded_files": 0, "nested_archives_extracted": _extract_nested_archives(raw_dir)}
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "url": config["gdrive_folder"],
        "output": str(destination),
        "quiet": False,
        "use_cookies": False,
    }
    # gdown 5.x exposed remaining_ok while 6.x removed it.  Only pass the
    # option when the installed version supports it.
    if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
        kwargs["remaining_ok"] = True
    files = gdown.download_folder(**kwargs)
    if not files:
        raise RuntimeError(f"Google Drive returned no files for {dataset}")
    marker.write_text(config["gdrive_folder"] + "\n", encoding="utf-8")
    return {"downloaded_files": len(files), "nested_archives_extracted": _extract_nested_archives(raw_dir)}


def _download_gdrive_archive(
    dataset: str, raw_dir: Path, download_dir: Path, overwrite: bool
) -> dict:
    """Download one public Google Drive archive with resume and integrity checks."""
    try:
        import gdown
    except ImportError as error:
        raise ImportError("gdown is required; run: python -m pip install -e .") from error
    config = DATASETS[dataset]
    filename = config["filename"]
    archive = download_dir / filename
    destination = _extraction_directory(raw_dir, archive)
    marker = destination / ".extraction-complete"
    if marker.is_file() and not overwrite:
        return {"archives_extracted": 0, "downloaded_files": 0}
    if overwrite:
        archive.unlink(missing_ok=True)
        partial = archive.with_suffix(archive.suffix + ".part")
        partial.unlink(missing_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
    download_dir.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        result = gdown.download(
            id=config["gdrive_file_id"],
            output=str(archive),
            quiet=False,
            use_cookies=False,
            resume=not overwrite,
        )
        if result is None or not archive.is_file():
            raise RuntimeError(
                f"Google Drive failed to download {dataset}. Browser URL: "
                f"https://drive.google.com/uc?id={config['gdrive_file_id']}"
            )
        downloaded = 1
    else:
        downloaded = 0
    _validate_archive(archive, config.get("md5", {}).get(filename))
    destination.mkdir(parents=True, exist_ok=True)
    _extract(archive, destination)
    marker.write_text(filename + "\n", encoding="utf-8")
    archive.unlink()
    return {"archives_extracted": 1, "downloaded_files": downloaded}


def _download_five_billion_pixels(raw_dir: Path, overwrite: bool) -> dict:
    """Download only RGB-capable 8-bit imagery and index masks from the official folder."""
    try:
        import gdown
    except ImportError as error:
        raise ImportError("gdown is required; run: python -m pip install -e .") from error
    config = DATASETS["five-billion-pixels"]
    destination = raw_dir / config["root"]
    marker = destination / ".download-complete"
    if marker.is_file() and not overwrite:
        return {"downloaded_files": 0, "selected_official_files": 301}
    entries = gdown.download_folder(
        url=config["gdrive_folder"],
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )
    if not entries:
        raise RuntimeError("Google Drive returned no Five-Billion-Pixels files")
    selected = []
    for entry in entries:
        relative = Path(str(entry.path).replace("\\", "/"))
        top = relative.parts[0] if relative.parts else ""
        if top in {"Image__8bit_NirRGB", "Annotation__index"} or relative.name.lower().startswith("readme"):
            selected.append((entry, relative))
    image_count = sum(relative.parts[0] == "Image__8bit_NirRGB" for _, relative in selected)
    mask_count = sum(relative.parts[0] == "Annotation__index" for _, relative in selected)
    if image_count != 150 or mask_count != 150:
        raise RuntimeError(
            f"Unexpected official folder contents: images={image_count}, index_masks={mask_count}"
        )
    downloaded = 0
    failures = []
    consecutive_failures = 0
    for entry, relative in tqdm(selected, desc="Downloading Five-Billion-Pixels", unit="file"):
        output = destination / relative
        if output.is_file() and output.stat().st_size > 0 and not overwrite:
            consecutive_failures = 0
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(4):
            try:
                result = gdown.download(
                    id=entry.id,
                    output=str(output),
                    quiet=True,
                    use_cookies=attempt % 2 == 0,
                    resume=not overwrite,
                )
                if result is not None and output.is_file() and output.stat().st_size > 0:
                    downloaded += 1
                    consecutive_failures = 0
                    last_error = None
                    break
                last_error = "gdown returned no completed file"
            except Exception as error:  # gdown uses several version-specific exception classes
                last_error = f"{type(error).__name__}: {error}"
            if attempt < 3:
                delay = (2, 5, 10)[attempt]
                tqdm.write(
                    f"Retry {attempt + 2}/4 in {delay}s: {relative} "
                    f"(Drive id={entry.id})"
                )
                time.sleep(delay)
        if last_error is not None:
            consecutive_failures += 1
            failures.append({
                "path": relative.as_posix(),
                "drive_id": str(entry.id),
                "browser_url": f"https://drive.google.com/uc?id={entry.id}",
                "error": last_error,
            })
            tqdm.write(
                f"WARNING: skipped inaccessible official file after 4 attempts: "
                f"{relative} (Drive id={entry.id})"
            )
            # A run of failures normally means Google has rate-limited the
            # client globally. Stop rather than waiting for every remaining file.
            if consecutive_failures >= 5:
                tqdm.write("Google Drive appears globally rate-limited; stop and rerun later to resume.")
                break
    marker.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination / ".download-incomplete.json"
    failures_by_path = {failure["path"]: failure for failure in failures}
    missing = []
    for entry, relative in selected:
        output = destination / relative
        if output.is_file() and output.stat().st_size > 0:
            continue
        missing.append(failures_by_path.get(relative.as_posix(), {
            "path": relative.as_posix(),
            "drive_id": str(entry.id),
            "browser_url": f"https://drive.google.com/uc?id={entry.id}",
            "error": "not attempted because the run stopped after consecutive failures",
        }))
    if missing:
        incomplete.write_text(json.dumps(missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if marker.exists():
            marker.unlink()
    else:
        incomplete.unlink(missing_ok=True)
        marker.write_text(config["gdrive_folder"] + "\n", encoding="utf-8")
    existing = len(selected) - len(missing)
    return {
        "downloaded_files": downloaded,
        "selected_official_files": len(selected),
        "existing_files": existing,
        "failed_files": len(failures),
        "remaining_files": len(missing),
        "failed_manifest": str(incomplete) if missing else None,
        "complete": not missing,
        "skipped_official_directories": [
            "Image_16bit_BGRNir", "Annotation__color", "Coordinate_files"
        ],
    }


def download_or_import(
    dataset: str,
    output_dir: Path,
    archives: list[Path],
    overwrite: bool,
    accept_license: bool = False,
) -> dict:
    config = DATASETS[dataset]
    output_dir = output_dir.expanduser().resolve()
    raw_dir, download_dir = output_dir / "raw", output_dir / "downloads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    details = {}
    if config.get("license_acceptance_required") and not archives and not accept_license:
        raise RuntimeError(
            f"{dataset} requires explicit acceptance of its non-commercial dataset license. "
            "Review the official terms, then rerun with --accept-license."
        )
    if dataset == "isaid" and not archives:
        details.update(_download_isaid(raw_dir, overwrite))
    elif dataset == "five-billion-pixels" and not archives:
        details.update(_download_five_billion_pixels(raw_dir, overwrite))
    elif config.get("gdrive_folder") and not archives:
        details.update(_download_gdrive_dataset(dataset, raw_dir, overwrite))
    elif config.get("gdrive_file_id") and not archives:
        details.update(_download_gdrive_archive(dataset, raw_dir, download_dir, overwrite))
    else:
        selected_archives = [path.expanduser().resolve() for path in archives]
        if not selected_archives:
            if config.get("manual"):
                raise RuntimeError(
                    f"{dataset} requires official-site download/terms. Pass each downloaded file "
                    f"with --archive, then this script will safely extract it."
                )
            for filename, url in config.get("urls", []):
                archive = download_dir / filename
                marker = _extraction_directory(raw_dir, archive) / ".extraction-complete"
                if marker.is_file() and not overwrite:
                    continue
                if overwrite or not archive.is_file():
                    _download(url, archive, config.get("download_headers"))
                _validate_archive(archive, config.get("md5", {}).get(filename))
                selected_archives.append(archive)
        extracted = 0
        for archive in selected_archives:
            if not archive.is_file():
                raise FileNotFoundError(archive)
            # Automatic downloads were checksum-verified above. This cheap
            # signature check also covers user-supplied archives.
            _validate_archive(archive)
            destination = _extraction_directory(raw_dir, archive)
            marker = destination / ".extraction-complete"
            if marker.is_file() and not overwrite:
                continue
            if destination.exists() and overwrite:
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            _extract(archive, destination)
            marker.write_text(str(archive) + "\n", encoding="utf-8")
            extracted += 1
            if archive.parent == download_dir:
                archive.unlink()
        details["archives_extracted"] = extracted
        details["nested_archives_extracted"] = _extract_nested_archives(raw_dir)
    manifest = {
        "dataset": dataset,
        "output_dir": str(output_dir),
        "raw_dir": str(raw_dir),
        **details,
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = DATASETS[args.dataset]
    if args.output_dir is not None:
        output = args.output_dir
    elif os.name == "nt":
        output = Path("datasets") / config["root"]
    else:
        output = Path("/home/Datasets/RSOD") / config["root"]
    print(f"Dataset output directory: {output.expanduser().resolve()}", flush=True)
    print(json.dumps(
        download_or_import(
            args.dataset, output, args.archive, args.overwrite, args.accept_license
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
