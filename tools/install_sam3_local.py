#!/usr/bin/env python3
"""Install the official SAM3 source below this OVRS project.

This is intended for bind-mounted Docker projects: mounting the project over
/home/Codes/OVRS_segmentation hides files that were placed there while the
image was built. Running this script once creates third_party/sam3 in the host
project and installs it editable into the active Python environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "third_party" / "sam3"
DEFAULT_REF = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone and editable-install SAM3 inside this project."
    )
    parser.add_argument(
        "--repository",
        default="https://github.com/facebookresearch/sam3.git",
        help="SAM3 Git repository URL.",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="Git branch, tag, or commit (default: the tested HS-OVSeg revision).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Install location; it must remain inside the OVRS project.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch the requested ref when an existing checkout is reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target.expanduser().resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"SAM3 must be stored inside {PROJECT_ROOT}, got {target}"
        ) from error

    package_marker = target / "sam3" / "__init__.py"
    if target.exists() and not package_marker.is_file():
        raise FileExistsError(
            f"{target} exists but is not a SAM3 source tree; move it aside first"
        )

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", args.repository, str(target)])
    elif args.update:
        _run(["git", "-C", str(target), "fetch", "--all", "--tags"])

    _run(["git", "-C", str(target), "checkout", args.ref])
    _run([sys.executable, "-m", "pip", "install", "setuptools<81", "-e", f"{target}[train,dev]"])

    verification = (
        "from pathlib import Path; import sam3; "
        f"root=Path({str(target)!r}).resolve(); "
        "loaded=Path(sam3.__file__).resolve(); "
        "assert loaded.is_relative_to(root), (loaded, root); "
        "print('SAM3 loaded from:', loaded)"
    )
    _run([sys.executable, "-c", verification])
    print(f"SAM3 is now part of this project at: {target}")


if __name__ == "__main__":
    main()
