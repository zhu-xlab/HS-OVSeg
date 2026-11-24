#!/usr/bin/env python
"""Small cross-platform Git diff and commit/push workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 20 * 1024 * 1024
SENSITIVE_NAMES = {".env", "credentials.json", "id_rsa", "id_ed25519"}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


def git(*args: str, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode:
        message = (result.stderr or result.stdout or "git command failed").strip()
        raise RuntimeError(message)
    return (result.stdout or "").strip()


def require_repository() -> None:
    if git("rev-parse", "--is-inside-work-tree", capture=True, check=False) != "true":
        raise RuntimeError(f"Not a Git repository: {ROOT}")


def show_diff(staged: bool, stat_only: bool) -> None:
    require_repository()
    print(git("status", "--short", "--branch", capture=True) or "working tree clean")
    mode = ["--cached"] if staged else []
    summary = git("--no-pager", "diff", *mode, "--stat", capture=True)
    if summary:
        print("\n" + summary)
    if not stat_only:
        detail = git("--no-pager", "diff", *mode, "--", capture=True)
        if detail:
            print("\n" + detail)


def configure_remote(url: str, remote: str) -> None:
    require_repository()
    remotes = set(git("remote", capture=True).splitlines())
    if remote in remotes:
        git("remote", "set-url", remote, url)
    else:
        git("remote", "add", remote, url)
    print(f"{remote} -> {git('remote', 'get-url', remote, capture=True)}")


def staged_paths() -> list[Path]:
    names = git(
        "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", capture=True
    )
    return [ROOT / name for name in names.split("\0") if name]


def validate_staged_files(allow_large: bool) -> None:
    violations = []
    for path in staged_paths():
        name = path.name.lower()
        if name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
            violations.append(f"sensitive file: {path.relative_to(ROOT)}")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES and not allow_large:
            size = path.stat().st_size / (1024 * 1024)
            violations.append(f"large file ({size:.1f} MiB): {path.relative_to(ROOT)}")
    if violations:
        raise RuntimeError("Refusing to upload:\n  " + "\n  ".join(violations))


def sync(message: str, remote: str, tracked_only: bool, allow_large: bool) -> None:
    require_repository()
    if remote not in set(git("remote", capture=True).splitlines()):
        raise RuntimeError(
            f"Remote {remote!r} is not configured. Run:\n"
            f"  python tools/git_workflow.py remote <repository-url>"
        )
    git("add", "-u" if tracked_only else "-A")
    git("diff", "--cached", "--check")
    validate_staged_files(allow_large)
    has_changes = (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        != 0
    )
    if has_changes:
        print(git("diff", "--cached", "--stat", capture=True))
        git("commit", "-m", message)
    else:
        print("No staged changes; pushing existing commits.")
    git("push", "-u", remote, "HEAD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diff = commands.add_parser("diff", help="Show status and working-tree diff")
    diff.add_argument("--staged", action="store_true")
    diff.add_argument("--stat-only", action="store_true")
    remote = commands.add_parser("remote", help="Add or replace a Git remote")
    remote.add_argument("url")
    remote.add_argument("--name", default="origin")
    upload = commands.add_parser("sync", help="Stage, commit, and push safely")
    upload.add_argument("-m", "--message", required=True)
    upload.add_argument("--remote", default="origin")
    upload.add_argument("--tracked-only", action="store_true")
    upload.add_argument("--allow-large", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "diff":
            show_diff(args.staged, args.stat_only)
        elif args.command == "remote":
            configure_remote(args.url, args.name)
        else:
            sync(args.message, args.remote, args.tracked_only, args.allow_large)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
