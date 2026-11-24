from __future__ import annotations

import json
import os
from pathlib import Path


FORMAT_VERSION = "ovrs-lmdb-v1"
INDEX_KEY = b"__index__"
META_KEY = b"__metadata__"
_ENVIRONMENT_PID = os.getpid()
_ENVIRONMENTS: dict[str, dict] = {}


def _reset_after_fork() -> None:
    global _ENVIRONMENT_PID
    pid = os.getpid()
    if pid == _ENVIRONMENT_PID:
        return
    for entry in _ENVIRONMENTS.values():
        entry["environment"].close()
    _ENVIRONMENTS.clear()
    _ENVIRONMENT_PID = pid


def _acquire_environment(path: Path):
    import lmdb

    _reset_after_fork()
    key = str(path)
    entry = _ENVIRONMENTS.get(key)
    if entry is None:
        entry = {
            "environment": lmdb.open(
                key,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
            ),
            "references": 0,
        }
        _ENVIRONMENTS[key] = entry
    entry["references"] += 1
    return entry["environment"]


def _release_environment(path: Path, pid: int | None) -> None:
    if pid != os.getpid():
        return
    entry = _ENVIRONMENTS.get(str(path))
    if entry is None:
        return
    entry["references"] -= 1
    if entry["references"] == 0:
        entry["environment"].close()
        del _ENVIRONMENTS[str(path)]


class LmdbDatasetStore:
    """Process-local read-only access to files packed below one dataset root."""

    def __init__(self, root: Path, path: Path) -> None:
        self.root = root
        self.path = path
        self._environment = None
        self._pid = None
        with self._env().begin() as transaction:
            self.paths = tuple(json.loads(transaction.get(INDEX_KEY).decode("utf-8")))

    def _env(self):
        pid = os.getpid()
        if self._environment is None or self._pid != pid:
            self.close()
            self._environment = _acquire_environment(self.path)
            self._pid = pid
        return self._environment

    def relative(self, path: str | Path) -> str:
        source = Path(path)
        if source.is_absolute():
            source = source.relative_to(self.root)
        return source.as_posix()

    def read(self, path: str | Path) -> bytes:
        relative = self.relative(path)
        with self._env().begin() as transaction:
            value = transaction.get(f"file:{relative}".encode("utf-8"))
        if value is None:
            raise FileNotFoundError(f"{relative} is absent from {self.path}")
        return value

    def index_files(self, directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
        prefix = self.relative(directory).rstrip("/") + "/"
        result = {}
        for relative in self.paths:
            if not relative.startswith(prefix) or Path(relative).suffix.lower() not in suffixes:
                continue
            sample_id = Path(relative[len(prefix):]).with_suffix("").as_posix()
            result[sample_id] = self.root / relative
        return result

    def has_prefix(self, directory: Path) -> bool:
        prefix = self.relative(directory).rstrip("/") + "/"
        return any(path.startswith(prefix) for path in self.paths)

    def close(self) -> None:
        if self._environment is not None:
            _release_environment(self.path, self._pid)
        self._environment = None
        self._pid = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_environment"] = None
        state["_pid"] = None
        return state


def open_lmdb_store(root: Path, metadata: dict) -> LmdbDatasetStore | None:
    storage = metadata.get("storage") or {}
    if storage.get("backend") != "lmdb":
        return None
    path = root / str(storage.get("path", "data.lmdb"))
    if not path.is_dir():
        raise FileNotFoundError(path)
    return LmdbDatasetStore(root, path)
