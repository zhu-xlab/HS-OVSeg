"""Publish selected data as tar shards, requiring explicit public-upload opt-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ovrs_segmentation.preprocessing.huggingface_data import dataset_entries, validate_datasets
from ovrs_segmentation.preprocessing.huggingface_export import archive_dataset, export_dataset


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish(repo_id, data_root, work_root, *, shard_bytes=4 * 1024**3, allow_public=False):
    from huggingface_hub import HfApi

    api = HfApi()
    if not api.repo_info(repo_id, repo_type="dataset").private and not allow_public:
        raise ValueError("Public upload requires explicit --allow-public")
    source, work = Path(data_root).resolve(), Path(work_root).resolve()
    if source == work or source in work.parents or work in source.parents:
        raise ValueError("Publication work root must be separate from source data")
    work.mkdir(parents=True, exist_ok=True)
    status_file = work / "publication-status.json"
    results = []
    entries = dataset_entries(source, "all", scope="training_source")
    grouped = {}
    for entry in entries:
        grouped.setdefault(Path(entry["root"]).resolve(), []).append(entry)

    def status(phase, **extra):
        value = dict(repo_id=repo_id, phase=phase, datasets_done=len(results),
                     datasets_total=len(grouped), datasets=results, **extra)
        _write_json(status_file, value)
        print(json.dumps({k: v for k, v in value.items() if k != "datasets"}), flush=True)

    def remote_sha(path):
        rows = api.get_paths_info(repo_id, [path], repo_type="dataset")
        if rows and getattr(rows[0], "lfs", None):
            return rows[0].lfs.sha256
        return None

    def upload(path, relative, sha):
        if remote_sha(relative) == sha:
            return
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=relative,
                    repo_id=repo_id, repo_type="dataset",
                    commit_message=f"Publish selected data: {relative}")
                if remote_sha(relative) != sha:
                    raise RuntimeError(f"Uploaded checksum mismatch: {relative}")
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(30, 2**attempt))

    try:
        for root, rows in grouped.items():
            relative = root.relative_to(source)
            selected = work / "selected" / relative
            result_path = work / "records" / relative / "export.json"
            status("exporting", current=relative.as_posix())
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                validate_datasets([{**entry, "root": selected} for entry in rows])
            else:
                result = export_dataset(root, selected, [row["split"] for row in rows],
                    reader=rows[0]["reader"], level=rows[0]["source_task"],
                    progress=lambda **counts: status("exporting", current=relative.as_posix(), **counts))
                result_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(result_path, result)
            shards_path = result_path.with_name("shards.json")
            if shards_path.is_file():
                shards = json.loads(shards_path.read_text())
            else:
                status("archiving", current=relative.as_posix())
                shards = archive_dataset(selected, relative, work / "archives", shard_bytes=shard_bytes)
                _write_json(shards_path, shards)
            public_shards = []
            for index, shard in enumerate(shards):
                path = Path(shard["path"])
                remote_path = path.relative_to(work).as_posix()
                status("uploading", current=relative.as_posix(), shard=index + 1,
                       shards_total=len(shards), bytes=shard["bytes"])
                upload(path, remote_path, shard["sha256"])
                public_shards.append({**shard, "path": remote_path})
            results.append(dict(directory=relative.as_posix(), **result, shards=public_shards))
        manifest = work / "data-manifest.json"
        _write_json(manifest, dict(complete=True, scope="training_source", storage="tar_shards",
                    datasets=results))
        api.upload_file(path_or_fileobj=str(manifest), path_in_repo=manifest.name,
                        repo_id=repo_id, repo_type="dataset", commit_message="Complete selected-data manifest")
        status("complete")
        return results
    except Exception as error:
        # Keep access tokens out of persistent status and logs.
        status("failed", error_type=type(error).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--shard-gib", type=float, default=4)
    parser.add_argument("--allow-public", action="store_true", help="Explicitly allow a public destination")
    args = parser.parse_args()
    if args.shard_gib <= 0:
        parser.error("--shard-gib must be positive")
    publish(args.repo_id, args.data_root, args.work_root,
            shard_bytes=int(args.shard_gib * 1024**3), allow_public=args.allow_public)


if __name__ == "__main__":
    main()
