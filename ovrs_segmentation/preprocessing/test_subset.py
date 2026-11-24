"""Fix a reproducible, source-balanced semantic test subset without changing training splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def source_group(sample_id: str) -> str:
    if re.search(r"__x\d+_y\d+$", sample_id):
        return re.sub(r"__x\d+_y\d+$", "", sample_id)
    if re.match(r"D\d{3}-\d{4}_AERIAL", sample_id):
        return re.sub(r"_\d+-\d+$", "", sample_id)
    parent = Path(sample_id).parent.as_posix()
    return parent if parent != "." else "all"


def balanced_pool(sample_ids, size, seed):
    groups = defaultdict(list)
    rank = lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest()
    for sample_id in sample_ids:
        groups[source_group(sample_id)].append(sample_id)
    groups = {key: sorted(values, key=rank) for key, values in groups.items()}
    order = sorted(groups, key=rank)
    result, offset = [], 0
    while len(result) < min(size, len(sample_ids)):
        for key in order:
            if offset < len(groups[key]):
                result.append(groups[key][offset])
                if len(result) == min(size, len(sample_ids)):
                    break
        offset += 1
    return result


def select_diverse(sample_ids, histograms, size, seed):
    """Keep balanced source quotas while preferring new classes and mask compositions."""
    if len(sample_ids) != len(set(sample_ids)) or size > len(sample_ids) or size < 1:
        raise ValueError("Selection requires unique IDs and a valid subset size")
    histograms = np.asarray(histograms, dtype=np.float64)
    if histograms.ndim != 2 or histograms.shape[0] != len(sample_ids):
        raise ValueError("One class histogram is required for every candidate")
    quotas = Counter(source_group(x) for x in balanced_pool(sample_ids, size, seed))
    features = np.sqrt(histograms / np.maximum(histograms.sum(axis=1, keepdims=True), 1))
    presence = histograms > 0
    frequency = np.maximum(presence.sum(axis=0), 1)
    groups = defaultdict(list)
    for i, sample_id in enumerate(sample_ids):
        groups[source_group(sample_id)].append(i)
    selected, class_counts, previous = [], np.zeros(histograms.shape[1]), defaultdict(list)
    nearest = np.ones(len(sample_ids))
    for _ in range(max(quotas.values())):
        for group, quota in quotas.items():
            if len(previous[group]) >= quota:
                continue
            candidates = np.asarray(groups[group])
            novelty = (presence[candidates] / ((1 + class_counts) * np.sqrt(frequency))).sum(axis=1)
            chosen = int(candidates[np.argmax(novelty + nearest[candidates])])
            distance = np.sum((features[candidates] - features[chosen]) ** 2, axis=1)
            nearest[candidates] = distance if not previous[group] else np.minimum(nearest[candidates], distance)
            groups[group].remove(chosen)
            previous[group].append(chosen)
            selected.append(chosen)
            class_counts += presence[chosen]
    return selected


def _write(path, text):
    temporary = path.with_name(path.name + ".subset-tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def prepare(root: Path, *, size=2000, seed=20261028, workers=8):
    from ovrs_segmentation.datasets.semantic_dataset import SemanticSegmentationDataset

    root = root.resolve()
    split = root / "splits/test.txt"
    full = root / "splits/test_full.txt"
    record = root / "splits/test_subset2000.json"
    ids = [x.strip() for x in (full if full.exists() else split).read_text().splitlines() if x.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate test IDs: {root}")
    if len(ids) <= size:
        return dict(dataset=root.name, samples=len(ids), changed=False)
    original_hash = hashlib.sha256((full if full.exists() else split).read_bytes()).hexdigest()
    if record.exists():
        prior = json.loads(record.read_text())
        if prior["seed"] != seed or prior["samples"] != size or prior["original_split_sha256"] != original_hash:
            raise ValueError(f"Existing subset has a different selection protocol: {record}")
        if hashlib.sha256(split.read_bytes()).hexdigest() != prior["selected_split_sha256"]:
            raise ValueError(f"Selected split was modified: {split}")
        return prior
    dataset = SemanticSegmentationDataset(root, "test_full" if full.exists() else "test", validate_masks=False)
    candidates = balanced_pool(ids, min(len(ids), size * 4), seed)
    class_ids = list(dataset.class_text_by_id)

    def histogram(sample_id):
        mask = dataset._read_mask(dataset._masks[sample_id])
        counts = np.bincount(mask.ravel().astype(np.int64), minlength=max(class_ids) + 1)
        return counts[class_ids]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = np.asarray(list(pool.map(histogram, candidates)))
    selected = select_diverse(candidates, counts, size, seed)
    selected_ids = sorted(candidates[i] for i in selected)
    if len(set(selected_ids)) != size or not set(selected_ids) <= set(ids):
        raise ValueError("Selected samples must be a unique subset of the original test split")
    for name in ["train", "val"]:
        other = root / f"splits/{name}.txt"
        if other.exists() and set(selected_ids) & set(other.read_text().splitlines()):
            raise ValueError(f"Selected test samples overlap {name}: {root}")
    available = (counts > 0).any(axis=0)
    covered = (counts[selected] > 0).any(axis=0)
    if np.any(available & ~covered):
        raise ValueError(f"Subset lost candidate-pool classes: {np.asarray(class_ids)[available & ~covered].tolist()}")
    content = "\n".join(selected_ids) + "\n"
    summary = dict(dataset=dataset.dataset_name, protocol="source-balanced-semantic-diversity-v1",
        samples=size, original_samples=len(ids), seed=seed, candidate_samples=len(candidates),
        original_split="test_full", original_split_sha256=original_hash,
        selected_split_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_groups_total=len({source_group(x) for x in ids}),
        source_groups_selected=len({source_group(x) for x in selected_ids}),
        selection_uses="sample provenance and candidate GT class histograms; no model predictions",
        candidate_class_image_counts={str(c):int(v) for c,v in zip(class_ids,(counts > 0).sum(axis=0))},
        selected_class_image_counts={str(c):int(v) for c,v in zip(class_ids,(counts[selected] > 0).sum(axis=0))})
    # Keep the original test list and metadata so the full official protocol is recoverable.
    if not full.exists():
        shutil.copyfile(split, full)
    meta_path = root / "dataset.json"
    meta = json.loads(meta_path.read_text())
    backup = root / "splits/test_full.dataset.json"
    if not backup.exists():
        shutil.copyfile(meta_path, backup)
    meta["evaluation_subset"] = summary
    if "split_counts" in meta:
        meta["split_counts"]["test_full"] = len(ids)
        meta["split_counts"]["test"] = size
    _write(split, content)
    _write(meta_path, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    _write(record, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for root in args.roots:
        print(json.dumps(prepare(root, size=args.size, seed=args.seed, workers=args.workers), ensure_ascii=False), flush=True)
