from __future__ import annotations

import hashlib
from typing import Any

import torch


def _replace_prompt_rows(
    batch: dict[str, Any],
    *,
    prompts: list[str],
    sample_indices: list[int],
    level: str,
) -> dict[str, Any]:
    expanded = dict(batch)
    for key in (
        "annotations",
        "class_ids",
        "pixel_targets",
        "valid_mask",
        "boxes",
        "boxes_xyxy",
        "instance_box_masks",
        "scene_targets",
        "semantic_levels",
        "supervision_levels",
        "supervision_sources",
    ):
        expanded.pop(key, None)
    expanded.update(
        {
            "prompt_texts": prompts,
            "task_type": level,
            "annotation_sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        }
    )
    return expanded


def _sample_prompt(
    row: dict[str, Any], seed_token: str, *, augment: bool = True
) -> str:
    if not augment:
        return str(row["native_prompt_texts"][0])
    digest = hashlib.sha1(seed_token.encode("utf-8")).hexdigest()
    groups = [
        values
        for values in (
            row["native_prompt_texts"],
            row["canonical_prompt_texts"],
            row["variant_prompt_texts"],
        )
        if values
    ]
    variants = groups[int(digest[:8], 16) % len(groups)]
    return str(variants[int(digest[8:16], 16) % len(variants)])


def _limit_prompt_candidates(
    candidates: list[dict[str, Any]],
    *,
    key_name: str,
    positive_keys: set[int],
    priority_negative_keys: set[int] | None,
    max_negatives: int | None,
    seed_token: str,
) -> list[dict[str, Any]]:
    """Keep every positive and a deterministic, bounded set of negatives."""
    if max_negatives is None:
        return candidates
    if max_negatives < 0:
        raise ValueError("max_negatives must be non-negative or None")
    positives = [row for row in candidates if int(row[key_name]) in positive_keys]
    negatives = [row for row in candidates if int(row[key_name]) not in positive_keys]
    priority_negative_keys = priority_negative_keys or set()

    def rank(row: dict[str, Any]) -> str:
        token = f"{seed_token}:{int(row[key_name])}".encode("utf-8")
        return hashlib.sha1(token).hexdigest()

    hard = sorted(
        [row for row in negatives if int(row[key_name]) in priority_negative_keys],
        key=rank,
    )
    random_pool = sorted(
        [row for row in negatives if int(row[key_name]) not in priority_negative_keys],
        key=rank,
    )
    hard_budget = min(len(hard), (max_negatives + 1) // 2)
    selected = hard[:hard_budget]
    selected.extend(random_pool[: max_negatives - len(selected)])
    if len(selected) < max_negatives:
        selected.extend(hard[hard_budget : hard_budget + max_negatives - len(selected)])
    return positives + selected


def _limit_total_prompt_candidates(
    candidates: list[dict[str, Any]],
    positive_keys: set[int],
    max_prompts: int | None,
    seed_token: str,
) -> list[dict[str, Any]]:
    if max_prompts is None:
        return candidates

    def rank(row: dict[str, Any]) -> str:
        token = f"{seed_token}:{int(row['local_class_id'])}".encode("utf-8")
        return hashlib.sha1(token).hexdigest()

    positives = sorted(
        [row for row in candidates if int(row["local_class_id"]) in positive_keys],
        key=rank,
    )
    negatives = sorted(
        [row for row in candidates if int(row["local_class_id"]) not in positive_keys],
        key=rank,
    )
    selected = positives[:max_prompts]
    selected.extend(negatives[: max_prompts - len(selected)])
    return selected


def expand_dataset_local_prompt_pairs(
    batch: dict[str, Any],
    level: str,
    *,
    max_prompts: int | None = None,
    sampling_seed: int = 0,
    augment_prompts: bool = True,
) -> dict[str, Any]:
    """Expand each image against only its own complete dataset vocabulary."""
    if level not in {"scene", "lulc"}:
        raise ValueError("Dataset-local prompt expansion only supports scene/lulc")
    height, width = map(int, batch["images"].shape[-2:])
    prompts: list[str] = []
    sample_indices: list[int] = []
    pair_targets: list[torch.Tensor] = []
    scene_targets: list[float] = []
    scene_local_targets = torch.full((len(batch["sample_ids"]),), -100, dtype=torch.long)
    for sample_index, sample_id in enumerate(batch["sample_ids"]):
        vocabulary = batch["dataset_vocabularies"][sample_index][level]
        classes = vocabulary["classes"]
        local_ids = [int(row["local_class_id"]) for row in classes]
        if local_ids != list(range(len(classes))):
            raise ValueError(
                f"Dataset-local IDs must be contiguous for {vocabulary['dataset_name']}"
            )
        if level == "scene":
            positive_classes = sorted(set(batch["scene_local_class_ids"][sample_index]))
            is_single_label = bool(vocabulary.get("single_label", True))
            if is_single_label:
                if len(positive_classes) != 1:
                    raise ValueError(
                        f"Single-label Scene sample {sample_id!r} has labels {positive_classes}"
                    )
                scene_local_targets[sample_index] = positive_classes[0]
        else:
            local_mask = batch["dataset_local_pixel_masks"][sample_index]
            local_valid = batch["dataset_local_pixel_valid_masks"][sample_index]
            positive_classes = sorted(
                int(value) for value in local_mask.unique().tolist() if int(value) != 255
            )

        seed_token = f"{sampling_seed}:{sample_id}:{level}"
        if max_prompts is not None:
            classes = _limit_total_prompt_candidates(
                classes, set(positive_classes), max_prompts, seed_token
            )

        for row in classes:
            local_class_id = int(row["local_class_id"])
            if level == "scene":
                present = local_class_id in positive_classes
                target = None
            else:
                target = torch.full((height, width), 255, dtype=torch.uint8)
                target[local_valid] = 0
                target[local_valid & local_mask.eq(local_class_id)] = 1
                present = bool(target.eq(1).any())
                pair_targets.append(target)
            prompt = _sample_prompt(
                row,
                f"{seed_token}:{local_class_id}:prompt",
                augment=augment_prompts,
            )
            prompts.append(prompt)
            sample_indices.append(sample_index)
            scene_targets.append(float(present) if level == "scene" else float("nan"))

    count = len(prompts)
    expanded = _replace_prompt_rows(
        batch,
        prompts=prompts,
        sample_indices=sample_indices,
        level=level,
    )
    prompt_owner = torch.tensor(sample_indices, dtype=torch.long)
    pixel_targets = (
        torch.stack(pair_targets)
        if level == "lulc"
        else torch.full((count, height, width), 255, dtype=torch.uint8)
    )
    expanded.update(
        {
            "multiclass_scene_targets": scene_local_targets,
            "multiclass_pixel_targets": batch["dataset_local_pixel_masks"],
            "pixel_targets": pixel_targets,
            "valid_mask": batch["valid_image_masks"].index_select(0, prompt_owner),
            "scene_targets": torch.tensor(scene_targets, dtype=torch.float32),
        }
    )
    if level == "lulc":
        expanded["valid_mask"] &= pixel_targets.ne(255)
    return expanded


def expand_object_prompt_pairs(
    batch: dict[str, Any],
    *,
    max_prompts: int | None = None,
    max_instances: int = 32,
    sampling_seed: int = 0,
) -> dict[str, Any]:
    """Sample Object instances, then build prompts for selected classes."""
    owners = batch["annotation_sample_indices"].tolist()
    positives = {index: {} for index in range(len(batch["sample_ids"]))}
    for annotation_index, sample_index in enumerate(owners):
        class_id = int(batch["annotations"][annotation_index]["dataset_class_id"])
        positives[sample_index].setdefault(class_id, []).append(annotation_index)

    prompted_positive_classes = {}
    for sample_index, sample_groups in positives.items():
        sample_id = batch["sample_ids"][sample_index]

        def class_rank(class_id: int) -> str:
            token = f"{sampling_seed}:{sample_id}:object:{class_id}".encode()
            return hashlib.sha1(token).hexdigest()

        class_ids = sorted(
            sample_groups,
            key=class_rank,
        )
        prompted_positive_classes[sample_index] = set(
            class_ids if max_prompts is None else class_ids[:max_prompts]
        )

    groups = [
        (sample_index, class_id, annotation_indices)
        for sample_index, sample_groups in positives.items()
        for class_id, annotation_indices in sample_groups.items()
        if class_id in prompted_positive_classes[sample_index]
    ]
    group_boxes = {
        (sample_index, class_id): [
            (annotation_index, box_index)
            for annotation_index in annotation_indices
            for box_index in range(len(batch["boxes"][annotation_index]))
        ]
        for sample_index, class_id, annotation_indices in groups
    }
    generator = torch.Generator().manual_seed(sampling_seed)
    selected_pairs: set[tuple[int, int]] = set()
    group_order = torch.randperm(len(groups), generator=generator).tolist()
    for position in group_order[:max_instances]:
        sample_index, class_id, _ = groups[position]
        candidates = group_boxes[(sample_index, class_id)]
        if candidates:
            choice = int(torch.randint(len(candidates), (), generator=generator))
            selected_pairs.add(candidates[choice])
    remaining = [
        pair
        for candidates in group_boxes.values()
        for pair in candidates
        if pair not in selected_pairs
    ]
    slots = max_instances - len(selected_pairs)
    if slots > 0 and remaining:
        order = torch.randperm(len(remaining), generator=generator)[:slots].tolist()
        selected_pairs.update(remaining[position] for position in order)

    selected_by_annotation: dict[int, list[int]] = {}
    for annotation_index, box_index in selected_pairs:
        selected_by_annotation.setdefault(annotation_index, []).append(box_index)
    selected_positives = {index: {} for index in positives}
    ignored_by_group: dict[tuple[int, int], torch.Tensor] = {}
    for sample_index, class_id, annotation_indices in groups:
        chosen_annotations = [
            index for index in annotation_indices if index in selected_by_annotation
        ]
        if chosen_annotations:
            selected_positives[sample_index][class_id] = chosen_annotations
        ignored = []
        for annotation_index in annotation_indices:
            boxes = batch["boxes"][annotation_index]
            chosen = set(selected_by_annotation.get(annotation_index, []))
            omitted = [index for index in range(len(boxes)) if index not in chosen]
            if omitted:
                ignored.append(boxes[omitted])
        ignored_by_group[(sample_index, class_id)] = (
            torch.cat(ignored) if ignored else torch.empty(0, 4, dtype=torch.float32)
        )

    catalogs: dict[int, list[dict[str, Any]]] = {}
    for sample_index in range(len(batch["sample_ids"])):
        vocabulary = batch["dataset_vocabularies"][sample_index]["object"]
        classes = vocabulary["classes"]
        dataset_id = str(vocabulary["dataset_id"])
        catalogs[sample_index] = [
            {
                "native_prompt_texts": list(row["native_prompt_texts"]),
                "canonical_prompt_texts": list(row["canonical_prompt_texts"]),
                "variant_prompt_texts": list(row["variant_prompt_texts"]),
                "dataset_class_id": int(row["local_class_id"]),
            }
            for row in classes
        ]

    height, width = map(int, batch["images"].shape[-2:])
    prompts, sample_indices, pair_boxes, ignored_boxes = [], [], [], []
    truncated_rows = []
    for sample_index, positive_groups in selected_positives.items():
        all_positive_groups = positives[sample_index]
        vocabulary = batch["dataset_vocabularies"][sample_index]["object"]
        dataset_id = str(vocabulary["dataset_id"])
        priority_negative_classes: set[int] = set()
        for other_index, other_vocabulary_set in enumerate(batch["dataset_vocabularies"]):
            if other_index == sample_index:
                continue
            other_vocabulary = other_vocabulary_set["object"]
            if str(other_vocabulary["dataset_id"]) == dataset_id:
                priority_negative_classes.update(positives[other_index])
        priority_negative_classes.difference_update(all_positive_groups)
        eligible_catalog = [
            candidate
            for candidate in catalogs[sample_index]
            if candidate["dataset_class_id"] in positive_groups
            or candidate["dataset_class_id"] not in all_positive_groups
        ]
        sample_catalog = _limit_prompt_candidates(
            eligible_catalog,
            key_name="dataset_class_id",
            positive_keys=set(positive_groups),
            priority_negative_keys=priority_negative_classes,
            max_negatives=(
                None
                if max_prompts is None
                else max(0, max_prompts - len(positive_groups))
            ),
            seed_token=f"{sampling_seed}:{batch['sample_ids'][sample_index]}:object",
        )
        for candidate in sample_catalog:
            prompt = _sample_prompt(
                candidate,
                f"{sampling_seed}:{batch['sample_ids'][sample_index]}:object:"
                f"{candidate['dataset_class_id']}:prompt",
            )
            positive_indices = positive_groups.get(candidate["dataset_class_id"], [])
            is_positive = bool(positive_indices)
            if is_positive:
                normalized_boxes = torch.cat(
                    [
                        batch["boxes"][index][selected_by_annotation[index]]
                        for index in positive_indices
                    ],
                    dim=0,
                )
                omitted_boxes = ignored_by_group[
                    (sample_index, candidate["dataset_class_id"])
                ]
            else:
                normalized_boxes = torch.empty(0, 4, dtype=torch.float32)
                omitted_boxes = torch.empty(0, 4, dtype=torch.float32)
            prompts.append(prompt)
            sample_indices.append(sample_index)
            pair_boxes.append(normalized_boxes)
            ignored_boxes.append(omitted_boxes)
            truncated_rows.append(bool(len(omitted_boxes)))

    count = len(prompts)
    expanded = _replace_prompt_rows(
        batch,
        prompts=prompts,
        sample_indices=sample_indices,
        level="object",
    )
    expanded.update(
        {
            "pixel_targets": torch.full((count, height, width), 255, dtype=torch.uint8),
            "valid_mask": batch["valid_image_masks"].index_select(
                0, torch.tensor(sample_indices, dtype=torch.long)
            ),
            "boxes": pair_boxes,
            "ignored_boxes": ignored_boxes,
            "object_truncated_rows": torch.tensor(truncated_rows, dtype=torch.bool),
            "object_presence_targets": torch.tensor(
                [bool(len(boxes)) for boxes in pair_boxes], dtype=torch.bool
            ),
            "scene_targets": torch.full((count,), float("nan"), dtype=torch.float32),
        }
    )
    return expanded
