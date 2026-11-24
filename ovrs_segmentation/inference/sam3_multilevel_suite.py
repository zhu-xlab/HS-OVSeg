"""Render reproducible validation predictions using every dataset-local class."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.distributed as dist

from ovrs_segmentation.datasets import open_text_prompt_dataset
from ovrs_segmentation.datasets.class_prompts import class_specs_by_id, combined_prompt_texts
from ovrs_segmentation.evaluation.multilevel_validation import _forward_prompt_chunks
from ovrs_segmentation.evaluation.sam3_common import close_distributed, distributed_context
from ovrs_segmentation.evaluation.sam3_multilevel import load_adapted_model
from ovrs_segmentation.evaluation.segmentation_test import load_suite
from ovrs_segmentation.visualization import render_semantic_overlay
from .common import _queryable_class, slug
from .tiling import image_tiles, minimum_size


def sample_ids(ids, count, seed, dataset_name):
    ids = sorted(ids)
    if len(set(ids)) != len(ids) or not 0 < count <= len(ids):
        raise ValueError(f"{dataset_name}: need {count} distinct validation images, got {len(ids)}")
    salt = int.from_bytes(hashlib.sha256(dataset_name.encode()).digest()[:8], "big")
    return random.Random(seed + salt).sample(ids, count)


def pixel_winners(logits, class_ids, empty_id, threshold):
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite unified pixel logits")
    best, winner = logits.float().max(0)
    raw_ids = torch.tensor(class_ids, device=logits.device, dtype=torch.int64)
    prediction = raw_ids[winner]
    if threshold is not None:
        prediction[best.sigmoid().le(threshold)] = empty_id
    return prediction.cpu().numpy().astype(np.uint16)


@torch.inference_mode()
def predict_pixels(model, image, prompts, class_ids, empty_id, device, precision, threshold, *, forward=None):
    forward = forward or _forward_prompt_chunks
    height, width = image.shape[:2]
    images = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float() / 255
    if minimum_size((height, width), model.sam3_input_size) != (height, width):
        # Resize continuous logits back to the original image before argmax.
        batch = {"images": images, "prompt_texts": prompts, "pixel_targets": torch.empty(0, height, width)}
        output = forward(model, batch, device, precision)
        return pixel_winners(output["unified_pixel_logits"], class_ids, empty_id, threshold)
    prediction = np.empty((height, width), dtype=np.uint16)
    for top, left, valid_h, valid_w, tile in image_tiles(images, model.sam3_input_size):
        batch = {"images": tile, "prompt_texts": prompts, "pixel_targets": torch.empty(0, *model.sam3_input_size)}
        output = forward(model, batch, device, precision)
        pixels = pixel_winners(output["unified_pixel_logits"], class_ids, empty_id, threshold)
        prediction[top:top + valid_h, left:left + valid_w] = pixels[:valid_h, :valid_w]
    return prediction


def save_prediction(directory, filename, image, prediction, class_names, background_id, ignore_id, title):
    mask = prediction.astype(np.uint8) if prediction.max() <= 255 else prediction
    Image.fromarray(mask).save(directory / "masks" / (filename + ".png"))
    height, width = prediction.shape
    scale = min(1600 / max(height, width), max(1.0, 768 / min(height, width)))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    preview_image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
    preview_mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
    rendered = render_semantic_overlay(preview_image, preview_mask, class_names,
                                       background_id=background_id, ignore_id=ignore_id)
    canvas = Image.fromarray(rendered)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    if not any(int(i) in class_names for i in np.unique(prediction)):
        title += " | No class above threshold"
    box = draw.textbbox((5, 5), title, font=font)
    draw.rectangle((0, 0, min(size[0], box[2] + 5), box[3] + 5), fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 255, 255), font=font)
    canvas.save(directory / "overlays" / (filename + ".jpg"), quality=92)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("multilevel", "native"), default="multilevel")
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--suite", type=Path, default=Path("configs/inference_val.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/Datasets/RSOD"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_multilevel_val50"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam3_multilevel/last.pt"))
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path("checkpoints/sam3.pt"))
    parser.add_argument("--sam3-bpe-path", type=Path)
    parser.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--sam3-input-size", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prompt-chunk-size", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.samples_per_dataset < 1 or args.prompt_chunk_size < 1:
        parser.error("Invalid threshold, sample count or prompt chunk size")
    return args


def main(*, model_loader=None):
    args = parse_args()
    rank, world, device = distributed_context(args.device)
    try:
        entries = load_suite(args.suite, args.data_root, args.data_root)
        reference = None
        if args.reference_summary:
            reference = json.loads(args.reference_summary.read_text())
            if not reference.get("complete") or reference.get("preflight"):
                raise ValueError("Reference predictions must be complete")
            reference = {(r["level"], r["dataset"]): r for r in reference["datasets"]}
            if set(reference) != {(e["source_task"], e["name"]) for e in entries}:
                raise ValueError("Reference dataset list differs from inference suite")
        output = args.output_dir.resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if (output / "summary.json").exists() and not args.preflight:
                raise FileExistsError(f"Completed or partial run exists: {output}")
        if world > 1:
            dist.barrier()
        model = None
        reports = []
        for entry in entries:
            name, level, root = entry["name"], entry["source_task"], entry["root"]
            split = entry.get("split", "val")
            if split != "val":
                raise ValueError(f"{name}: validation suite requires split=val")
            dataset = open_text_prompt_dataset(root, split, validate_masks=False, level=level)
            source = getattr(dataset, "semantic", dataset)
            metadata = source.metadata
            background_id = metadata.get("background_id")
            ignore_id = int(metadata.get("ignore_id", 255))
            classes = {i: spec for i, spec in sorted(class_specs_by_id(root / metadata.get("classes_file", "classes.json")).items())
                       if _queryable_class(i, spec, background_id, ignore_id)}
            if not classes:
                raise ValueError(f"No queryable classes in {name}")
            class_names = {i: row["class_text"] for i, row in classes.items()}
            class_ids = list(classes)
            prompts = [combined_prompt_texts(class_names[i], level, name)["native_prompt_texts"][0] for i in class_ids]
            selected = sample_ids(source.sample_ids, args.samples_per_dataset, args.seed, name)
            if reference is not None:
                ref = reference[level, name]
                selected = [row["sample_id"] for row in ref["selected"]]
                if (len(selected) != args.samples_per_dataset or len(set(selected)) != len(selected)
                        or not set(selected).issubset(source.sample_ids)
                        or ref["split"] != split
                        or ref["classes"] != {str(k): v for k, v in class_names.items()}
                        or ref["prompts"] != prompts):
                    raise ValueError(f"{name}: reference samples or class prompts differ")
            indices = {sample_id: i for i, sample_id in enumerate(source.sample_ids)}
            directory = output / level / name
            rows = []
            for position, sample_id in enumerate(selected):
                filename = f"{position + 1:02d}_{slug(sample_id)[:110]}_{hashlib.sha256(sample_id.encode()).hexdigest()[:8]}"
                rows.append({"sample_id": sample_id, "overlay": f"overlays/{filename}.jpg", "mask": f"masks/{filename}.png"})
            if reference is not None:
                if rows != [{k: row[k] for k in ("sample_id", "overlay", "mask")} for row in ref["selected"]]:
                    raise ValueError(f"{name}: reference filenames differ")
            report = {"dataset": name, "level": level, "root": str(root), "split": split,
                      "samples": len(selected), "classes": class_names, "prompts": prompts,
                      "background_id": background_id, "ignore_id": ignore_id,
                      "unassigned_id": background_id if background_id is not None else ignore_id,
                      "seed": args.seed, "selected": rows, "status": "ready"}
            if args.preflight:
                if rank == 0:
                    print(f"[preflight] {name}: {len(selected)}/{len(source.sample_ids)} val images, {len(classes)} classes", flush=True)
                reports.append(report)
                continue
            if model is None:
                if args.model_type == "native":
                    from .sam3_native_suite import load_native_model
                    model, checkpoint = load_native_model(args, device)
                else:
                    model, state, settings, checkpoint = (model_loader or load_adapted_model)(args, device)
            if rank == 0:
                for folder in ("masks", "overlays"):
                    (directory / folder).mkdir(parents=True, exist_ok=True)
                (directory / "manifest.json").write_text(json.dumps(report, indent=2))
            if world > 1:
                dist.barrier()
            local_rows = []
            for position in range(rank, len(selected), world):
                sample_id = selected[position]
                index = indices[sample_id]
                if hasattr(source, "image_only_sample"):
                    image = source._read_rgb(source._images[sample_id])
                else:
                    image = source[index]["image"]
                forward = None
                if args.model_type == "native":
                    from .sam3_native_suite import forward_native
                    forward = forward_native
                prediction = predict_pixels(model, image, prompts, class_ids, report["unassigned_id"],
                                            device, args.precision, args.threshold, forward=forward)
                filename = Path(rows[position]["mask"]).stem
                save_prediction(directory, filename, image, prediction, class_names, background_id,
                                ignore_id, f"{name} | {sample_id} | p > {args.threshold}")
                local_rows.append({**rows[position], "original_shape": list(image.shape[:2]),
                                   "predicted_class_ids": [int(i) for i in np.unique(prediction) if int(i) in class_names]})
                print(f"[infer rank={rank}] {name}: {position + 1}/{len(selected)} {sample_id}", flush=True)
            if world > 1:
                gathered = [None] * world if rank == 0 else None
                dist.gather_object(local_rows, gathered, dst=0)
            else:
                gathered = [local_rows]
            report["status"] = "predicted"
            if rank == 0:
                by_id = {row["sample_id"]: row for shard in gathered for row in shard}
                report["selected"] = [by_id[sample_id] for sample_id in selected]
                for row in rows:
                    assert (directory / row["mask"]).is_file() and (directory / row["overlay"]).is_file()
                (directory / "manifest.json").write_text(json.dumps(report, indent=2))
                print(f"[infer] {name}: complete {len(selected)} predictions", flush=True)
            reports.append(report)
        if rank == 0:
            result = {"complete": True, "preflight": args.preflight, "datasets": reports,
                      "total_images": sum(r["samples"] for r in reports), "seed": args.seed,
                      "model": args.model_type,
                      "pixel_score": "sigmoid(semantic_seg)" if args.model_type == "native" else "sigmoid(unified_pixel_logits)",
                      "decision": "argmax across all dataset classes, max probability > threshold",
                      "threshold": args.threshold, "world_size": world,
                      "checkpoint": str((args.sam3_checkpoint if args.model_type == "native" else args.checkpoint).resolve()),
                      "overlay_max_side": 1600, "mask_resolution": "original_image"}
            if model is not None:
                result.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              input_size=model.sam3_input_size if args.model_type == "native" else settings["sam3_input_size"])
                if args.model_type == "multilevel":
                    result["checkpoint_epoch"] = int(state["epoch"]) + 1
                else:
                    result["adapter_or_finetuned_weights_loaded"] = False
            if args.reference_summary:
                result["reference_summary"] = str(args.reference_summary)
            (output / ("preflight.json" if args.preflight else "summary.json")).write_text(json.dumps(result, indent=2))
            print(f"[infer] complete: {result['total_images']} images in {len(reports)} datasets", flush=True)
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
