"""Stream a georeferenced RGB raster through SAM3 using a complete class catalog."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn import functional as F

from ovrs_segmentation.datasets.class_prompts import class_specs_by_id, combined_prompt_texts
from ovrs_segmentation.evaluation.multilevel_validation import _forward_prompt_chunks
from ovrs_segmentation.evaluation.sam3_common import close_distributed, distributed_context
from ovrs_segmentation.evaluation.sam3_multilevel import load_adapted_model
from ovrs_segmentation.visualization import colorize_mask, palette_for_classes, render_semantic_overlay
from .common import _queryable_class

PROBABILITY_ENCODING = "uint8(255 * hann_weighted_mean(sigmoid(unified_pixel_logits) * sigmoid(concept_logits)))"
NATIVE_PROBABILITY_ENCODING = "uint8(255 * hann_weighted_mean(sigmoid(semantic_seg) * sigmoid(presence_logit_dec)))"


def raster_windows(height, width, tile_size, stride=None):
    from rasterio.windows import Window
    stride = tile_size if stride is None else stride
    def starts(length, size, step):
        if not 0 < step <= size:
            raise ValueError("Stride must be positive and no larger than the window")
        if step == size:
            return range(0, length, size)
        last = max(0, length - size)
        positions = list(range(0, last + 1, step))
        if positions[-1] != last:
            positions.append(last)
        return positions
    return [Window(left, top, min(tile_size[1], width - left), min(tile_size[0], height - top))
            for top in starts(height, tile_size[0], stride[0])
            for left in starts(width, tile_size[1], stride[1])]


def blend_weights(tile_size):
    return np.maximum(np.outer(np.hanning(tile_size[0]), np.hanning(tile_size[1])), 1e-3).astype(np.float32)


def batched_window_predictor(windows, batch_size, predict_batch):
    from itertools import islice
    if batch_size < 1:
        raise ValueError("Image batch size must be positive")
    pending, buffered = iter(windows), iter(())
    def predict(window):
        nonlocal buffered
        try:
            expected, values = next(buffered)
        except StopIteration:
            batch = list(islice(pending, batch_size))
            predictions = predict_batch(batch)
            if len(predictions) != len(batch):
                raise ValueError("Each window needs its own prediction")
            buffered = iter(zip(batch, predictions))
            expected, values = next(buffered)
        if expected != window:
            raise ValueError("Batched windows must follow the blending order")
        return values
    return predict


@torch.inference_mode()
def predict_window_batch(model, source, windows, prompts, device, precision, *, probabilities, forward=None):
    images, sizes = [], []
    for window in windows:
        image = source.read([1, 2, 3], window=window)
        height, width = image.shape[-2:]
        sizes.append((height, width))
        tensor = torch.from_numpy(image.copy()).float() / 255
        images.append(F.pad(tensor, (0, model.sam3_input_size[1] - width,
                                    0, model.sam3_input_size[0] - height), value=.5))
    count, classes = len(images), len(prompts)
    batch = {"images": torch.stack(images), "prompt_texts": [p for p in prompts for _ in images],
             "prompt_image_indices": torch.arange(count).repeat(classes),
             "prompt_chunk_size": model.prompt_chunk_size * count,
             "pixel_targets": torch.empty(0, *model.sam3_input_size)}
    output = (forward or _forward_prompt_chunks)(model, batch, device, precision)
    logits = output["unified_pixel_logits"]
    if probabilities:
        logits = presence_weighted_probabilities(logits, output["concept_logits"])
    if not torch.isfinite(logits).all():
        raise ValueError("Model returned non-finite pixel predictions")
    values = logits.reshape(classes, count, *model.sam3_input_size).permute(1, 0, 2, 3).float().cpu().numpy()
    return [value[:, :h, :w] for value, (h, w) in zip(values, sizes)]


def blended_stripe(windows, top, bottom, width, tile_size, class_ids, predict, *, return_logits=False):
    """Yield finalized pixel rows, retaining only one window-height of float32 logits."""
    from itertools import groupby
    from rasterio.windows import Window
    if top == bottom:
        return
    selected = [w for w in windows if w.row_off < bottom and w.row_off + w.height > top]
    rows = [(int(y), list(group)) for y, group in groupby(selected, key=lambda w: w.row_off)]
    scores = np.zeros((len(class_ids), tile_size[0], width), dtype=np.float32)
    normalizer = np.zeros((tile_size[0], width), dtype=np.float32)
    weights = blend_weights(tile_size)
    ids = np.asarray(class_ids, dtype=np.uint8)
    cursor = top
    for index, (y, row) in enumerate(rows):
        for window in row:
            logits = predict(window)
            left, height, columns = int(window.col_off), int(window.height), int(window.width)
            first, last = max(y, cursor), min(y + height, bottom)
            src = slice(first - y, last - y)
            dst = slice(first - cursor, last - cursor)
            weight = weights[src, :columns]
            scores[:, dst, left:left + columns] += logits[:, src, :columns] * weight
            normalizer[dst, left:left + columns] += weight
        end = min(bottom, rows[index + 1][0] if index + 1 < len(rows) else bottom)
        ready = end - cursor
        if ready <= 0:
            continue
        assert ready <= tile_size[0] and np.all(normalizer[:ready] > 0), "Uncovered inference pixels"
        scores[:, :ready] /= normalizer[None, :ready]
        values = scores[:, :ready] if return_logits else ids[scores[:, :ready].argmax(axis=0)]
        yield Window(0, cursor, width, ready), values
        remaining = tile_size[0] - ready
        scores[:, :remaining] = scores[:, ready:]
        normalizer[:remaining] = normalizer[ready:]
        scores[:, remaining:] = 0
        normalizer[remaining:] = 0
        cursor = end
    assert cursor == bottom, "Inference stripe was not completely finalized"


def class_colors(names):
    colors = colorize_mask(np.asarray([list(names)]), palette_for_classes(names))[0]
    return {i: (*map(int, color), 255) for i, color in zip(names, colors)}


def presence_weighted_probabilities(pixel_logits, presence_logits):
    """Gate each class's conditional pixel probability with its tile Presence."""
    presence = presence_logits.float().reshape(-1)
    if pixel_logits.ndim != 3 or pixel_logits.shape[0] != len(presence):
        raise ValueError("Pixel and Presence class dimensions must agree")
    if not torch.isfinite(pixel_logits).all() or not torch.isfinite(presence).all():
        raise ValueError("Pixel and Presence logits must be finite")
    return pixel_logits.float().sigmoid() * presence.sigmoid()[:, None, None]


def probability_uint8(probabilities):
    values = np.asarray(probabilities, dtype=np.float32)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return (values * 255).astype(np.uint8)


def activation_filename(name):
    if not name or name in {".", ".."} or name[-1] in ". " or any(c in name for c in '/\\<>:"|?*\0'):
        raise ValueError(f"Class name is not a safe TIFF filename: {name!r}")
    return name + ".tif"


def write_probability_statistics(target, histogram):
    levels = np.flatnonzero(histogram)
    mean = float(np.dot(histogram, np.arange(256, dtype=np.float64)) / histogram.sum())
    variance = float(np.dot(histogram, (np.arange(256) - mean) ** 2) / histogram.sum())
    target.update_tags(1, STATISTICS_MINIMUM=int(levels[0]), STATISTICS_MAXIMUM=int(levels[-1]),
                       STATISTICS_MEAN=mean, STATISTICS_STDDEV=variance**.5,
                       STATISTICS_VALID_PERCENT=100, STATISTICS_APPROXIMATE="NO")


def export_activations(output, partials, profile, names, world, *, resume=False, float_probabilities=False,
                       probability_encoding=PROBABILITY_ENCODING):
    """Merge uint8 probability stripes into one losslessly compressed GeoTIFF per class."""
    from contextlib import ExitStack
    from concurrent.futures import ThreadPoolExecutor
    import rasterio
    from rasterio.windows import Window
    filenames = [activation_filename(name) for name in names.values()]
    if len(set(name.casefold() for name in filenames)) != len(filenames):
        raise ValueError("Class names must produce unique TIFF filenames")
    class_profile = {**profile, "count": 1, "nodata": None, "SPARSE_OK": False,
                     "interleave": "band", "compress": "deflate", "zlevel": 9, "num_threads": 4}
    def export_one(item):
        band, ((class_id, name), filename) = item
        path = output / filename
        if resume and path.exists():
            with rasterio.open(path) as existing:
                assert existing.shape == (profile["height"], profile["width"]) and existing.dtypes == ("uint8",)
                assert int(existing.tags()["class_id"]) == class_id
                if (existing.tags().get("encoding") != probability_encoding or
                        existing.tags().get("value_semantics") != "presence_weighted_probability"):
                    raise ValueError(f"Existing TIFF has incompatible probability encoding: {path}")
            print(f"[activations] retained {band}/{len(names)}: {filename}", flush=True)
            return
        with ExitStack() as stack:
            shards = [stack.enter_context(rasterio.open(partials / f"rank_{r:02d}.tif")) for r in range(world)]
            temporary = path.with_suffix(".partial.tif")
            with rasterio.open(temporary, "w", **class_profile) as target:
                histogram = np.zeros(256, np.int64)
                for rank, shard in enumerate(shards):
                    top, bottom = profile["height"] * rank // world, profile["height"] * (rank + 1) // world
                    for row in range(top, bottom, 512):
                        window = Window(0, row, profile["width"], min(512, bottom - row))
                        values = shard.read(band, window=window)
                        if float_probabilities:
                            values = probability_uint8(values)
                        target.write(values, 1, window=window)
                        histogram += np.bincount(values.ravel(), minlength=256)
                target.set_band_description(1, name)
                write_probability_statistics(target, histogram)
                target.update_tags(class_id=class_id, class_name=name,
                                   blending="hann_weighted_mean_probabilities",
                                   encoding=probability_encoding,
                                   probability_divisor=255, value_semantics="presence_weighted_probability")
            temporary.replace(path)
            print(f"[activations] exported {band}/{len(names)}: {filename}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(export_one, enumerate(zip(names.items(), filenames), 1)))
    return {str(i): filename for i, filename in zip(names, filenames)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("multilevel", "native"), default="multilevel")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--dataset-name", default="NWPU-RESISC45")
    parser.add_argument("--level", choices=("scene", "object", "lulc"), default="scene")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam3_multilevel/last.pt"))
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path("checkpoints/sam3.pt"))
    parser.add_argument("--sam3-bpe-path", type=Path)
    parser.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--sam3-input-size", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--prompt-chunk-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--stride", type=int, help="Window stride in pixels; defaults to half the model input size")
    parser.add_argument("--activations-only", action="store_true", help="Export per-class uint8 Presence-weighted probability TIFFs")
    parser.add_argument("--save-probabilities", action="store_true",
                        help="Retain float32 probability stripes for fusion before uint8 quantization")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.image_batch_size < 1 or args.prompt_chunk_size < 1:
        parser.error("Image and prompt batch sizes must be positive")
    if args.save_probabilities and not args.activations_only:
        parser.error("--save-probabilities requires --activations-only")
    return args


@torch.inference_mode()
def main():
    import rasterio
    from rasterio.enums import Resampling
    args = parse_args()
    # Large raster stripes can finish far apart while flushing compressed TIFF blocks.
    rank, world, device = distributed_context(args.device, timeout=timedelta(hours=2))
    try:
        output = args.output_dir.resolve()
        if (output / "prediction.tif").exists() or (output / "summary.json").exists():
            raise FileExistsError(f"Inference output already exists: {output}")
        partials = output / "partial"
        partials.mkdir(parents=True, exist_ok=True)
        meta_path = args.classes_json.parent / "dataset.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        classes = {i: spec for i, spec in sorted(class_specs_by_id(args.classes_json).items())
                   if _queryable_class(i, spec, meta.get("background_id"), int(meta.get("ignore_id", 255)))}
        class_ids = list(classes)
        if not class_ids or min(class_ids) < 0 or max(class_ids) >= 255:
            raise ValueError("Georaster export requires queryable class IDs in [0, 254]")
        names = {i: spec["class_text"] for i, spec in classes.items()}
        if args.activations_only:
            for name in names.values():
                if (output / activation_filename(name)).exists():
                    raise FileExistsError(output / activation_filename(name))
        prompts = [combined_prompt_texts(names[i], args.level, args.dataset_name)["native_prompt_texts"][0] for i in class_ids]
        forward = None
        if args.model_type == "native":
            from .sam3_native_suite import load_native_model, forward_native_tile
            model, checkpoint = load_native_model(args, device)
            state, settings = {}, {"sam3_input_size": model.sam3_input_size[0]}
            forward = forward_native_tile
        else:
            model, state, settings, checkpoint = load_adapted_model(args, device)
        model_fields = dict(model_type=args.model_type,
                            adapter_or_finetuned_weights_loaded=args.model_type != "native",
                            checkpoint_epoch=None if args.model_type == "native" else int(state["epoch"]) + 1,
                            input_size=settings["sam3_input_size"])
        probability_encoding = NATIVE_PROBABILITY_ENCODING if args.model_type == "native" else PROBABILITY_ENCODING
        tile_size = tuple(model.sam3_input_size)
        counts = torch.zeros(256, dtype=torch.int64, device=device)
        with rasterio.Env(GDAL_CACHEMAX=(4 if args.save_probabilities else 1) * 1024**3), rasterio.open(args.image) as source:
            if source.count < 3 or source.dtypes[0] != "uint8" or source.crs is None:
                raise ValueError("Input must be a georeferenced uint8 RGB raster")
            profile = dict(driver="GTiff", width=source.width, height=source.height, count=1,
                           dtype="uint8", crs=source.crs, transform=source.transform, nodata=255,
                           tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2,
                           BIGTIFF="IF_SAFER", SPARSE_OK=True)
            stride = (args.stride, args.stride) if args.stride is not None else tuple(max(1, n // 2) for n in model.sam3_input_size)
            windows = raster_windows(source.height, source.width, model.sam3_input_size, stride)
            top, bottom = source.height * rank // world, source.height * (rank + 1) // world
            processed = 0
            selected = [w for w in windows if w.row_off < bottom and w.row_off + w.height > top]
            expected = len(selected)
            def predict_batch(batch_windows):
                nonlocal processed
                values = predict_window_batch(model, source, batch_windows, prompts, device, args.precision,
                                              probabilities=args.activations_only, forward=forward)
                first = processed == 0
                processed += len(batch_windows)
                if first or processed % 20 < len(batch_windows) or processed == expected:
                    print(f"[raster rank={rank}] window {processed}/{expected}", flush=True)
                return values
            predict = batched_window_predictor(selected, args.image_batch_size, predict_batch)
            partial_profile = {**profile, "count": len(class_ids), "nodata": None, "interleave": "band"} if args.activations_only else profile
            if args.save_probabilities:
                partial_profile = {**partial_profile, "dtype": "float32", "predictor": 3}
            with rasterio.open(partials / f"rank_{rank:02d}.tif", "w", **partial_profile) as target:
                for window, prediction in blended_stripe(windows, top, bottom, source.width,
                                                        model.sam3_input_size, class_ids, predict,
                                                        return_logits=args.activations_only):
                    if args.activations_only:
                        # Weighted float32 means can exceed probability bounds by rounding error.
                        np.clip(prediction, 0, 1, out=prediction)
                        for band, logits in enumerate(prediction, 1):
                            target.write(logits if args.save_probabilities else probability_uint8(logits), band, window=window)
                        counts[0] += int(window.width * window.height)
                    else:
                        target.write(prediction, 1, window=window)
                        counts += torch.as_tensor(np.bincount(prediction.ravel(), minlength=256), device=device)
            if world > 1:
                dist.all_reduce(counts)
                dist.barrier()
            if args.activations_only:
                close_distributed()
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if rank != 0:
                    return
            if rank == 0:
                pixel_counts = counts.cpu().tolist()
                assert sum(pixel_counts) == source.width * source.height and pixel_counts[255] == 0
                if args.activations_only:
                    files = export_activations(output, partials, profile, names, world,
                                               float_probabilities=args.save_probabilities, probability_encoding=probability_encoding)
                    report = dict(complete=True, output_type="per_class_uint8_probabilities",
                                  input=str(args.image.resolve()), width=source.width, height=source.height,
                                  crs=str(source.crs), transform=list(source.transform), pixel_size=list(source.res),
                                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  **model_fields,
                                  dataset=args.dataset_name, classes=names, prompts=prompts, threshold=None,
                                  encoding=probability_encoding, probability_divisor=255,
                                  value_semantics="presence_weighted_probability",
                                  dtype="uint8", nodata=None, compression="DEFLATE", compression_level=9,
                                  stride=list(stride), blending="hann_weighted_mean_probabilities",
                                  overlap_fraction=[1 - s / n for s, n in zip(stride, tile_size)],
                                  tiles=len(windows), world_size=world, files=files,
                                  image_batch_size=args.image_batch_size, prompt_chunk_size=args.prompt_chunk_size,
                                  covered_pixels=sum(pixel_counts), pixels_per_class=sum(pixel_counts), unassigned_pixels=0)
                    if args.save_probabilities:
                        report["probability_shards"] = [str(p.relative_to(output)) for p in
                                                        [partials / f"rank_{r:02d}.tif" for r in range(world)]]
                        report["probability_dtype"] = "float32"
                    (output / "summary.json").write_text(json.dumps(report, indent=2))
                    if not args.save_probabilities:
                        shutil.rmtree(partials)
                    print(f"[activations] complete: {len(names)} classes, {source.width}x{source.height} each", flush=True)
                    return
                color_table = class_colors(names)
                lookup = np.zeros((256, 3), dtype=np.uint8)
                for i, color in color_table.items():
                    lookup[i] = color[:3]
                color_profile = {**profile, "count": 3, "nodata": None, "SPARSE_OK": False, "photometric": "RGB"}
                with rasterio.open(output / "prediction.tif", "w", **profile) as result, \
                        rasterio.open(output / "overlay.tif", "w", **color_profile) as overlay:
                    shards = [rasterio.open(partials / f"rank_{r:02d}.tif") for r in range(world)]
                    try:
                        for shard_rank, shard in enumerate(shards):
                            start, end = source.height * shard_rank // world, source.height * (shard_rank + 1) // world
                            for window in raster_windows(end - start, source.width, model.sam3_input_size):
                                window = rasterio.windows.Window(window.col_off, window.row_off + start, window.width, window.height)
                                pixels = shard.read(1, window=window)
                                assert set(np.unique(pixels)) <= set(class_ids)
                                result.write(pixels, 1, window=window)
                                rgb = source.read([1, 2, 3], window=window).transpose(1, 2, 0)
                                fused = (rgb * .45 + lookup[pixels] * .55).clip(0, 255).astype(np.uint8)
                                overlay.write(fused.transpose(2, 0, 1), window=window)
                    finally:
                        for shard in shards:
                            shard.close()
                    result.write_colormap(1, color_table)
                    result.update_tags(class_names=json.dumps(names), decision="argmax_all_classes_no_threshold")
                    result.update_tags(stride=json.dumps(stride), blending="hann_weighted_mean_logits")
                size = (max(1, round(source.height * min(1., 3000 / max(source.height, source.width)))),
                        max(1, round(source.width * min(1., 3000 / max(source.height, source.width)))))
                preview = source.read([1, 2, 3], out_shape=(3, *size), resampling=Resampling.average).transpose(1, 2, 0)
                with rasterio.open(output / "prediction.tif") as result:
                    labels = result.read(1, out_shape=size, resampling=Resampling.nearest)
                Image.fromarray(preview).save(output / "input_preview.jpg", quality=94)
                Image.fromarray(lookup[labels]).save(output / "prediction_preview.png")
                rendered = render_semantic_overlay(preview, labels, names, background_id=None)
                Image.fromarray(rendered).save(output / "overlay_preview.jpg", quality=94)
                report = dict(complete=True, input=str(args.image.resolve()), width=source.width, height=source.height,
                              crs=str(source.crs), transform=list(source.transform), pixel_size=list(source.res),
                              checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              **model_fields,
                              dataset=args.dataset_name, classes=names, prompts=prompts, threshold=None,
                              pixel_rule="argmax(weighted_mean(unified_pixel_logits))", tiles=len(windows), world_size=world,
                              stride=list(stride), blending="hann_weighted_mean_logits",
                              overlap_fraction=[1 - s / n for s, n in zip(stride, model.sam3_input_size)],
                              class_pixel_counts={str(i): pixel_counts[i] for i in class_ids},
                              assigned_pixels=sum(pixel_counts), unassigned_pixels=0)
                (output / "summary.json").write_text(json.dumps(report, indent=2))
                shutil.rmtree(partials)
                print(f"[raster] complete: {source.width}x{source.height}, {len(classes)} classes, every pixel assigned", flush=True)
            if world > 1:
                dist.barrier()
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
