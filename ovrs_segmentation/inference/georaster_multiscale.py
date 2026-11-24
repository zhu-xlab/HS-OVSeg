"""Infer every class at four ground resolutions and fuse probabilities at 1 metre."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .georaster import activation_filename, probability_uint8, write_probability_statistics


RESOLUTIONS = (.5, 1., 2., 5.)


def scale_name(resolution):
    return f"{resolution:g}m"


def resample_rgb(source_path, destination, resolution):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as source:
        width = round((source.bounds.right - source.bounds.left) / resolution)
        height = round((source.bounds.top - source.bounds.bottom) / resolution)
        transform = from_origin(source.bounds.left, source.bounds.top, resolution, resolution)
        if not np.allclose([width * resolution, height * resolution],
                           [source.bounds.right - source.bounds.left, source.bounds.top - source.bounds.bottom]):
            raise ValueError("Source bounds do not fit the requested ground resolution")
        if destination.exists():
            with rasterio.open(destination) as existing:
                if existing.shape != (height, width) or existing.transform != transform or existing.crs != source.crs:
                    raise ValueError(f"Existing image has a different grid: {destination}")
            return destination
        profile = {**source.profile, "driver": "GTiff", "width": width, "height": height,
                   "transform": transform, "count": 3, "nodata": None, "tiled": True,
                   "blockxsize": 512, "blockysize": 512, "compress": "deflate", "predictor": 2,
                   "BIGTIFF": "IF_SAFER"}
        temporary = destination.with_suffix(".partial.tif")
        with WarpedVRT(source, crs=source.crs, transform=transform, width=width, height=height,
                       resampling=Resampling.average) as resized, rasterio.open(temporary, "w", **profile) as target:
            for _, window in target.block_windows(1):
                target.write(resized.read([1, 2, 3], window=window), window=window)
            target.update_tags(source_image=str(Path(source_path).resolve()), resolution_m=resolution,
                               resampling="area_average", source_product=source.tags().get("source_product", "NAIP"))
        temporary.replace(destination)
    print(f"[multiscale] image prepared: {destination}", flush=True)
    return destination


def probability_vrt(directory, summary, band, destination):
    """Expose only each rank's owned rows, preserving unquantized probabilities."""
    import rasterio
    paths = [Path(directory) / name for name in summary["probability_shards"]]
    width, height = summary["width"], summary["height"]
    with rasterio.open(paths[0]) as first:
        transform, crs = first.transform, first.crs
    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, "SRS").text = crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = ",".join(map(str, transform.to_gdal()))
    raster_band = ET.SubElement(root, "VRTRasterBand", dataType="Float32", band="1")
    for rank, path in enumerate(paths):
        top, bottom = height * rank // len(paths), height * (rank + 1) // len(paths)
        with rasterio.open(path) as shard:
            if (shard.shape != (height, width) or shard.transform != transform or shard.crs != crs
                    or shard.dtypes[band - 1] != "float32"):
                raise ValueError(f"Invalid probability stripe: {path}")
        source = ET.SubElement(raster_band, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="0").text = str(path.resolve())
        ET.SubElement(source, "SourceBand").text = str(band)
        rectangle = dict(xOff="0", yOff=str(top), xSize=str(width), ySize=str(bottom - top))
        ET.SubElement(source, "SrcRect", **rectangle)
        ET.SubElement(source, "DstRect", **rectangle)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return Path(destination)


def fuse_probabilities(directories, reference_image, output):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    output = Path(output)
    if (output / "summary.json").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((Path(p) / "summary.json").read_text()) for p in directories]
    first = summaries[0]
    for summary in summaries:
        if (not summary["complete"] or summary.get("probability_dtype") != "float32"
                or summary["classes"] != first["classes"] or summary["prompts"] != first["prompts"]
                or summary["checkpoint_sha256"] != first["checkpoint_sha256"]
                or summary.get("model_type", "multilevel") != first.get("model_type", "multilevel")
                or summary["value_semantics"] != "presence_weighted_probability"):
            raise ValueError("Fusion requires complete, matching class-wise float32 probabilities")
    files, statistics = {}, {}
    with rasterio.Env(GDAL_CACHEMAX=1024**3), rasterio.open(reference_image) as reference, \
            tempfile.TemporaryDirectory(prefix=".fusion_", dir=output) as temporary:
        if reference.res != (1, 1):
            raise ValueError("Fusion reference must have 1 metre pixels")
        profile = {**reference.profile, "count": 1, "dtype": "uint8", "nodata": None,
                   "compress": "deflate", "zlevel": 9, "predictor": 2, "num_threads": 4,
                   "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"}
        def fuse_class(item):
            band, (class_id, name) = item
            filename = activation_filename(name)
            destination = output / filename
            partial = destination.with_suffix(".partial.tif")
            histogram = np.zeros(256, np.int64)
            with ExitStack() as stack:
                readers = []
                methods = []
                for index, (directory, summary) in enumerate(zip(directories, summaries)):
                    path = probability_vrt(directory, summary, band, Path(temporary) / f"{band}_{index}.vrt")
                    source = stack.enter_context(rasterio.open(path))
                    if source.crs != reference.crs or not np.allclose(source.bounds, reference.bounds):
                        raise ValueError("All probability scales must cover the same geographic bounds")
                    method = Resampling.average if source.res[0] < 1 else Resampling.bilinear
                    methods.append(method.name)
                    readers.append(stack.enter_context(WarpedVRT(
                        source, crs=reference.crs, transform=reference.transform,
                        width=reference.width, height=reference.height, resampling=method,
                    )))
                target = stack.enter_context(rasterio.open(partial, "w", **profile))
                for _, window in target.block_windows(1):
                    fused = np.zeros((int(window.height), int(window.width)), np.float32)
                    for reader in readers:
                        values = reader.read(1, window=window)
                        if not np.isfinite(values).all() or np.any((values < -1e-6) | (values > 1 + 1e-6)):
                            raise ValueError("Invalid floating point probabilities in fusion")
                        np.maximum(fused, values, out=fused)
                    quantized = probability_uint8(np.clip(fused, 0, 1))
                    target.write(quantized, 1, window=window)
                    histogram += np.bincount(quantized.ravel(), minlength=256)
                levels = np.flatnonzero(histogram)
                target.set_band_description(1, name)
                target.update_tags(class_id=class_id, class_name=name, value_semantics="presence_weighted_probability",
                                   encoding="uint8(255 * max_scale(resample_1m(P_scale)))", probability_divisor=255)
                write_probability_statistics(target, histogram)
            partial.replace(destination)
            stats = dict(min=int(levels[0]), max=int(levels[-1]), nonzero_pixels=int(histogram[1:].sum()))
            print(f"[multiscale] fused {band}/{len(first['classes'])}: {filename}", flush=True)
            return class_id, filename, stats, methods

        with ThreadPoolExecutor(max_workers=4) as executor:
            for class_id, filename, stats, methods in executor.map(fuse_class, enumerate(first["classes"].items(), 1)):
                files[class_id] = filename
                statistics[class_id] = stats
        report = dict(complete=True, classes=first["classes"], prompts=first["prompts"], files=files,
                      model_type=first.get("model_type", "multilevel"),
                      adapter_or_finetuned_weights_loaded=first.get("adapter_or_finetuned_weights_loaded", True),
                      checkpoint_sha256=first["checkpoint_sha256"], checkpoint_epoch=first["checkpoint_epoch"],
                      width=reference.width, height=reference.height, crs=str(reference.crs),
                      transform=list(reference.transform), pixel_size=[1, 1], dtype="uint8", nodata=None,
                      encoding="uint8(255 * max_scale(resample_1m(P_scale)))", probability_divisor=255,
                      value_semantics="presence_weighted_probability", fusion="per_class_max_before_quantization",
                      source_directories=[str(Path(p).resolve()) for p in directories],
                      source_resolutions=[s["pixel_size"] for s in summaries], resampling=methods, statistics=statistics)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("multilevel", "native"), default="multilevel")
    parser.add_argument("--image", type=Path, required=True, help="0.5 metre RGB source for all four scales")
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam3_multilevel/last.pt"))
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path("checkpoints/sam3.pt"))
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--prompt-chunk-size", type=int, default=45)
    parser.add_argument("--auto-gpus", action="store_true", help="Select GPUs with at least 16 GiB free before each scale")
    args = parser.parse_args()
    import rasterio
    with rasterio.open(args.image) as source:
        if source.res != (.5, .5):
            parser.error("Input must use a 0.5 metre grid")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pinned = output / "checkpoint_snapshot.pt"
    checkpoint_source = args.sam3_checkpoint if args.model_type == "native" else args.checkpoint
    if pinned.exists():
        if hashlib.sha256(pinned.read_bytes()).digest() != hashlib.sha256(checkpoint_source.read_bytes()).digest():
            raise ValueError("Existing multiscale run uses a different checkpoint")
    else:
        shutil.copy2(checkpoint_source, pinned)
    checkpoint_hash = hashlib.sha256(pinned.read_bytes()).hexdigest()
    images, directories = {}, {}
    # Finish the smaller scales first while waiting for GPUs to become available.
    for resolution in reversed(RESOLUTIONS):
        name = scale_name(resolution)
        image = args.image if resolution == .5 else resample_rgb(
            args.image, args.image.parent.parent / name / "image.tif", resolution)
        images[name] = image
        directory = output / name
        directories[resolution] = directory
        if (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            if (not summary["complete"] or summary["checkpoint_sha256"] != checkpoint_hash
                    or summary.get("model_type", "multilevel") != args.model_type
                    or summary.get("image_batch_size", 1) != args.image_batch_size
                    or summary.get("prompt_chunk_size", 8) != args.prompt_chunk_size):
                raise ValueError(f"Existing scale is incomplete or uses different checkpoint/batch settings: {directory}")
            continue
        environment = os.environ.copy()
        processes = args.nproc_per_node
        if args.auto_gpus:
            rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
            devices = [row.split(",")[0].strip() for row in rows.splitlines() if int(row.split(",")[1]) >= 16384][:processes]
            if not devices:
                raise RuntimeError("No GPU has 16 GiB free for the next scale")
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            processes = len(devices)
            print(f"[multiscale] using GPU(s) {environment['CUDA_VISIBLE_DEVICES']}", flush=True)
        command = [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
                   "--master-port=29500", "--nnodes=1", f"--nproc-per-node={processes}",
                   "-m", "ovrs_segmentation.inference.georaster", "--image", str(image),
                   "--model-type", args.model_type,
                   "--classes-json", str(args.classes_json), "--output-dir", str(directory),
                   "--checkpoint", str(pinned), "--sam3-checkpoint", str(pinned if args.model_type == "native" else args.sam3_checkpoint),
                   "--device", "cuda", "--precision", "bf16", "--prompt-chunk-size", str(args.prompt_chunk_size),
                   "--image-batch-size", str(args.image_batch_size),
                   "--stride", "259", "--activations-only", "--save-probabilities"]
        print(f"[multiscale] starting all classes at {resolution:g}m", flush=True)
        subprocess.run(command, check=True, env=environment)
    ordered_directories = [directories[r] for r in RESOLUTIONS]
    report = fuse_probabilities(ordered_directories, images["1m"], output / "max_fusion_1m")
    (output / "summary.json").write_text(json.dumps(dict(
        complete=True, resolutions=list(RESOLUTIONS), checkpoint_sha256=checkpoint_hash,
        model_type=args.model_type, adapter_or_finetuned_weights_loaded=args.model_type != "native",
        per_scale_classes=len(report["classes"]), scale_directories=[p.name for p in ordered_directories],
        image_batch_size=args.image_batch_size, prompt_chunk_size=args.prompt_chunk_size,
        fused_directory="max_fusion_1m", fusion=report["fusion"],
    ), indent=2))


if __name__ == "__main__":
    main()
