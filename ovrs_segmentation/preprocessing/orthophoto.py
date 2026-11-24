"""Download an explicitly projected, metre-resolution RGB orthophoto mosaic."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen


NAIP_2014 = "https://gis.wildlife.ca.gov/images/rest/services/Base_Remote_Sensing/NAIP_2014/ImageServer"


def export_tiles(bounds, resolution, tile_pixels):
    left, bottom, right, top = bounds
    if resolution <= 0 or tile_pixels < 1 or right <= left or top <= bottom:
        raise ValueError("Invalid image bounds, resolution or tile size")
    width, height = (right - left) / resolution, (top - bottom) / resolution
    if not math.isclose(width, round(width), abs_tol=1e-6) or not math.isclose(height, round(height), abs_tol=1e-6):
        raise ValueError("Bounds must describe an integer number of pixels")
    width, height = round(width), round(height)
    return [dict(row=row, col=col, width=min(tile_pixels, width - col), height=min(tile_pixels, height - row),
                 bounds=[left + col * resolution, top - min(row + tile_pixels, height) * resolution,
                         left + min(col + tile_pixels, width) * resolution, top - row * resolution],
                 filename=f"r{row:05d}_c{col:05d}.tif")
            for row in range(0, height, tile_pixels) for col in range(0, width, tile_pixels)]


def download_orthophoto(service_url, bounds, epsg, resolution, output_dir, tile_pixels=4000, workers=3):
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "image.tif"
    if destination.exists():
        raise FileExistsError(destination)
    crs = CRS.from_epsg(epsg)
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Use a projected CRS whose coordinate unit is metre")
    tiles = export_tiles(bounds, resolution, tile_pixels)
    tile_dir = output_dir / "tiles"
    tile_dir.mkdir(exist_ok=True)
    metadata_url = service_url.rstrip("/") + "?f=json"
    with urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    if "error" in metadata:
        raise ValueError(metadata["error"])
    product = metadata.get("name", service_url.rstrip("/").split("/")[-2]).split("/")[-1].replace("_", " ") + " RGB orthophoto"
    (output_dir / "service_metadata.json").write_text(json.dumps(metadata, indent=2))

    def validate(path, tile):
        with rasterio.open(path) as source:
            expected = from_origin(tile["bounds"][0], tile["bounds"][3], resolution, resolution)
            if source.crs != crs or not source.transform.almost_equals(expected, precision=1e-6):
                raise ValueError(f"Unexpected export georeferencing: {path}")
            if (source.width, source.height) != (tile["width"], tile["height"]) or source.count < 3:
                raise ValueError(f"Unexpected export dimensions: {path}")
            if not np.all(source.dataset_mask() == 255):
                raise ValueError(f"Missing imagery in {path}; choose a fully covered region")

    def download(tile):
        path = tile_dir / tile["filename"]
        params = dict(f="json", bbox=",".join(map(str, tile["bounds"])), bboxSR=epsg, imageSR=epsg,
                      size=f"{tile['width']},{tile['height']}", format="tiff", pixelType="U8",
                      bandIds="0,1,2", interpolation="RSP_BilinearInterpolation", compression="LZ77")
        url = service_url.rstrip("/") + "/exportImage?" + urlencode(params)
        if not path.exists():
            temporary = path.with_suffix(".part")
            for attempt in range(3):
                try:
                    print(f"[download] requesting {tile['filename']}", flush=True)
                    with urlopen(url, timeout=180) as response:
                        export = json.load(response)
                    if "error" in export:
                        raise ValueError(export["error"])
                    with urlopen(export["href"], timeout=180) as response, temporary.open("wb") as target:
                        content_type = response.headers.get("Content-Type", "")
                        if "json" in content_type or "text/" in content_type:
                            raise ValueError(response.read(2000).decode(errors="replace"))
                        while block := response.read(1024 * 1024):
                            target.write(block)
                    validate(temporary, tile)
                    temporary.replace(path)
                    break
                except Exception as error:
                    temporary.unlink(missing_ok=True)
                    print(f"[download] attempt {attempt + 1}/3 {tile['filename']}: {type(error).__name__}: {error}", flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        validate(path, tile)
        return {**tile, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}

    downloaded = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, tile) for tile in tiles]
        for future in as_completed(futures):
            tile = future.result()
            downloaded.append(tile)
            print(f"[download] {len(downloaded)}/{len(tiles)} {tile['filename']} verified", flush=True)
    width = round((bounds[2] - bounds[0]) / resolution)
    height = round((bounds[3] - bounds[1]) / resolution)
    profile = dict(driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs,
                   transform=from_origin(bounds[0], bounds[3], resolution, resolution), tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    temporary_mosaic = output_dir / "image.partial.tif"
    with rasterio.open(temporary_mosaic, "w", **profile) as target:
        for tile in downloaded:
            with rasterio.open(tile_dir / tile["filename"]) as source:
                target.write(source.read([1, 2, 3]), window=Window(tile["col"], tile["row"], tile["width"], tile["height"]))
        target.update_tags(source=service_url, source_product=product,
                           attribution="USDA NAIP; California Department of Fish and Wildlife")
    temporary_mosaic.replace(destination)
    provenance = dict(source=product, service_url=service_url, metadata_url=metadata_url,
                      source_service_pixel_size=[metadata.get("pixelSizeX"), metadata.get("pixelSizeY")],
                      attribution="USDA NAIP; California Department of Fish and Wildlife",
                      downloaded_utc=datetime.now(timezone.utc).isoformat(), crs=str(crs), bounds=list(bounds),
                      resolution_m=resolution, width=width, height=height, covered_pixels=width * height,
                      missing_pixels=0, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      tiles=sorted(downloaded, key=lambda x: (x["row"], x["col"])))
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[download] complete {width}x{height}, {resolution} metre pixels: {destination}", flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default=NAIP_2014)
    parser.add_argument("--bounds", type=float, nargs=4, required=True)
    parser.add_argument("--epsg", type=int, default=32611)
    parser.add_argument("--resolution", type=float, default=1)
    parser.add_argument("--tile-pixels", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    download_orthophoto(args.service_url, args.bounds, args.epsg, args.resolution,
                       args.output_dir, args.tile_pixels, args.workers)


if __name__ == "__main__":
    main()
