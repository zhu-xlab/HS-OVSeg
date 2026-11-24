import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ovrs_segmentation.inference.georaster import export_activations, probability_uint8
from ovrs_segmentation.inference.georaster_multiscale import fuse_probabilities, resample_rgb


class MultiscaleTest(unittest.TestCase):
    def test_float_stripes_preserve_rank_rows_class_order_and_fusion_before_quantization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = []
            names = {0: "airport", 44: "forest", 99: "road", 100: "water"}
            for resolution in [.5, 1, 2, 5]:
                directory = root / str(resolution)
                (directory / "partial").mkdir(parents=True)
                width = int(10 / resolution)
                profile = dict(driver="GTiff", width=width, height=width, count=4, dtype="float32",
                               crs="EPSG:32611", transform=from_origin(0, 10, resolution, resolution))
                # Class maxima must come from different scales independently.
                probabilities = { .5: [.1, .9, .25, .02], 1: [.2, .8, .75, .9801],
                                  2: [.7, .6, .5, .8], 5: [.501, .3, .6, .9]}[resolution]
                for rank in range(2):
                    top, bottom = width * rank // 2, width * (rank + 1) // 2
                    data = np.zeros((4, width, width), np.float32)
                    for band, value in enumerate(probabilities):
                        data[band, top:bottom] = value
                    if resolution == 2:
                        data[0, top:bottom, 0] = .501
                        data[0, top:bottom, 1] = .999
                    with rasterio.open(directory / f"partial/rank_{rank:02d}.tif", "w", **profile) as target:
                        target.write(data)
                summary = dict(complete=True, classes=names, prompts=list(names.values()),
                               checkpoint_sha256="same", checkpoint_epoch=26, width=width, height=width,
                               value_semantics="presence_weighted_probability", probability_dtype="float32",
                               probability_shards=["partial/rank_00.tif", "partial/rank_01.tif"],
                               pixel_size=[resolution, resolution])
                (directory / "summary.json").write_text(json.dumps(summary))
                export_activations(directory, directory / "partial", {**profile, "dtype": "uint8", "count": 1},
                                   names, 2, float_probabilities=True)
                with rasterio.open(directory / "airport.tif") as f:
                    expected = np.full((width, width), probabilities[0], np.float32)
                    if resolution == 2:
                        expected[:, 0], expected[:, 1] = .501, .999
                    np.testing.assert_array_equal(f.read(1), probability_uint8(expected))
                directories.append(directory)
            reference = root / "reference.tif"
            with rasterio.open(reference, "w", driver="GTiff", width=10, height=10, count=3,
                               dtype="uint8", crs="EPSG:32611", transform=from_origin(0, 10, 1, 1)) as f:
                f.write(np.ones((3, 10, 10), np.uint8))
            report = fuse_probabilities(directories, reference, root / "fused")
            for name, value in [("airport", .7), ("forest", .9), ("road", .75), ("water", .9801)]:
                with rasterio.open(root / f"fused/{name}.tif") as f:
                    if name == "airport":
                        # Interpolating uint8 first gives 158; probabilities first gives 159.
                        self.assertEqual(int(f.read(1)[4, 1]), 159)
                    else:
                        np.testing.assert_array_equal(f.read(1), probability_uint8(np.full((10, 10), value)))
                    self.assertEqual(f.scales, (1.,))
                    self.assertIsNone(f.nodata)
                    self.assertEqual(int(f.tags(1)["STATISTICS_MAXIMUM"]), int(f.read(1).max()))
            self.assertEqual(report["fusion"], "per_class_max_before_quantization")
            self.assertEqual(report["resampling"], ["average", "bilinear", "bilinear", "bilinear"])

    def test_resampled_rgb_preserves_extent_and_uses_area_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tif"
            image = np.broadcast_to(np.arange(16, dtype=np.uint8).reshape(4, 4), (3, 4, 4))
            with rasterio.open(source, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8",
                               crs="EPSG:32611", transform=from_origin(0, 2, .5, .5)) as f:
                f.write(image)
            resample_rgb(source, root / "1m/image.tif", 1)
            with rasterio.open(root / "1m/image.tif") as f:
                self.assertEqual(f.shape, (2, 2))
                self.assertEqual(tuple(f.bounds), (0, 0, 2, 2))
                np.testing.assert_array_equal(f.read(1), [[3, 5], [11, 13]])


if __name__ == "__main__":
    unittest.main()
