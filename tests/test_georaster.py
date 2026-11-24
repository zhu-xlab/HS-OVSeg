import unittest
import numpy as np

from ovrs_segmentation.inference.georaster import (
    activation_filename, blend_weights, blended_stripe,
    class_colors, export_activations, raster_windows,
    presence_weighted_probabilities, probability_uint8,
)


class GeorasterTest(unittest.TestCase):
    def test_presence_gates_each_class_before_overlap_blending_and_quantization(self):
        import torch
        pixels = torch.tensor([[[0., 1000.]], [[1000., -1000.]]])
        presence = torch.tensor([0., -1000.])
        probabilities = presence_weighted_probabilities(pixels, presence).numpy()
        np.testing.assert_array_equal(probabilities, [[[.25, .5]], [[0., 0.]]])
        np.testing.assert_array_equal(probability_uint8(probabilities), [[[63, 127]], [[0, 0]]])
        np.testing.assert_array_equal(probability_uint8([0., .5, 1.]), [0, 127, 255])
        windows = raster_windows(4, 6, (4, 4), (2, 2))
        def predict(window):
            return presence_weighted_probabilities(torch.zeros(1, 4, 4),
                                                   torch.tensor([0. if window.col_off == 0 else 1000.])).numpy()
        scores = np.concatenate([a.copy() for _, a in blended_stripe(windows, 0, 4, 6, (4, 4), [0], predict,
                                                             return_logits=True)], axis=1)
        self.assertTrue(np.all((scores >= .25 - 1e-7) & (scores <= .5 + 1e-7)))
        self.assertAlmostEqual(float(scores[0, 1, 0]), .25)
        self.assertAlmostEqual(float(scores[0, 1, -1]), .5)
        self.assertGreater(float(scores[0, 1, 2]), .25)
        self.assertLess(float(scores[0, 1, 2]), .5)
        with self.assertRaises(ValueError):
            presence_weighted_probabilities(pixels, torch.zeros(3))
        with self.assertRaises(ValueError):
            presence_weighted_probabilities(pixels, torch.tensor([float('nan'), 0.]))
        for values in ([float('nan')], [-.01], [1.01]):
            with self.assertRaises(ValueError):
                probability_uint8(values)

    def test_all_45_classes_have_a_geotiff_color(self):
        names = {i: f"class {i}" for i in range(1, 46)}
        colors = class_colors(names)
        self.assertEqual(set(colors), set(names))
        self.assertTrue(all(len(v) == 4 and v[-1] == 255 for v in colors.values()))
        self.assertNotEqual(colors[7], colors[45])

    def test_windows_cover_every_pixel_exactly_once(self):
        covered = np.zeros((13, 19), dtype=np.uint8)
        for window in raster_windows(13, 19, (7, 7)):
            covered[int(window.row_off):int(window.row_off + window.height),
                    int(window.col_off):int(window.col_off + window.width)] += 1
        np.testing.assert_array_equal(covered, np.ones_like(covered))

    def test_twenty_thousand_pixels_require_39_by_39_windows(self):
        windows = raster_windows(20000, 20000, (518, 518))
        self.assertEqual(len(windows), 39 * 39)
        self.assertEqual(sum(w.width * w.height for w in windows), 400_000_000)
        self.assertEqual((windows[-1].width, windows[-1].height), (316, 316))

    def test_overlap_windows_cover_full_image_and_align_to_last_edge(self):
        windows = raster_windows(20000, 20000, (518, 518), (259, 259))
        self.assertEqual(len(windows), 77 * 77)
        self.assertTrue(all(w.height == w.width == 518 for w in windows))
        self.assertEqual((windows[-1].row_off, windows[-1].col_off), (19482, 19482))
        for stride in (0, 519):
            with self.assertRaises(ValueError):
                raster_windows(1000, 1000, (518, 518), (stride, stride))

    def test_streaming_blend_matches_dense_reference_across_rank_boundaries(self):
        height, width, tile = 17, 19, (6, 6)
        windows = raster_windows(height, width, tile, (3, 3))
        rng = np.random.default_rng(7)
        values = {(w.row_off, w.col_off): rng.normal(size=(3, int(w.height), int(w.width))).astype(np.float32)
                  for w in windows}
        def predict(w):
            return values[w.row_off, w.col_off]
        scores = np.zeros((3, height, width), np.float32)
        normalizer = np.zeros((height, width), np.float32)
        weight = blend_weights(tile)
        self.assertGreater(weight[3, 3], weight[0, 0])
        for w in windows:
            y, x = int(w.row_off), int(w.col_off)
            scores[:, y:y + 6, x:x + 6] += predict(w) * weight
            normalizer[y:y + 6, x:x + 6] += weight
        self.assertTrue(np.all(normalizer > 0))
        ids = np.asarray([2, 8, 44], np.uint8)
        expected = ids[(scores / normalizer[None]).argmax(0)]
        for boundaries in ([0, height], [0, 4, 9, 14, height]):
            actual = np.full((height, width), 255, np.uint8)
            actual_scores = np.empty_like(scores)
            for top, bottom in zip(boundaries, boundaries[1:]):
                for window, labels in blended_stripe(windows, top, bottom, width, tile, ids, predict):
                    y = int(window.row_off)
                    actual[y:y + int(window.height)] = labels
                for window, logits in blended_stripe(windows, top, bottom, width, tile, ids, predict, return_logits=True):
                    y = int(window.row_off)
                    actual_scores[:, y:y + int(window.height)] = logits
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(actual_scores, scores / normalizer[None], rtol=1e-6, atol=1e-7)

    def test_fuses_logits_before_selecting_class(self):
        windows = raster_windows(4, 6, (4, 4), (2, 2))
        def predict(w):
            logits = np.zeros((2, 4, 4), np.float32)
            logits[0 if w.col_off == 0 else 1] = .01 if w.col_off == 0 else 100
            return logits
        pieces = list(blended_stripe(windows, 0, 4, 6, (4, 4), [3, 9], predict))
        actual = np.concatenate([labels for _, labels in pieces])
        self.assertEqual(actual[1, 2], 9)
        self.assertTrue(np.all(actual[:, :2] == 3))

    def test_smaller_than_window_has_no_unassigned_pixels(self):
        windows = raster_windows(2, 3, (6, 6), (3, 3))
        pieces = list(blended_stripe(windows, 0, 2, 3, (6, 6), [0, 44],
                                    lambda w: np.full((2, 2, 3), -100, np.float32)))
        self.assertEqual(len(pieces), 1)
        np.testing.assert_array_equal(pieces[0][1], np.zeros((2, 3), np.uint8))
        self.assertEqual(list(blended_stripe(windows, 1, 1, 3, (6, 6), [0, 44],
                                            lambda w: self.fail("Empty GPU stripe must not infer"))), [])


    def test_activation_filenames_retain_class_names(self):
        self.assertEqual(activation_filename("dense residential"), "dense residential.tif")
        for name in ("../airport", "a/b", "a\\b", "..", "", "airport."):
            with self.assertRaises(ValueError):
                activation_filename(name)

    def test_class_tiffs_merge_rank_stripes_without_nodata_or_band_mixing(self):
        import tempfile
        from pathlib import Path
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            partials = output / "partial"
            partials.mkdir()
            profile = dict(driver="GTiff", width=6, height=5, count=1, dtype="uint8",
                           crs="EPSG:32611", transform=from_origin(476000, 3634000, 1, 1),
                           tiled=True, blockxsize=512, blockysize=512, nodata=255, predictor=2)
            expected = np.stack([np.arange(30, dtype=np.uint8).reshape(5, 6), np.full((5, 6), 255, np.uint8)])
            for rank in range(2):
                top, bottom = 5 * rank // 2, 5 * (rank + 1) // 2
                data = np.zeros_like(expected)
                data[:, top:bottom] = expected[:, top:bottom]
                with rasterio.open(partials / f"rank_{rank:02d}.tif", "w", **{**profile, "count": 2, "nodata": None, "interleave": "band"}) as f:
                    f.write(data)
            names = {0: "airplane", 44: "wetland"}
            files = export_activations(output, partials, profile, names, 2)
            self.assertEqual(files, {"0": "airplane.tif", "44": "wetland.tif"})
            for band, (class_id, name) in enumerate(names.items()):
                with rasterio.open(output / files[str(class_id)]) as f:
                    np.testing.assert_array_equal(f.read(1), expected[band])
                    np.testing.assert_array_equal(f.dataset_mask(), np.full((5, 6), 255, np.uint8))
                    self.assertEqual(f.count, 1)
                    self.assertIsNone(f.nodata)
                    self.assertEqual(f.dtypes, ("uint8",))
                    self.assertEqual(f.compression.value, "DEFLATE")
                    self.assertEqual(f.transform, profile["transform"])
                    self.assertEqual(f.crs.to_epsg(), 32611)
                    self.assertEqual(f.scales[0], 1)
                    self.assertEqual(f.descriptions[0], name)
            timestamps = {name: (output / name).stat().st_mtime_ns for name in files.values()}
            export_activations(output, partials, profile, names, 2, resume=True)
            self.assertEqual(timestamps, {name: (output / name).stat().st_mtime_ns for name in files.values()})
            with rasterio.open(output / files["0"], "r+") as f:
                f.update_tags(value_semantics="relative_activation_not_probability")
            with self.assertRaisesRegex(ValueError, "incompatible probability encoding"):
                export_activations(output, partials, profile, names, 2, resume=True)


if __name__ == "__main__":
    unittest.main()
