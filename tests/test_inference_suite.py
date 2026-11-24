import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ovrs_segmentation.inference.sam3_multilevel_suite import pixel_winners, predict_pixels, sample_ids
from ovrs_segmentation.evaluation.segmentation_test import load_suite
from pathlib import Path


class InferenceSuiteTest(unittest.TestCase):
    def test_all_twenty_datasets_use_native_validation_roots(self):
        entries = load_suite(Path("configs/inference_val.yaml"), Path("data"), Path("unused"))
        self.assertEqual(len(entries), 20)
        self.assertEqual([sum(e["source_task"] == level for e in entries) for level in ("scene", "object", "lulc")], [3, 4, 13])
        self.assertTrue(all(e["root"] == Path("data") / e["name"] and e["split"] == "val" for e in entries))

    def test_sample_is_unique_reproducible_and_independent_of_input_order(self):
        ids = [str(i) for i in range(100)]
        chosen = sample_ids(ids, 50, 42, "AID")
        self.assertEqual(len(set(chosen)), 50)
        self.assertEqual(chosen, sample_ids(list(reversed(ids)), 50, 42, "AID"))
        self.assertNotEqual(chosen, sample_ids(ids, 50, 42, "NWPU"))
        with self.assertRaises(ValueError):
            sample_ids(ids[:49], 50, 42, "small")

    def test_pixel_argmax_uses_raw_ids_strict_threshold_and_valid_class_zero(self):
        logits = torch.tensor([[[2., -1., 0., -3.]], [[1., 2., -1., -2.]]])
        prediction = pixel_winners(logits, [0, 94], 255, .5)
        np.testing.assert_array_equal(prediction, [[0, 94, 255, 255]])
        with self.assertRaises(ValueError):
            pixel_winners(torch.full((2, 1, 1), float("nan")), [0, 94], 255, .5)

    def test_native_tiles_stitch_at_original_resolution_and_ignore_presence(self):
        image = np.arange(35, dtype=np.uint8).reshape(5, 7, 1).repeat(3, axis=2)
        calls = []

        def forward(model, batch, *args):
            tile = batch["images"]
            self.assertEqual(tuple(tile.shape), (1, 3, 4, 4))
            calls.append(tile)
            value = tile[0, 0] * 255 - 15
            return {"unified_pixel_logits": torch.stack((value, -value)), "concept_logits": torch.tensor([-100., 100.])}

        with patch("ovrs_segmentation.inference.sam3_multilevel_suite._forward_prompt_chunks", forward):
            result = predict_pixels(SimpleNamespace(sam3_input_size=(4, 4)), image, ["a", "b"],
                                    [11, 94], 0, torch.device("cpu"), "fp32", .5)
        expected = np.where(image[:, :, 0] > 15, 11, 94)
        expected[image[:, :, 0] == 15] = 0
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(len(calls), 4)

    def test_small_image_keeps_original_grid_for_continuous_logit_restoration(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        def forward(model, batch, *args):
            self.assertEqual(tuple(batch["pixel_targets"].shape), (0, 2, 3))
            return {"unified_pixel_logits": torch.ones(1, 2, 3)}
        with patch("ovrs_segmentation.inference.sam3_multilevel_suite._forward_prompt_chunks", forward):
            prediction = predict_pixels(SimpleNamespace(sam3_input_size=(4, 4)), image, ["a"],
                                        [0], 255, torch.device("cpu"), "fp32", .5)
        np.testing.assert_array_equal(prediction, np.zeros((2, 3), dtype=np.uint16))

    def test_no_threshold_assigns_every_pixel_even_when_all_logits_are_negative(self):
        logits = torch.tensor([[[-1000., -2.]], [[-999., -3.]]])
        np.testing.assert_array_equal(pixel_winners(logits, [11, 94], 255, None), [[94, 11]])


if __name__ == "__main__":
    unittest.main()
