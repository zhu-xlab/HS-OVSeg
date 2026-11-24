import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ovrs_segmentation.evaluation.presence import PresenceMetrics


class PresenceTest(unittest.TestCase):
    def test_average_precision_uses_scores_and_handles_ties_and_unsupported_classes(self):
        from ovrs_segmentation.evaluation.presence import average_precision
        self.assertAlmostEqual(average_precision([1, 0, 1], [.9, .8, .7]), 5 / 6)
        self.assertEqual(average_precision([1, 0], [.5, .5]), .5)
        self.assertEqual(average_precision([0, 1], [.5, .5]), .5)
        self.assertIsNone(average_precision([0, 0], [.1, .9]))
        metric = PresenceMetrics([(1, "plane"), (2, "ship")], source="native_head")
        metric.update(np.array([1]), np.array([0]), np.array([True]), scores=[.1, .9], sample_id="a")
        result = metric.result()
        self.assertEqual(result["mean_average_precision"], 1)
        self.assertEqual(result["micro_f1"], 0)
        self.assertEqual(result["score_rows"][0]["sample_id"], "a")

    def test_ignore_false_positive_and_multi_label_counts(self):
        metric = PresenceMetrics([(1, "plane"), (2, "ship")], source="segmentation_mask")
        metric.update(np.array([1, 0, 255]), np.array([1, 2, 2]), np.array([1, 1, 0]))
        metric.update(np.array([0, 2, 255]), np.array([0, 0, 1]), np.array([1, 1, 0]))
        result = metric.result()
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["micro_precision"], .5)
        self.assertEqual(result["micro_recall"], .5)
        self.assertEqual(result["micro_f1"], .5)
        self.assertEqual(result["exact_match_accuracy"], 0)
        self.assertEqual(result["macro_f1"], .5)

    def test_native_is_independent_of_mask_and_threshold_is_fixed(self):
        metric = PresenceMetrics([(1, "plane"), (2, "ship")], source="native_head")
        metric.update(np.array([1, 0]), np.array([2, 2]), np.ones(2, bool), scores=[.5, .49])
        self.assertEqual(metric.result()["accuracy"], 1)
        with self.assertRaises(ValueError):
            metric.update(np.array([1]), np.array([1]), np.ones(1, bool), scores=[float("nan"), .1])

    def test_all_ignored_is_not_a_negative_image(self):
        metric = PresenceMetrics([(1, "plane")], source="segmentation_mask")
        metric.update(np.array([255]), np.array([1]), np.zeros(1, bool))
        self.assertIsNone(metric.result()["accuracy"])
        self.assertEqual(metric.result()["images"], 0)

    def test_native_tile_max_does_not_change_segmentation(self):
        from ovrs_segmentation.evaluation.segmentation_test import tiled_pixel_predictions
        outputs = [dict(unified_pixel_logits=torch.ones(2, 4, 4), concept_logits=torch.tensor(values))
                   for values in ([2., -2.], [-2., 2.])]
        scores = []
        with patch("ovrs_segmentation.evaluation.segmentation_test._forward_prompt_chunks", side_effect=outputs):
            prediction = tiled_pixel_predictions(SimpleNamespace(sam3_input_size=(4, 4)),
                torch.zeros(1, 3, 4, 8), ["plane", "ship"], torch.device("cpu"), "fp32",
                background=False, presence_scores=scores)
        self.assertEqual(prediction.shape, (4, 8))
        self.assertTrue(prediction.eq(0).all())
        np.testing.assert_allclose(np.max(scores, axis=0), [.880797, .880797], rtol=1e-5)
