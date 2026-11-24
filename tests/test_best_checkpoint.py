import unittest

from ovrs_segmentation.training.sam3_multilevel import (
    _best_validation_state,
    _mean_validation_score,
)


class BestCheckpointTest(unittest.TestCase):
    def test_every_dataset_has_equal_weight_and_uses_its_primary_metric(self):
        metrics = {
            "scene/A": {"scene_full_vocab_top1_accuracy": 0.9},
            "lulc/B": {"lulc_mean_iou": 0.5, "lulc_pixel_accuracy": 0.99},
            "lulc/C": {"lulc_mean_iou": 0.3},
            "object/D": {"object_map50": 0.7, "object_mean_recall50": 0.99},
        }
        self.assertAlmostEqual(_mean_validation_score(metrics), 0.6)

    def test_skipped_validation_has_no_score(self):
        self.assertIsNone(_mean_validation_score({}))

    def test_invalid_dataset_score_is_not_silently_omitted(self):
        for score in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _mean_validation_score({"object/D": {"object_map50": score}})
        with self.assertRaises(KeyError):
            _mean_validation_score({"object/D": {}})

    def test_resume_keeps_historical_best_after_skipped_validation(self):
        self.assertEqual(_best_validation_state({
            "epoch": 6, "validation": {},
            "best_validation_score": 0.8, "best_epoch": 3,
        }), (0.8, 3))

    def test_legacy_checkpoint_can_seed_best_from_its_own_validation(self):
        self.assertEqual(_best_validation_state({
            "epoch": 19,
            "validation": {"scene/A": {"scene_full_vocab_top1_accuracy": 0.9}},
        }), (0.9, 20))

    def test_legacy_checkpoint_without_validation_has_no_known_best(self):
        self.assertEqual(_best_validation_state({"epoch": 6}), (None, None))

    def test_empty_best_metadata_uses_checkpoint_validation(self):
        self.assertEqual(_best_validation_state({
            "epoch": 20, "best_validation_score": None, "best_epoch": None,
            "validation": {"object/D": {"object_map50": 0.7}},
        }), (0.7, 21))


if __name__ == "__main__":
    unittest.main()
