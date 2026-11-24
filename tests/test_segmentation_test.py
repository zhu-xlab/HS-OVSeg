import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ovrs_segmentation.evaluation.segmentation_test import evaluate_segmentation, load_suite


class SegmentationTest(unittest.TestCase):
    def test_original_resolution_background_ignore_and_absent_class(self):
        dataset = SimpleNamespace(
            class_specs_by_id={1: {"class_text": "aircraft", "semantic_level": "object"},
                               2: {"class_text": "ship", "semantic_level": "object"}},
            background_id=0, ignore_id=255, metadata={}, dataset_name="DIOR-R",
            _validate_mask=lambda sample_id, mask: None,
        )
        sample = {"sample_id": "a", "image": np.zeros((2, 3, 3), dtype=np.uint8),
                  "class_id_mask": np.array([[1, 1, 0], [0, 255, 0]], dtype=np.uint8)}
        # One foreground miss, one false positive for an absent class, and an ignored pixel.
        logits = torch.tensor([[[1., -1., -1.], [-1., 1., -1.]],
                               [[-1., -1., 1.], [-1., 2., -1.]]])
        with patch("ovrs_segmentation.evaluation.segmentation_test._forward_prompt_chunks",
                   return_value={"unified_pixel_logits": logits}) as forward:
            metrics, confusion = evaluate_segmentation(SimpleNamespace(sam3_input_size=(4, 4)), [sample], dataset, torch.device("cpu"), "fp32")
        assert forward.call_args.args[1]["pixel_targets"].shape[-2:] == (2, 3)
        assert confusion == [[1, 0, 1], [0, 0, 0], [0, 1, 2]]
        assert metrics["valid_pixels"] == 5
        assert metrics["mean_iou"] == 0.25
        assert metrics["pixel_accuracy"] == 0.6
        assert metrics["presence_mask"]["accuracy"] == 0.5
        assert metrics["presence_native"] is None


    def test_suite_covers_all_training_datasets(self):
        from pathlib import Path
        entries = load_suite(Path("configs/segmentation_test.yaml"), Path("native"), Path("manual"))
        assert len(entries) == 25
        assert {entry["source_task"] for entry in entries} == {"scene", "object", "lulc"}
        assert all(entry["split"] == "test" for entry in entries)
        assert next(entry for entry in entries if entry["name"] == "DIOR-R")["root"] == Path("manual/DIOR-R-GT200")
        assert all(entry["root"].is_relative_to(Path("manual")) for entry in entries)
        assert len(load_suite(Path("configs/segmentation_test.yaml"), Path("native"), Path("manual"), scope="training_source")) == 20
        assert len(load_suite(Path("configs/segmentation_test.yaml"), Path("native"), Path("manual"), scope="out_of_domain")) == 5

    def test_manual_suite_includes_all_seven_reviewed_datasets(self):
        from pathlib import Path
        from ovrs_segmentation.preprocessing.manual_test_names import MANUAL_TEST_COUNTS
        entries = [e for e in load_suite(Path("configs/segmentation_test.yaml"), Path("native"), Path("manual"))
                   if e["name"] in MANUAL_TEST_COUNTS]
        assert {entry["name"] for entry in entries} == set(MANUAL_TEST_COUNTS)
        for entry in entries:
            count = MANUAL_TEST_COUNTS[entry["name"]]
            assert entry["root"] == Path("manual") / f"{entry['name']}-GT{count}"

if __name__ == "__main__":
    unittest.main()
