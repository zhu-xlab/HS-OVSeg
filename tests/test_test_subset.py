import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ovrs_segmentation.preprocessing.test_subset import balanced_pool, prepare, select_diverse, source_group


class TestSubset(unittest.TestCase):
    def test_groups_preserve_original_image_and_flair_site(self):
        self.assertEqual(source_group("region/original__x512_y1024"), "region/original")
        self.assertEqual(source_group("D012-2019_AERIAL_RGBI_AA-S1-12_1-10"), "D012-2019_AERIAL_RGBI_AA-S1-12")
        self.assertEqual(source_group("agricultural/agricultural00"), "agricultural")

    def test_balanced_reproducible_and_order_independent(self):
        ids = [f"{group}__x{i}_y0" for group in ["a", "b", "c"] for i in range(20)]
        chosen = balanced_pool(ids, 12, 7)
        self.assertEqual(chosen, balanced_pool(list(reversed(ids)), 12, 7))
        self.assertEqual([sum(source_group(x) == g for x in chosen) for g in ["a", "b", "c"]], [4, 4, 4])

    def test_rare_class_and_composition_diversity(self):
        ids = [f"scene__x{i}_y0" for i in range(20)]
        hist = np.tile([100, 0, 0], (20, 1))
        hist[18] = [0, 100, 0]
        hist[19] = [99, 0, 1]
        chosen = select_diverse(ids, hist, 4, 7)
        self.assertEqual(len(set(chosen)), 4)
        self.assertTrue((hist[chosen].sum(axis=0) > 0).all())
        self.assertEqual(chosen, select_diverse(ids, hist, 4, 7))

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            select_diverse(["same", "same"], [[1], [1]], 1, 7)

    def test_prepare_preserves_full_split_and_training_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "splits").mkdir()
            original = "\n".join(f"image{i}" for i in range(8)) + "\n"
            (root / "splits/test.txt").write_text(original)
            for name in ["train", "val"]:
                (root / f"splits/{name}.txt").write_text(name + "_sample\n")
            (root / "dataset.json").write_text(json.dumps({"split_counts": {"train": 1, "val": 1, "test": 8}}))
            fake = SimpleNamespace(dataset_name="demo", class_text_by_id={1: "a", 2: "b"},
                _masks={f"image{i}": i for i in range(8)},
                _read_mask=lambda i: np.array([[1, 2 if i % 2 else 1]], dtype=np.uint8))
            with patch("ovrs_segmentation.datasets.semantic_dataset.SemanticSegmentationDataset", return_value=fake) as reader:
                first = prepare(root, size=4, workers=1)
                self.assertEqual(prepare(root, size=4, workers=1), first)
                self.assertEqual(reader.call_count, 1)
            self.assertEqual((root / "splits/test_full.txt").read_text(), original)
            self.assertEqual(len((root / "splits/test.txt").read_text().splitlines()), 4)
            for name in ["train", "val"]:
                self.assertEqual((root / f"splits/{name}.txt").read_text(), name + "_sample\n")
            self.assertEqual(json.loads((root / "dataset.json").read_text())["split_counts"],
                             {"train": 1, "val": 1, "test": 4, "test_full": 8})


if __name__ == "__main__":
    unittest.main()
