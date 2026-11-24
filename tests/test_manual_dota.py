import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ovrs_segmentation.preprocessing.manual_dota import DOTA_CLASSES, convert, rasterize_annotations
from ovrs_segmentation.datasets import SemanticSegmentationDataset


class ManualDotaTest(unittest.TestCase):
    def test_union_conflicts_and_background(self):
        def row(category, polygon):
            return {"id": category, "category_id": category, "segmentation": [polygon]}
        annotations = [
            row(1, [0, 0, 2, 0, 2, 2, 0, 2]),
            row(1, [1, 1, 3, 1, 3, 3, 1, 3]),
            row(2, [2, 2, 4, 2, 4, 4, 2, 4]),
            row(3, [2, 2, 3, 2, 3, 3, 2, 3]),
        ]
        mask = rasterize_annotations(annotations, (6, 6), {1: "a", 2: "b", 3: "c"})
        self.assertEqual(mask[1, 1], 1)
        self.assertEqual(mask[2, 2], 255)
        self.assertEqual(mask[4, 4], 2)
        self.assertEqual(mask[5, 5], 0)
        np.testing.assert_array_equal(mask, rasterize_annotations(list(reversed(annotations)), (6, 6), {1: "a", 2: "b", 3: "c"}))

    def test_refuses_boxes_and_unknown_categories(self):
        with self.assertRaises(ValueError):
            rasterize_annotations([{"id": 1, "category_id": 1, "segmentation": []}], (3, 3), {1: "a"})
        with self.assertRaises(ValueError):
            rasterize_annotations([{"id": 1, "category_id": 19}], (3, 3), {1: "a"})

    def test_conversion_normalizes_grayscale_and_keeps_test_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "Images").mkdir(parents=True)
            (source / "Annotations").mkdir()
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(source / "Images/P0001.png")
            document = {
                "images": [{"id": 1, "file_name": "P0001.png", "width": 4, "height": 4}],
                "categories": [{"id": i, "name": name} for i, name in enumerate(DOTA_CLASSES, 1)],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                                 "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]}],
            }
            (source / "Annotations/P0001.json").write_text(json.dumps(document))
            output = root / "DOTA-v2-GT160"
            convert(source, output)
            dataset = SemanticSegmentationDataset(output, "test")
            self.assertEqual(dataset[0]["image"].shape, (4, 4, 3))
            self.assertEqual(dataset[0]["class_id_mask"][0, 0], 2)
            self.assertEqual(dataset[0]["class_id_mask"][3, 3], 0)
            self.assertEqual(len(dataset.classes), 19)
            self.assertFalse((output / "splits/train.txt").exists())
            with self.assertRaises(FileExistsError):
                convert(source, output)


if __name__ == "__main__":
    unittest.main()
