import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import SemanticSegmentationDataset
from ovrs_segmentation.preprocessing.manual_coco import convert


class ManualCocoTest(unittest.TestCase):
    def source(self, root):
        source = root / "source"
        (source / "Images").mkdir(parents=True)
        (source / "Annotations").mkdir()
        Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(source / "Images/a.png")
        categories = [{"id": 11, "name": "fixed wing aircraft"}, {"id": 94, "name": "tower"}]
        document = {
            "images": [{"id": 1, "file_name": "a.png", "width": 5, "height": 5}],
            "categories": categories + [{"id": 95, "name": "unused editing category"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 94,
                             "segmentation": [[-1, -1, 2, -1, 2, 2, -1, 2]]}],
        }
        (source / "Annotations/_workspace_categories.json").write_text(json.dumps({"categories": categories}))
        (source / "Annotations/a.json").write_text(json.dumps(document))
        return source, document

    def test_preserves_sparse_ids_clips_polygons_and_exports_portable_pixel_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.source(root)
            output = root / "xView"
            convert(source, output, dataset_name="xView", semantic_level="object")
            dataset = SemanticSegmentationDataset(output, "test")
            sample = dataset[0]
            self.assertEqual(sample["image"].shape, (5, 5, 3))
            self.assertEqual(set(dataset.class_text_by_id), {0, 11, 94})
            self.assertEqual(sample["class_id_mask"][0, 0], 94)
            self.assertEqual(sample["class_id_mask"][4, 4], 0)
            self.assertFalse((output / "splits/train.txt").exists())
            report = json.loads((output / "export_report.json").read_text())["samples"][0]
            self.assertEqual(report["unused_extra_categories"], {"95": "unused editing category"})
            with self.assertRaises(FileExistsError):
                convert(source, output, dataset_name="xView", semantic_level="object")

    def test_refuses_used_extra_category_and_leaves_source_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, document = self.source(root)
            document["annotations"][0]["category_id"] = 95
            (source / "Annotations/a.json").write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                convert(source, root / "result", dataset_name="xView", semantic_level="object")
            self.assertFalse((root / "result").exists())
            self.assertTrue((source / "Annotations/a.json").exists())
            self.assertFalse(list(root.glob(".manual-coco-export-*")))

    def test_retries_transient_directory_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.source(root)
            rename = Path.rename
            calls = []

            def temporarily_locked(path, target):
                calls.append(path)
                if len(calls) == 1:
                    raise PermissionError("sync lock")
                return rename(path, target)

            with patch.object(Path, "rename", temporarily_locked), patch(
                "ovrs_segmentation.preprocessing.manual_coco.time.sleep"
            ):
                convert(source, root / "result", dataset_name="xView", semantic_level="object")
            self.assertEqual(len(calls), 2)
            self.assertTrue((root / "result/dataset.json").is_file())

    def test_refuses_misaligned_image_annotation_pairs(self):
        for field, value in [("image_id", 2), ("width", 6), ("file_name", "other.png")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, document = self.source(root)
                if field == "image_id":
                    document["annotations"][0][field] = value
                else:
                    document["images"][0][field] = value
                (source / "Annotations/a.json").write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    convert(source, root / "result", dataset_name="xView", semantic_level="object")
                self.assertFalse((root / "result").exists())


if __name__ == "__main__":
    unittest.main()
