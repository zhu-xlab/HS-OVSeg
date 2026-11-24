from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets.class_prompts import build_class_catalog
from ovrs_segmentation.datasets.semantic_dataset import SemanticSegmentationDataset
from ovrs_segmentation.preprocessing.test_collection import export_test
from ovrs_segmentation.test_catalog import dataset_identity, load_suite, DEFAULT_CATALOG


class TestCollectionTest(unittest.TestCase):
    def test_files_are_independent_and_encoded_pixels_ids_and_cache_identity_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train_dataset"
            for name in ("images/city", "masks/city", "splits"):
                (source / name).mkdir(parents=True)
            Image.new("RGB", (3, 2), "red").save(source / "images/city/sample.jpg")
            mask = np.asarray([[0, 1, 255], [1, 0, 1]], np.uint8)
            Image.fromarray(mask).save(source / "masks/city/sample.png")
            (source / "splits/test.txt").write_bytes(b"city/sample\r\n")
            (source / "splits/train.txt").write_text("another_sample\n")
            metadata = dict(format_version="ovrs-semantic-v1", dataset_name="Toy", background_id=0, ignore_id=255)
            (source / "dataset.json").write_text(json.dumps(metadata))
            (source / "classes.json").write_text(json.dumps(build_class_catalog("Toy", {0: "background", 1: "building"}, default_semantic_level="lulc", background_id=0)))
            expected = dataset_identity(source)
            destination = Path(directory) / "segmentation_test/Toy"
            result = export_test(source, destination)
            self.assertEqual(result["samples"], 1)
            self.assertEqual(dataset_identity(destination), expected)
            self.assertEqual((source / "dataset.json").read_text(), json.dumps(metadata))
            self.assertEqual((destination / "splits/test.txt").read_bytes(), b"city/sample\r\n")
            self.assertFalse((destination / "splits/train.txt").exists())
            self.assertEqual((source / "images/city/sample.jpg").read_bytes(), (destination / "images/city/sample.jpg").read_bytes())
            np.testing.assert_array_equal(SemanticSegmentationDataset(destination, "test")[0]["class_id_mask"], mask)
            self.assertTrue(export_test(source, destination)["already_exported"])
            (destination / "splits/test.txt").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "vocabulary or test IDs"):
                dataset_identity(destination)

    def test_catalog_keeps_source_and_external_means_separate_under_one_directory(self):
        all_rows = load_suite(DEFAULT_CATALOG, Path("data"))
        self.assertEqual(len(all_rows), 25)
        self.assertEqual(len({r["root"] for r in all_rows}), 25)
        self.assertTrue(all(r["root"].is_relative_to(Path("data/segmentation_test")) for r in all_rows))
        source = load_suite(DEFAULT_CATALOG, Path("data"), scope="training_source")
        self.assertEqual(len(source), 20)
        self.assertFalse({"DLRSD", "VDD", "DynamicEarthNet"} & {r["name"] for r in source})
        external = load_suite(DEFAULT_CATALOG, Path("data"), scope="out_of_domain")
        self.assertEqual(len(external), 5)
        self.assertIn("DynamicEarthNet", {r["name"] for r in external})


if __name__ == "__main__":
    unittest.main()
