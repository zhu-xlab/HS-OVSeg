import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from ovrs_segmentation.preprocessing.out_of_domain import decode_mask, prepare, records, safe_id


class OutOfDomainDataTest(unittest.TestCase):
    def test_native_palette_indices_and_unknown_labels(self):
        label = Image.fromarray(np.array([[1, 6]], np.uint8)).convert("P")
        np.testing.assert_array_equal(decode_mask(label, "WHDLD"), [[0, 5]])
        np.testing.assert_array_equal(decode_mask(Image.fromarray(np.array([[0, 5]], np.uint8)), "UDD6"), [[0, 5]])
        with self.assertRaises(ValueError):
            decode_mask(Image.fromarray(np.array([[0]], np.uint8)), "WHDLD")
        with self.assertRaises(ValueError):
            decode_mask(Image.new("RGB", (2, 2)), "UDD6")

    def test_author_validation_manifest_and_safe_ids(self):
        data = b"val/src/000061.JPG val/gt/000061.png\r\n"
        self.assertEqual(records("UDD6", [], lambda _: io.BytesIO(data)),
            [("000061", "UDD/UDD6/val/src/000061.JPG", "UDD/UDD6/val/gt/000061.png")])
        for bad in ("", "..", "../escape", "/absolute", "a\\b"):
            with self.assertRaises(ValueError):
                safe_id(bad)

    def test_prepared_pair_is_test_only_and_pixels_are_not_resized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "WHDLD.zip"
            rgb = io.BytesIO(); Image.new("RGB", (3, 2), "red").save(rgb, format="JPEG")
            mask = io.BytesIO(); Image.fromarray(np.array([[1, 2, 3], [4, 5, 6]], np.uint8)).save(mask, format="PNG")
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("WHDLD/Images/wh0001.jpg", rgb.getvalue())
                source.writestr("WHDLD/Labels/wh0001.png", mask.getvalue())
            output = root / "prepared"
            metadata = prepare("WHDLD", archive, output, expected_count=1)
            self.assertEqual(metadata["splits"], {"test": 1})
            self.assertEqual((output / "images/wh0001.jpg").read_bytes(), rgb.getvalue())
            with Image.open(output / "masks/wh0001.png") as label:
                np.testing.assert_array_equal(label, [[0, 1, 2], [3, 4, 5]])
            self.assertFalse((output / "splits/train.txt").exists())
            self.assertEqual(len(json.loads((output / "image_hashes.json").read_text())), 1)
            with self.assertRaises(FileExistsError):
                prepare("WHDLD", archive, output, expected_count=1)
