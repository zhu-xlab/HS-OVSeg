import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ovrs_segmentation.datasets import write_class_catalog
from ovrs_segmentation.preprocessing.test_tiles import prepare


def fixture(root, count=1):
    for name in ("images", "masks", "splits"):
        (root / name).mkdir(parents=True)
    ids = [f"scene{i}" for i in range(count)]
    for i in ids:
        mask = (np.arange(30).reshape(5, 6) % 2).astype(np.uint8)
        Image.fromarray(np.repeat(mask[..., None] * 100, 3, axis=2)).save(root / f"images/{i}.png")
        Image.fromarray(mask).save(root / f"masks/{i}.png")
    write_class_catalog(root / "classes.json", "demo", {0: "background", 1: "building"},
                        default_semantic_level="lulc", background_id=0, ignore_id=255)
    (root / "dataset.json").write_text(json.dumps(dict(format_version="ovrs-semantic-v1", dataset_name="demo",
        image_dir="images", mask_dir="masks", background_id=0, ignore_id=255)))
    (root / "splits/test.txt").write_text("\n".join(ids) + "\n")


class TestTiles(unittest.TestCase):
    def test_pixels_and_masks_match_and_padding_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "output"
            fixture(source)
            before = (source / "splits/test.txt").read_bytes()
            result = prepare(source, output, tile_size=4)
            self.assertEqual(result["samples"], 4)
            rows = [json.loads(s) for s in (output / "tile_manifest.jsonl").read_text().splitlines()]
            total = 0
            with Image.open(source / "images/scene0.png") as im:
                original = np.array(im)
            for row in rows:
                image = np.array(Image.open(output / f"images/{row['id']}.png"))
                mask = np.array(Image.open(output / f"masks/{row['id']}.png"))
                w, h, x, y = (row[k] for k in ("width", "height", "left", "top"))
                self.assertEqual(mask.shape, (4, 4))
                np.testing.assert_array_equal(image[:h, :w], original[y:y+h, x:x+w])
                np.testing.assert_array_equal(image[:h, :w, 0] // 100, mask[:h, :w])
                self.assertTrue((mask[h:] == 255).all())
                self.assertTrue((mask[:, w:] == 255).all())
                total += int((mask != 255).sum())
            self.assertEqual(total, 30)
            self.assertEqual(before, (source / "splits/test.txt").read_bytes())
            self.assertEqual(result, prepare(source, output, tile_size=4))
            with self.assertRaises(ValueError):
                prepare(source, output, tile_size=3)

    def test_limit_is_applied_to_tiles_and_preserves_source_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "output"
            fixture(source, 4)
            result = prepare(source, output, tile_size=4, size=8)
            self.assertEqual(result["original_samples"], 4)
            self.assertEqual(result["candidate_tiles"], 16)
            self.assertEqual(result["samples"], 8)
            self.assertEqual(result["source_groups_selected"], 4)
            self.assertEqual(len(list((output / "images").glob("*.png"))), 8)
            self.assertEqual(set(result["selected_class_image_counts"]), {"0", "1"})

    def test_directory_lock_falls_back_to_validated_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "output"
            fixture(source)
            with patch.object(Path, "rename", side_effect=PermissionError("sync lock")), patch(
                    "ovrs_segmentation.preprocessing.test_tiles.time.sleep"):
                result = prepare(source, output, tile_size=4)
            self.assertEqual(result["samples"], 4)
            self.assertTrue((output / "test_tiles.json").exists())


if __name__ == "__main__":
    unittest.main()
