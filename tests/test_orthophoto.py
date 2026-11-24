import unittest

from ovrs_segmentation.preprocessing.orthophoto import export_tiles


class OrthophotoTest(unittest.TestCase):
    def test_twenty_kilometre_grid_is_exactly_one_metre_and_covers_all_pixels(self):
        tiles = export_tiles([476000, 3614000, 496000, 3634000], 1, 4000)
        self.assertEqual(len(tiles), 25)
        self.assertEqual(sum(t["width"] * t["height"] for t in tiles), 400_000_000)
        self.assertEqual(tiles[0]["bounds"], [476000, 3630000, 480000, 3634000])
        self.assertEqual(tiles[-1]["bounds"], [492000, 3614000, 496000, 3618000])
        self.assertEqual(len({(t["row"], t["col"]) for t in tiles}), 25)

    def test_partial_export_edges_preserve_pixel_size(self):
        tiles = export_tiles([0, 0, 13, 9], .5, 10)
        self.assertEqual(sum(t["width"] * t["height"] for t in tiles), 26 * 18)
        for t in tiles:
            self.assertEqual((t["bounds"][2] - t["bounds"][0]) / t["width"], .5)
            self.assertEqual((t["bounds"][3] - t["bounds"][1]) / t["height"], .5)
        with self.assertRaises(ValueError):
            export_tiles([0, 0, 13.1, 9], .5, 10)


if __name__ == "__main__":
    unittest.main()
