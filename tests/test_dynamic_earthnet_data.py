import io
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from ovrs_segmentation.preprocessing.dynamic_earthnet import (
    HTTPRangeFile, decode_onehot, rgb_uint8, selected_members, tiles,
)


class DynamicEarthNetDataTest(unittest.TestCase):
    def test_range_zip_reads_only_selected_member(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('unused.bin', b'x' * 1_000_000)
            archive.writestr('labels/5111_4560_13-38S/Labels/Raster/38S-test/38S-test-2018-07-01.tif', b'GT')
        content = buffer.getvalue()
        requests = []

        def fetch(url, start, end):
            requests.append((start, end))
            return content[start:end+1], len(content)

        with patch('ovrs_segmentation.preprocessing.dynamic_earthnet.fetch_range', fetch):
            with HTTPRangeFile('https://official.example/labels.zip') as remote, zipfile.ZipFile(remote) as archive:
                members = selected_members(archive, '2018-07-01')
                self.assertEqual(archive.read(members['5111_4560_13_38S']), b'GT')
        self.assertLess(sum(end-start+1 for start, end in requests), 2000)

    def test_onehot_preserves_snow_and_ignores_missing_or_conflicting_labels(self):
        label = np.zeros((7, 1, 4), dtype=np.uint8)
        label[6, 0, 0] = 255
        label[1, 0, 1] = 255
        label[[2, 3], 0, 2] = 255
        self.assertEqual(decode_onehot(label).tolist(), [[6, 1, 255, 255]])

    def test_planet_uses_surface_reflectance_not_quality_flags(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('planet/15N/25E-155N/2006_3280_13/PF-SR/2018-07-01.tif', b'RGB')
            archive.writestr('planet/15N/25E-155N/2006_3280_13/PF-QA/2018-07-01.tif', b'QA')
        with zipfile.ZipFile(buffer) as archive:
            member = selected_members(archive, '2018-07-01')['2006_3280_13_15N']
            self.assertEqual(archive.read(member), b'RGB')

    def test_rgb_bands_and_edge_tiles_preserve_the_source_grid(self):
        bands = np.array([300, 1500, 6000, 10000], dtype=np.uint16)[:, None, None]
        self.assertEqual(rgb_uint8(bands).tolist(), [[[255, 128, 26]]])
        mask = np.ones((1024, 1024), dtype=np.uint8)
        rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
        patches = list(tiles(rgb, mask))
        self.assertEqual(len(patches), 4)
        self.assertEqual(sum(np.count_nonzero(y != 255) for _, _, _, y in patches), mask.size)
        self.assertEqual(patches[-1][2][-1, -1].tolist(), [128, 128, 128])
        self.assertEqual(patches[-1][3][-1, -1], 255)


if __name__ == '__main__':
    unittest.main()
