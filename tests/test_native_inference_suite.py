import unittest
import numpy as np
import torch

from ovrs_segmentation.inference.sam3_native_suite import forward_native
from ovrs_segmentation.inference.sam3_multilevel_suite import predict_pixels


class NativeSuiteTest(unittest.TestCase):
    def test_native_semantic_only_chunking_tiling_and_original_size(self):
        calls = []

        class Backbone:
            def forward_image(self, image):
                return image

            def forward_text(self, prompts, device):
                calls.append(prompts)
                return prompts

        class Native:
            sam3_input_size = (4, 4)
            prompt_chunk_size = 1
            image_mean = torch.zeros(1, 3, 1, 1)
            image_std = torch.ones(1, 3, 1, 1)
            backbone = Backbone()

            def _decode_grounding(self, image, prompts, owners):
                value = image[0, 0] * 255 - 15
                # No Router, query masks, presence or GT annotations are needed.
                return {"semantic_seg": torch.stack([value if p == "a" else -value for p in prompts])}

            def _native_semantic_logits(self, native, count):
                return native["semantic_seg"]

        image = np.arange(35, dtype=np.uint8).reshape(5, 7, 1).repeat(3, axis=2)
        result = predict_pixels(Native(), image, ["a", "b"], [0, 94], 255,
                                torch.device("cpu"), "fp32", .5, forward=forward_native)
        expected = np.where(image[:, :, 0] > 15, 0, 94)
        expected[image[:, :, 0] == 15] = 255
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(calls, [["a"], ["b"]] * 4)
        small = predict_pixels(Native(), image[:2, :3], ["a", "b"], [0, 94], 255,
                               torch.device("cpu"), "fp32", .5, forward=forward_native)
        self.assertEqual(small.shape, (2, 3))
        np.testing.assert_array_equal(small, np.full((2, 3), 94))


if __name__ == "__main__":
    unittest.main()
