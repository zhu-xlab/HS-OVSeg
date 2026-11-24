import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

from ovrs_segmentation.evaluation.segmentation_test import tiled_pixel_predictions
from ovrs_segmentation.evaluation.multilevel_validation import _forward_prompt_chunks


class SegmentationTilesTest(unittest.TestCase):
    def test_inference_cli_restores_small_image_output_dimensions(self):
        from ovrs_segmentation.inference.sam3_multilevel import _predict
        def tile_predict(model, tile, queries, **kwargs):
            self.assertEqual(tile.shape, (4, 4, 3))
            return np.array([1.], dtype=np.float32), None, np.ones((1, 4, 4), dtype=np.float32)
        with patch("ovrs_segmentation.inference.sam3_multilevel._predict_tile", tile_predict):
            _, _, pixels = _predict(SimpleNamespace(sam3_input_size=(4, 4)), np.zeros((2, 3, 3), dtype=np.uint8), ["a"],
                                   device=torch.device("cpu"), precision="fp32", prompt_batch_size=1, aggregation="max")
        self.assertEqual(pixels.shape, (1, 2, 3))
        self.assertTrue((pixels == 1).all())

    def test_inference_cli_stitches_native_pixels(self):
        from ovrs_segmentation.inference.sam3_multilevel import _predict
        image = np.arange(24, dtype=np.uint8).reshape(4, 6, 1).repeat(3, axis=2)
        calls = []

        def tile_predict(model, tile, queries, **kwargs):
            self.assertEqual(tile.shape, (4, 4, 3))
            calls.append(tile)
            return np.array([len(calls)], dtype=np.float32), None, tile[:, :, :1].transpose(2, 0, 1)

        with patch("ovrs_segmentation.inference.sam3_multilevel._predict_tile", tile_predict):
            logits, _, pixels = _predict(SimpleNamespace(sam3_input_size=(4, 4)), image, ["a"],
                                        device=torch.device("cpu"), precision="fp32", prompt_batch_size=1, aggregation="max")
        np.testing.assert_allclose(pixels[0], image[:, :, 0], rtol=1e-6)
        self.assertEqual(logits[0], 2)

    def test_matching_model_input_is_not_resized_by_shared_forward(self):
        model = SimpleNamespace(
            sam3_input_size=(4, 4), image_mean=0.5, image_std=0.5, prompt_chunk_size=1,
            backbone=SimpleNamespace(forward_image=lambda images: {}, forward_text=lambda text, device: {}),
            _decode_grounding=lambda *args, **kwargs: {"pred_masks": torch.zeros(1, 1, 4, 4)},
            _prompt_outputs=lambda native, count: (torch.zeros(1, 1), torch.zeros(1)),
            _native_semantic_logits=lambda native, count: torch.zeros(1, 4, 4),
            pixel_router=lambda *args, **kwargs: {"unified_pixel_logits": torch.zeros(1, 4, 4)},
        )
        batch = {"images": torch.ones(1, 3, 4, 4), "prompt_texts": ["a"],
                 "pixel_targets": torch.empty(0, 4, 4)}
        with patch("ovrs_segmentation.evaluation.multilevel_validation.F.interpolate",
                   side_effect=AssertionError("Matching input must not be resized")):
            result = _forward_prompt_chunks(model, batch, torch.device("cpu"), "fp32")
        self.assertEqual(tuple(result["unified_pixel_logits"].shape), (1, 4, 4))

    def test_large_image_is_cropped_without_resizing_and_edges_are_unpadded(self):
        image = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7) / 35
        image = image.expand(1, 3, 5, 7)
        seen = []

        def forward(model, batch, device, precision):
            tile = batch["images"]
            self.assertEqual(tuple(tile.shape), (1, 3, 4, 4))
            self.assertEqual(tuple(batch["pixel_targets"].shape[-2:]), (4, 4))
            seen.append(tile.clone())
            value = tile[0, 0] - 0.4
            return {"unified_pixel_logits": torch.stack((value, -value))}

        with patch("ovrs_segmentation.evaluation.segmentation_test._forward_prompt_chunks", forward):
            prediction = tiled_pixel_predictions(SimpleNamespace(sam3_input_size=(4, 4)), image,
                                                 ["a", "b"], torch.device("cpu"), "fp32", background=False)
        self.assertEqual(len(seen), 4)
        torch.testing.assert_close(seen[0], image[..., :4, :4])
        self.assertTrue(seen[3][..., 1:, :].eq(0.5).all())
        self.assertTrue(seen[3][..., :, 3:].eq(0.5).all())
        torch.testing.assert_close(prediction, (image[0, 0] < 0.4).long())

    def test_small_image_is_upsampled_and_prediction_returns_to_original_size(self):
        image = torch.zeros(1, 3, 2, 3)
        output = {"unified_pixel_logits": -torch.ones(2, 4, 4),
                  "concept_logits": torch.zeros(2), "boxes_xyxy": torch.zeros(2, 1, 4),
                  "query_confidence": torch.ones(2, 1), "detection_scores": torch.ones(2, 1)}
        with patch("ovrs_segmentation.evaluation.multilevel_validation._forward_single_tile",
                   return_value=output) as forward:
            prediction = tiled_pixel_predictions(SimpleNamespace(sam3_input_size=(4, 4)), image,
                                                 ["a", "b"], torch.device("cpu"), "fp32", background=True)
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(tuple(forward.call_args_list[0].args[1]["images"].shape), (1, 3, 4, 4))
        self.assertTrue(forward.call_args_list[0].args[1]["images"].eq(0).all())
        self.assertEqual(tuple(prediction.shape), (2, 3))
        self.assertTrue(prediction.eq(2).all())

    def test_validation_stitches_pixels_and_offsets_boxes_with_tile_scores(self):
        image = torch.arange(24).reshape(1, 1, 4, 6).float().expand(1, 3, 4, 6)
        calls = []

        def tile_forward(model, batch, device, precision, **kwargs):
            calls.append(batch["images"])
            return {"unified_pixel_logits": batch["images"][0, :1],
                    "concept_logits": torch.tensor([float(len(calls))]),
                    "boxes_xyxy": torch.tensor([[[0., 0., 1., 1.], [.75, 0., 1., 1.]]]),
                    "query_confidence": torch.ones(1, 2),
                    "detection_scores": torch.full((1, 2), 0.2 * len(calls))}

        with patch("ovrs_segmentation.evaluation.multilevel_validation._forward_single_tile", tile_forward):
            result = _forward_prompt_chunks(SimpleNamespace(sam3_input_size=(4, 4)),
                                           {"images": image, "pixel_targets": torch.empty(0, 4, 6)},
                                           torch.device("cpu"), "fp32")
        torch.testing.assert_close(result["unified_pixel_logits"], image[0, :1])
        torch.testing.assert_close(result["boxes_xyxy"][0, 0], torch.tensor([0., 0., 4/6, 1.]))
        torch.testing.assert_close(result["boxes_xyxy"][0, 2], torch.tensor([4/6, 0., 1., 1.]))
        torch.testing.assert_close(result["detection_scores"], torch.tensor([[.2, .2, .4, 0.]]))
        self.assertEqual(result["concept_logits"].item(), 2.)

    def test_small_rectangular_image_keeps_aspect_ratio(self):
        from ovrs_segmentation.inference.tiling import upsample_small_image
        small = torch.zeros(1, 3, 2, 3)
        self.assertEqual(tuple(upsample_small_image(small, (4, 4)).shape), (1, 3, 4, 6))
        large = torch.zeros(1, 3, 5, 7)
        self.assertIs(upsample_small_image(large, (4, 4)), large)


if __name__ == "__main__":
    unittest.main()
