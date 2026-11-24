from types import SimpleNamespace
import unittest

import numpy as np
import torch
from rasterio.windows import Window

from ovrs_segmentation.inference.georaster import batched_window_predictor, predict_window_batch
from ovrs_segmentation.evaluation.multilevel_validation import _forward_single_tile
from ovrs_segmentation.inference.sam3_native_suite import forward_native_tile
from ovrs_segmentation.modeling import NativeSAM3EvaluationModel


class GeorasterBatchTest(unittest.TestCase):
    def test_native_batches_use_raw_semantic_presence_and_correct_image_owners(self):
        model = SimpleNamespace(sam3_input_size=(2, 2), prompt_chunk_size=2,
                                image_mean=torch.zeros(1, 3, 1, 1), image_std=torch.ones(1, 3, 1, 1))
        model.backbone = SimpleNamespace(forward_image=lambda x: x[:, 0, 0, 0],
                                         forward_text=lambda text, device: torch.tensor([int(t) for t in text]))
        owners = []
        def decode(images, text, indices):
            owners.extend(indices.tolist())
            return {"semantic_seg": (images[indices] * 10 + text)[:, None, None],
                    "presence_logit_dec": (images[indices] + text - 2)[:, None]}
        model._decode_grounding = decode
        model._native_semantic_logits = lambda native, count: native["semantic_seg"]
        model._presence_logits = NativeSAM3EvaluationModel._presence_logits
        source = SimpleNamespace(read=lambda bands, window: np.full((3, 2, 2), 0 if window.col_off == 0 else 255, np.uint8))
        values = predict_window_batch(model, source, [Window(0, 0, 2, 2), Window(2, 0, 2, 2)],
                                      ["0", "1", "2"], torch.device("cpu"), "fp32",
                                      probabilities=True, forward=forward_native_tile)
        self.assertEqual(owners, [0, 1, 0, 1, 0, 1])
        np.testing.assert_allclose(values[0][:, 0, 0], (torch.arange(3.).sigmoid() * torch.arange(-2., 1.).sigmoid()).numpy())
        np.testing.assert_allclose(values[1][:, 0, 0], (torch.arange(10., 13.).sigmoid() * torch.arange(-1., 2.).sigmoid()).numpy())

    def test_batches_preserve_window_order_and_final_remainder(self):
        windows = [Window(i * 2, 0, 2, 2) for i in range(5)]
        calls = []
        def run(batch):
            calls.append(len(batch))
            return [w.col_off for w in batch]
        predict = batched_window_predictor(windows, 2, run)
        self.assertEqual([predict(w) for w in windows], [0, 2, 4, 6, 8])
        self.assertEqual(calls, [2, 2, 1])

    def test_each_class_is_matched_to_each_image_without_cross_image_leakage(self):
        model = SimpleNamespace(sam3_input_size=(2, 2), prompt_chunk_size=2,
                                image_mean=torch.zeros(1, 3, 1, 1), image_std=torch.ones(1, 3, 1, 1))
        model.backbone = SimpleNamespace(forward_image=lambda images: images[:, 0, 0, 0],
                                         forward_text=lambda text, device: torch.tensor([int(t) for t in text]))
        owners = []
        def decode(backbone, text, indices, decoder_grad):
            owners.extend(indices.tolist())
            values = backbone[indices] * 10 + text
            return {"semantic_seg": values[:, None, None].expand(-1, 2, 2),
                    "pred_masks": torch.ones(len(values), 1, 2, 2)}
        model._decode_grounding = decode
        model._prompt_outputs = lambda native, count: (torch.zeros(count, 1), torch.zeros(count))
        model._native_semantic_logits = lambda native, count: native["semantic_seg"]
        model.pixel_router = lambda semantic, query, masks, output_size: {"unified_pixel_logits": semantic}
        source = SimpleNamespace(read=lambda bands, window: np.full((3, 2, 2), 0 if window.col_off == 0 else 255, np.uint8))
        windows = [Window(0, 0, 2, 2), Window(2, 0, 2, 2)]
        values = predict_window_batch(model, source, windows, ["0", "1", "2"], torch.device("cpu"), "fp32", probabilities=False)
        self.assertEqual(owners, [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(values[0][:, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(values[1][:, 0, 0], [10, 11, 12])
        values = predict_window_batch(model, source, windows, ["0", "1", "2"], torch.device("cpu"), "fp32", probabilities=True)
        np.testing.assert_allclose(values[0][:, 0, 0], torch.tensor([0., 1., 2.]).sigmoid().numpy() * .5)
        np.testing.assert_allclose(values[1][:, 0, 0], torch.tensor([10., 11., 12.]).sigmoid().numpy() * .5)
        with self.assertRaises(ValueError):
            _forward_single_tile(model, {"images": torch.zeros(2, 3, 2, 2), "prompt_texts": ["0"],
                                         "pixel_targets": torch.empty(0, 2, 2)}, torch.device("cpu"), "fp32")


if __name__ == "__main__":
    unittest.main()
