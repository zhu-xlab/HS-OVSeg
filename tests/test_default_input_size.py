import inspect
import unittest
from unittest.mock import patch

from ovrs_segmentation.evaluation import sam3_multilevel, sam3_native
from ovrs_segmentation.inference import sam3_native as native_inference
from ovrs_segmentation.modeling import NativeSAM3EvaluationModel, SAM3NativeMultilevelModel
from ovrs_segmentation.training.sam3_multilevel import parse_args as training_args


class DefaultInputSizeTest(unittest.TestCase):
    def test_training_defaults_to_518_and_canonical_directories(self):
        with patch("sys.argv", ["train"]):
            args = training_args()
        self.assertEqual(args.sam3_input_size, 518)
        self.assertEqual(args.checkpoint_dir.as_posix(), "checkpoints/sam3_multilevel")
        self.assertEqual(args.output_dir.as_posix(), "outputs/sam3_multilevel")

    def test_model_defaults_match_training(self):
        for model in (SAM3NativeMultilevelModel, NativeSAM3EvaluationModel):
            self.assertEqual(inspect.signature(model).parameters["sam3_input_size"].default, 518)

    def test_native_evaluation_and_inference_default_to_518(self):
        with patch("sys.argv", ["eval", "--dataset-root", "dataset", "--level", "object",
                               "--load-from-hf"]):
            self.assertEqual(sam3_native.parse_args().sam3_input_size, 518)
        with patch("sys.argv", ["infer", "--prompt", "building", "--level", "object"]):
            self.assertEqual(native_inference.parse_args().sam3_input_size, 518)

    def test_adapted_evaluation_respects_checkpoint_size_and_override(self):
        with patch("sys.argv", ["eval", "--dataset-root", "dataset", "--level", "object",
                               "--checkpoint", "last.pt"]):
            args = sam3_multilevel.parse_args()
        state = {"args": {"lora_rank": 8, "lora_scale": 1.0}}
        self.assertEqual(sam3_multilevel._checkpoint_settings(args, state)["sam3_input_size"], 518)
        state["args"]["sam3_input_size"] = 700
        self.assertEqual(sam3_multilevel._checkpoint_settings(args, state)["sam3_input_size"], 700)
        args.sam3_input_size = 518
        self.assertEqual(sam3_multilevel._checkpoint_settings(args, state)["sam3_input_size"], 518)


if __name__ == "__main__":
    unittest.main()
