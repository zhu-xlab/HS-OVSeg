import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DATASETS = {
    "scene": ["AID", "NWPU-RESISC45", "PatternNet"],
    "object": ["DIOR-R", "DOTA-v2", "FAIR1M-2.0", "xView"],
    "lulc": ["DeepGlobe-LandCover", "FLAIR-HUB", "Five-Billion-Pixels", "FloodNet",
             "GID-15", "Global-LoveDA", "ISPRS-Potsdam", "LandCover.ai", "OpenEarthMap",
             "RescueNet", "SkyScapes", "UAVid", "iSAID"],
}
# Capture the portable launcher command without training or filesystem writes.
HARNESS = r'''
mkdir() { :; }
cd() { :; }
tee() { command cat; }
torchrun() {
    "$OVRS_TEST_PYTHON" -c 'import json,os,sys; open(os.environ["OVRS_TEST_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))' "$@"
}
export -f mkdir cd tee torchrun
bash "$@"
'''


class PrimaryTrainingScriptsTest(unittest.TestCase):
    def test_joint_launcher_resources(self):
        text = (SCRIPTS / "train_sam3_multilevel.sh").read_text()
        for token in ("CUDA_VISIBLE_DEVICES:-0,1,2,3", 'nproc_per_node="${#visible_gpus[@]}"'):
            self.assertIn(token, text)

    @unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Requires Linux Bash")
    def test_joint_fresh_and_resume_commands(self):
        for mode in ("fresh", "resume"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                capture = Path(directory) / "args.json"
                env = {**os.environ, "OVRS_DATA_ROOT": "/datasets", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OVRS_TEST_PYTHON": sys.executable,
                       "OVRS_TEST_CAPTURE": str(capture)}
                subprocess.run(["bash", "-c", HARNESS, "test", str(SCRIPTS / "train_sam3_multilevel.sh"), mode],
                               env=env, check=True, capture_output=True, text=True, timeout=20)
                args = json.loads(capture.read_text())
                value = lambda flag: args[args.index(flag) + 1]
                self.assertIn("ovrs_segmentation.training.sam3_multilevel", args)
                for level, names in DATASETS.items():
                    self.assertEqual([Path(args[i + 1]).name for i, flag in enumerate(args)
                                      if flag == f"--{level}-root"], names)
                self.assertEqual(value("--samples-per-epoch"), "100000")
                self.assertEqual(value("--epochs"), "30")
                self.assertEqual(value("--sam3-input-size"), "518")
                self.assertEqual(value("--validation-every"), "3")
                self.assertEqual(value("--batch-size"), "4")
                self.assertEqual(Path(value("--checkpoint-dir")).parts[-2:], ("checkpoints", "sam3_multilevel"))
                if mode == "resume":
                    self.assertEqual(value("--resume"), value("--checkpoint-dir") + "/last.pt")
                else:
                    self.assertNotIn("--resume", args)


if __name__ == "__main__":
    unittest.main()
