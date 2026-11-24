import unittest

from ovrs_segmentation.training.sam3_multilevel import _should_validate


class ValidationScheduleTest(unittest.TestCase):
    def test_every_third_epoch_and_non_multiple_final_epoch(self):
        self.assertEqual([epoch for epoch in range(1, 21) if _should_validate(epoch, 20, 3)],
                         [3, 6, 9, 12, 15, 18, 20])

    def test_single_epoch_and_divisible_final_epoch(self):
        self.assertTrue(_should_validate(1, 1, 3))
        self.assertEqual([epoch for epoch in range(1, 7) if _should_validate(epoch, 6, 3)], [3, 6])

    def test_resume_preserves_global_epoch_schedule(self):
        self.assertEqual([epoch for epoch in range(8, 11) if _should_validate(epoch, 10, 3)], [9, 10])


if __name__ == "__main__":
    unittest.main()
