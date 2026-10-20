import unittest

from scripts.rl.submit_lr_pilots import grid_rates


class LearningRateGridTests(unittest.TestCase):
    def test_upper_grid_includes_existing_control(self):
        self.assertEqual(grid_rates([2e-6, 3e-6]), [1e-6, 2e-6, 3e-6])

    def test_duplicate_rates_are_deduplicated(self):
        self.assertEqual(grid_rates([1e-6, 2e-6, 2e-6]), [1e-6, 2e-6])

    def test_invalid_rates_rejected(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                grid_rates([rate])

    def test_ambiguous_labels_rejected(self):
        with self.assertRaises(ValueError):
            grid_rates([2e-6, 2.1e-6])


if __name__ == "__main__":
    unittest.main()
