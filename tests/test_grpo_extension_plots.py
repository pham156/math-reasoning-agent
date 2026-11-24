import math
import unittest

from src.evaluation.plot_grpo_extension import training_curve, trailing_mean


class ExtensionPlotTests(unittest.TestCase):
    def test_smoothing_preserves_initial_values(self):
        self.assertEqual(trailing_mean([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
        self.assertEqual(trailing_mean([], 4), [])
        with self.assertRaises(ValueError):
            trailing_mean([1], 0)

    def test_loss_requires_every_update_and_allows_negative_values(self):
        history = [{"step": i, "loss": -i / 1000} for i in range(1, 513)]
        self.assertEqual(len(training_curve(history, "loss")), 512)
        with self.assertRaises(ValueError):
            training_curve(history[:-1], "loss")
        history[0]["loss"] = math.nan
        with self.assertRaises(ValueError):
            training_curve(history, "loss")

    def test_reward_requires_rollout_observations_without_interpolation(self):
        history = [{"step": i, "reward": 0.5} for i in range(1, 512, 4)]
        self.assertEqual(len(training_curve(history, "reward")), 128)
        history.append({"step": 512, "reward": 0.5})
        with self.assertRaises(ValueError):
            training_curve(history, "reward")


if __name__ == "__main__":
    unittest.main()
