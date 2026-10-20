import unittest

from scripts.paper.plot_midterm import METHODS, paired_lr_points, pass_at_k, trailing_mean, training_points


class MidtermPlotTests(unittest.TestCase):
    def test_report_methods_are_focused(self):
        self.assertEqual(list(METHODS), ["baseline", "grpo", "dr_grpo"])

    def test_entropy_requires_all_updates_and_finite_values(self):
        history = [{"step": step, "entropy": 0.5} for step in range(1, 129)]
        self.assertEqual(len(training_points(history, "entropy")), 128)
        with self.assertRaises(ValueError):
            training_points(history[:-1], "entropy")
        history[0]["entropy"] = float("nan")
        with self.assertRaises(ValueError):
            training_points(history, "entropy")

    def test_trailing_mean_uses_available_observations(self):
        self.assertEqual(trailing_mean([1, 2, 3, 4, 5]), [1, 1.5, 2, 2.5, 3.5])
        self.assertEqual(trailing_mean([]), [])

    def test_pass_at_k_boundary_cases(self):
        for k in range(1, 9):
            self.assertEqual(pass_at_k(0, k), 0)
            self.assertEqual(pass_at_k(8, k), 1)
        self.assertEqual(pass_at_k(1, 1), 0.125)
        self.assertEqual(pass_at_k(1, 4), 0.5)
        self.assertEqual(pass_at_k(1, 8), 1)

    def test_sampled_coverage_is_monotone(self):
        for correct in range(9):
            values = [pass_at_k(correct, k) for k in range(1, 9)]
            self.assertEqual(values, sorted(values))

    def test_paired_learning_rate_values_and_completeness(self):
        summary = {"runs": [{"label": f"seed{seed}_lr{rate:.0e}", "n": 300,
                             "settings": {"seed": seed, "learning_rate": rate},
                             "accuracy": 0.7 if rate == 1e-6 else 0.75}
                            for seed in (0, 1) for rate in (1e-6, 3e-6)],
                   "two_seed_lr_summary": {"1e-06": {"mean_accuracy": 0.7}, "3e-06": {"mean_accuracy": 0.75}}}
        self.assertEqual(paired_lr_points(summary), {0: [0.7, 0.75], 1: [0.7, 0.75]})
        summary["runs"].pop()
        with self.assertRaises(ValueError):
            paired_lr_points(summary)


if __name__ == "__main__":
    unittest.main()
