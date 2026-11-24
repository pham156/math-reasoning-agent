import unittest

from scripts.paper.prepare_final import CAPTIONS, collect, inference_tables, pct, table, validate_latest


class FinalReportTests(unittest.TestCase):
    def test_missing_results_are_not_zero(self):
        self.assertEqual(pct(None), "---")
        self.assertEqual(pct(0), "0.0")
        self.assertEqual(pct(0.674), "67.4")

    def test_table_preserves_values_and_ieee_environment(self):
        rendered = table(["Method", "Accuracy"], [["GRPO", "67.4"]], "Caption", "methods", True)
        self.assertIn(r"\begin{table*}", rendered)
        self.assertIn(r"\label{tab:methods}", rendered)
        self.assertIn("GRPO & 67.4", rendered)
        self.assertIn(r"\end{table*}", rendered)

    def test_focused_studies_have_captions_without_seed_comparison(self):
        self.assertEqual(set(CAPTIONS), {"methods", "rates", "regularization",
                                        "rewards", "exploration", "duration", "retention"})

    def test_missing_evidence_fails_instead_of_inventing_results(self):
        from pathlib import Path
        from unittest.mock import patch

        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                collect(Path("/missing-evidence"))

    def latest_fixture(self):
        latest = {"runs": [{"benchmark": benchmark, "model": model, "n": n}
                           for benchmark, n in (("math500", 500), ("gsm8k100", 100))
                           for model in ("base_no_rl", "step128", "step256", "step512")]}
        proofs = {step: {"step": step, "rollouts_identical": True, "training_metrics_matched": True,
                         "development": {"n": 100}} for step in (192, 256)}
        return latest, proofs

    def test_latest_requires_all_eight_benchmark_model_pairs(self):
        latest, proofs = self.latest_fixture()
        validate_latest(latest, proofs)
        latest["runs"][-1] = latest["runs"][-2]
        with self.assertRaises(ValueError):
            validate_latest(latest, proofs)

    def test_latest_rejects_partial_sample_counts(self):
        latest, proofs = self.latest_fixture()
        latest["runs"][0]["n"] = 499
        with self.assertRaises(ValueError):
            validate_latest(latest, proofs)

    def test_latest_requires_both_replay_proofs(self):
        for step in (192, 256):
            latest, proofs = self.latest_fixture()
            proofs[step]["training_metrics_matched"] = False
            with self.assertRaises(ValueError):
                validate_latest(latest, proofs)

    def test_extra_point_cannot_use_a_different_size_subset(self):
        latest, proofs = self.latest_fixture()
        proofs[192]["development"]["n"] = 300
        with self.assertRaises(ValueError):
            validate_latest(latest, proofs)

    def inference_fixture(self):
        settings = [("greedy", 0.0, cap) for cap in (512, 1024, 2048, 3000)] + [("sampled", t, 3000) for t in (0.4, 0.7, 1.0)]
        return [{"checkpoint": model, "mode": mode, "temperature": t, "cap": cap,
                 "n": 100, "samples_per_problem": 8 if mode == "sampled" else 1,
                 "first_accuracy": 0.74, "pass_at_1": 0.75375, "selected_accuracy": 0.84,
                 "pass_at_8": 0.92, "majority_rate": 0.82, "avg_problem_tokens": 4976.01}
                for model in ("base", "step128", "step256", "step512") for mode, t, cap in settings]

    def test_inference_tables_preserve_distinct_metrics(self):
        tables = inference_tables(self.inference_fixture())
        rows = tables["inference_temperature"][1]
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[-1][2:6], ["75.38", "84.0", "92.0", "82.0"])
        self.assertEqual(len(tables["inference_budget"][1]), 4)

    def test_inference_tables_reject_incomplete_or_duplicate_conditions(self):
        runs = self.inference_fixture()
        with self.assertRaises(ValueError):
            inference_tables(runs[:-1])
        runs[-1] = runs[-2]
        with self.assertRaises(ValueError):
            inference_tables(runs)

    def test_inference_tables_reject_different_cohorts_or_candidate_counts(self):
        for key, value in (("n", 99), ("samples_per_problem", 1)):
            runs = self.inference_fixture()
            runs[-1][key] = value
            with self.assertRaises(ValueError):
                inference_tables(runs)


if __name__ == "__main__":
    unittest.main()
