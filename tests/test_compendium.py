import unittest
import json
from pathlib import Path

from scripts.paper.prepare_compendium import Ledger, audit_sample_count, case_study_appendix, completion_metrics, escape, longtable, numeric_details, reference_only, summarize_training


class CompendiumTests(unittest.TestCase):
    def test_case_studies_integrated_without_second_document(self):
        source = Path(__file__).resolve().parents[1] / "docs/problem_case_studies/main.tex"
        if not source.exists():
            self.skipTest("Reviewed case-study source unavailable")
        text = case_study_appendix(source)
        self.assertEqual(text.count(r"\clearpage"), 3)
        self.assertNotIn(r"\section*{Case ", text)
        self.assertNotIn(r"\documentclass", text)
        self.assertNotIn(r"\begin{document}", text)
        self.assertNotIn(r"\end{document}", text)
        self.assertIn("{figures/case_matrix.pdf}", text)
        self.assertLess(len(text), 11000)
        self.assertIn("A genuine algebraic correction", text)

    def test_completion_errors_and_truncations_overlap(self):
        rows = [{"correct": c, "truncated": t, "generated_tokens": 20} for c, t in
                ((True, True), (False, True), (True, False), (False, False))]
        m = completion_metrics(rows)
        self.assertEqual((m["wrong"], m["truncated"], m["truncated_correct"], m["truncated_wrong"]), (2, 2, 1, 1))
        self.assertEqual(m["nontruncated_accuracy"], .5)
        self.assertIsNone(completion_metrics(rows[:2])["nontruncated_accuracy"])
        with self.assertRaises(ValueError):
            completion_metrics([])

    def test_tex_escaping(self):
        text = escape(r"a_b & 50% {x} #1 $2 \box")
        for fragment in (r"\_", r"\&", r"\%", r"\{", r"\#", r"\$", r"\textbackslash{}"):
            self.assertIn(fragment, text)

    def test_complete_conditions_only_and_duplicate_guard(self):
        ledger = Ledger()
        ledger.add("Study", "Control", "dev300", {"n": 300, "accuracy": 0.45}, "source", 300)
        self.assertEqual(ledger.records[0]["id"], "R0001")
        with self.assertRaises(ValueError):
            ledger.add("Study", "Partial", "dev300", {"n": 299}, "source", 300)
        with self.assertRaises(ValueError):
            ledger.add("Study", "Failed", "dev300", {"n": 300, "status": "failed"}, "source", 300)
        with self.assertRaises(ValueError):
            ledger.add("Study", "Control", "dev300", {"n": 300}, "source", 300)

    def test_details_preserve_negative_results_and_count_ids(self):
        values = dict(numeric_details({"accuracy": .4, "loss": -.2, "fixed": [1, 2], "broken": [],
                                       "ci95": [-.1, .2], "training_history": [{"loss": 99}]}))
        self.assertEqual(values["loss"], "-0.2")
        self.assertEqual(values["fixed.count"], "2")
        self.assertEqual(values["broken.count"], "0")
        self.assertEqual(values["ci95"], "[-0.1, 0.2]")
        self.assertNotIn("training_history[0].loss", values)

    def test_tables_repeat_headings_and_escape_categories(self):
        text = longtable(["Category", "Accuracy"], [["Counting & Probability", "50.0"]])
        self.assertIn(r"\endfirsthead", text)
        self.assertIn(r"Counting \& Probability", text)

    def test_fixed_reference_does_not_choose_favorable_seed(self):
        runs = [{"settings": {"seed": 0}, "accuracy": .5}, {"settings": {"seed": 1}, "accuracy": .9}]
        self.assertEqual(reference_only(runs), runs[:1])

    def test_audit_counts_responses_not_problems(self):
        self.assertEqual(audit_sample_count("run/aime/sampled/grpo"), 480)
        self.assertEqual(audit_sample_count("run/math500/sampled/grpo"), 4000)
        self.assertEqual(audit_sample_count("checkpoint_ablations/test/greedy/grpo"), 100)
        self.assertIsNone(audit_sample_count("run/smoke/test"))

    def test_atlas_groups_comparisons_without_database_dump(self):
        from scripts.paper.compendium_layout import render
        path = Path(__file__).resolve().parents[1] / "docs/results_compendium/results_ledger.json"
        if not path.exists():
            self.skipTest("Local completed-evidence archive unavailable")
        archive = json.loads(path.read_text())
        records = archive["records"]
        content = render(records, archive["setup"])
        self.assertEqual(content.count(r"\caption{"), 62)
        self.assertEqual(content.count(r"\captionof{figure}"), 3)
        self.assertLess(len(content), 140000)
        for unwanted in ("Metric path", "R0001", "Source Index", "Complete Aggregate Metric Details", "/scratch/", "seconds", "Seconds"):
            self.assertNotIn(unwanted, content)
        for section in ("Optimizer Ablations", "AIME Transfer", "Supervised and Rejection", "Candidate Selection"):
            self.assertIn(section, content)
        self.assertIn("71.20", content)
        for metric in ("Loss SD", "Gradient norm", "Train KL", "Clip fraction", "Correctness variance", "Response tokens"):
            self.assertIn(metric, content)
        self.assertNotIn("Retrospective selection at T=", content)
        for record in records:
            if record["section"] == "Training duration" or (record["section"] == "Initial controls" and record["label"] in ("grpo", "dr_grpo", "dapo_style", "gspo")):
                loss = record["metrics"]["training_diagnostics"]["loss"]
                self.assertGreater(loss["observations"], 0)
                self.assertIn(f"{loss['mean']:.3e}", content)
        self.assertLess(content.index("Experimental Setup"), content.index("Benchmark Results"))
        self.assertLess(content.index("Datasets, Development Split"), content.index("Benchmark Results"))
        self.assertLess(content.index("Algorithms and Training Parameters"), content.index("Benchmark Results"))
        self.assertEqual(archive["setup"]["overlaps"], {"train_dev": 0, "train_math500": 0, "dev_math500": 0})
        self.assertFalse(archive["setup"]["precision"]["peft"])
        self.assertIn("no LoRA adapter", content)
        self.assertIn("From Rollouts to Parameter Updates", content)
        self.assertIn("Prompt labels are $-100$", content)
        self.assertIn("Dev100 is contained within Dev300", content)
        for addition in ("DAPO-style", "GSPO", "Zero-variance groups", "All wrong", "Inference Subgroup Ablations", "Extraction and Grading Diagnostics", "Checkpoint and Restart Validation"):
            self.assertIn(addition, content)
        self.assertEqual(len(archive["setup"]["restart_checks"]), 5)
        self.assertTrue(all(p["passed"] for p in archive["setup"]["restart_checks"]))

    def test_supplement_labels_and_full_breakdown_coverage(self):
        from scripts.paper.compendium_supplement import audit_label, render_supplement
        path = Path(__file__).resolve().parents[1] / "docs/results_compendium/results_ledger.json"
        if not path.exists():
            self.skipTest("Local evidence unavailable")
        archive = json.loads(path.read_text())
        tables = []
        render_supplement(archive["records"], archive["setup"], lambda _: None, lambda _: None,
                          lambda caption, headers, rows, widths: tables.append((caption, rows)))
        self.assertEqual(sum(len(rows) for caption, rows in tables if "subgroup metrics" in caption), 120)
        self.assertEqual(sum(len(rows) for caption, rows in tables if "extraction audit" in caption), 123)
        self.assertEqual(audit_label("controls_exploration_20261014/sampled/dapo_style"), "DAPO-style / sampled")
        self.assertEqual(audit_label("checkpoint_followup_20261115/base_math500_original_sampled/sampled/baseline"),
                         "No RL MATH-500 original prompt sampled")
        with self.assertRaises(ValueError):
            audit_label("unknown/foo/bar")

    def test_training_statistics_use_observed_prefix_not_zero_fill(self):
        history = [{"step": 0, "loss": 100}, {"step": 1, "loss": -.2, "entropy": .8},
                   {"step": 2, "loss": .1}, {"step": 3, "loss": .3, "entropy": .4},
                   {"step": 4, "loss": .6}, {"step": 5, "loss": 900}]
        stats = summarize_training(history, 4)
        self.assertAlmostEqual(stats["loss"]["mean"], .2)
        self.assertEqual(stats["loss"]["first_quarter"], -.2)
        self.assertEqual(stats["loss"]["last_quarter"], .6)
        self.assertEqual(stats["entropy"]["observations"], 2)
        self.assertAlmostEqual(stats["entropy"]["mean"], .6)
        self.assertNotIn("kl", stats)
        with self.assertRaises(ValueError):
            summarize_training([{"step": 1, "loss": .1}, {"step": 1, "loss": .2}], 4)
        with self.assertRaises(ValueError):
            summarize_training([{"step": 1, "loss": float("nan")}], 4)


if __name__ == "__main__":
    unittest.main()
