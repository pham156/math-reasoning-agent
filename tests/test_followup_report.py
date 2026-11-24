import json
from pathlib import Path
import tempfile
import unittest

from scripts.paper.prepare_final import followup_tables
from scripts.paper.prepare_followup import EXCLUDED, MODELS, load_export, validate_payload
from src.evaluation.checkpoint_ablations import digest


def fixture():
    completed = [{"checkpoint": model, "dataset": "development", "prompt_variant": variant,
                  "n": 100, "samples_per_problem": 1, "control_accuracy": .74,
                  "selected_accuracy": .8 if variant == "reworded" else .77}
                 for model in MODELS for variant in ("reworded", "concise")]
    completed += [{"checkpoint": model, "dataset": dataset, "prompt_variant": "original",
                   "n": 500 if dataset == "math500" else 100, "samples_per_problem": 8,
                   "control_accuracy": .85, "selected_accuracy": .86,
                   "avg_problem_tokens": 2796.03, "control_tokens": 331.65}
                  for model, dataset in (("base", "math500"), ("base", "gsm8k100"), ("step512", "gsm8k100"))]
    saved = [{"checkpoint": model, "temperature": temperature,
              "mean_diversity": {"answer_entropy_nats": .477},
              "candidate_budgets": [{"k": k, "rule": rule, "accuracy": score, "oracle_accuracy": .94}
                  for k in (1, 2, 4, 8)
                  for rule, score in (("first", .78125), ("majority_first", .8225), ("plurality", .8916666667))]}
             for model in MODELS for temperature in (.4, .7, 1.0)]
    return {"completed": completed, "saved": saved, "excluded_conditions": [EXCLUDED], "sources": {}}


class FollowupReportTests(unittest.TestCase):
    def test_tables_keep_prompt_and_order_averaged_selection_distinct(self):
        tables = followup_tables(fixture())
        self.assertEqual(tables["prompt_sensitivity"][1][-1], ["512", "74.0", "80.0", "77.0"])
        self.assertEqual(tables["selection_rules"][1][-1], ["512", "78.12", "82.25", "89.17", "94.0"])
        external = tables["external_selection"][1]
        self.assertEqual(len(external), 3)
        self.assertEqual(external[-1], ["GSM8K-100", "512", "100", "85.0", "86.0", "8.43"])
        self.assertFalse(any(row[:2] == ["MATH-500", "512"] for row in external))

    def test_rejects_partial_external_or_prompt_counts(self):
        for index in (0, 8, 10):
            data = fixture()
            data["completed"][index]["n"] -= 1
            with self.assertRaises(ValueError):
                validate_payload(data)

    def test_failed_condition_cannot_silently_enter_report(self):
        data = fixture()
        data["completed"].append({**data["completed"][-1], "dataset": "math500", "n": 480})
        with self.assertRaises(ValueError):
            validate_payload(data)
        data = fixture()
        data["excluded_conditions"] = []
        with self.assertRaises(ValueError):
            validate_payload(data)

    def test_rejects_duplicate_or_missing_conditions(self):
        data = fixture()
        data["completed"][-1] = data["completed"][-2]
        with self.assertRaises(ValueError):
            validate_payload(data)
        data = fixture()
        data["saved"].pop()
        with self.assertRaises(ValueError):
            validate_payload(data)

    def test_rejects_wrong_number_of_candidates(self):
        data = fixture()
        data["completed"][-1]["samples_per_problem"] = 1
        with self.assertRaises(ValueError):
            validate_payload(data)

    def test_cached_export_checks_source_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            source, export = Path(folder) / "source.txt", Path(folder) / "export.json"
            source.write_text("original")
            data = fixture()
            data["sources"] = {str(source): {"path": str(source), "sha256": digest(source)}}
            export.write_text(json.dumps(data))
            self.assertEqual(len(load_export(export)["completed"]), 11)
            source.write_text("changed")
            with self.assertRaises(ValueError):
                load_export(export)


if __name__ == "__main__":
    unittest.main()
