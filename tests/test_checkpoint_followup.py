from itertools import permutations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.checkpoint_ablations import selection, write
from src.evaluation.checkpoint_followup import PROMPTS, input_rows, object_hash, summarize, validate_condition, validate_design, verify_smoke
from src.evaluation.saved_candidate_ablations import diversity, paired_breakdown, selection_weights, subset_metrics
from tests import test_checkpoint_ablations


def candidates(answers):
    return [{"answer": a, "correct": a == "1", "generated_tokens": 10} for a in answers]


class SavedCandidateTests(unittest.TestCase):
    def test_order_averaging_matches_exhaustive_permutations(self):
        for answers in (["0", "1", "1", "1"], ["0", "1", "1", "2"], ["", "1", "0", ""]):
            group = candidates(answers)
            expected = sum(order[selection(order)["selected_index"]]["correct"]
                           for order in permutations(group)) / 24
            actual = sum(w * c["correct"] for w, c in zip(selection_weights(group, "majority_first"), group))
            self.assertAlmostEqual(actual, expected)

    def test_plurality_ties_and_empty_answers(self):
        group = candidates(["1", "1", "0", "0", "2", ""])
        self.assertEqual(selection_weights(group, "plurality"), [.25, .25, .25, .25, 0, 0])
        self.assertEqual(selection_weights(candidates(["", ""]), "plurality"), [.5, .5])

    def test_abstention_not_inflated_accuracy(self):
        group = candidates(["1", "0"])
        result = subset_metrics(group, 2, "majority_abstain")
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["oracle_accuracy"], 1)

    def test_subset_expectation_has_no_prefix_bias(self):
        result = subset_metrics(candidates(["0"] * 4 + ["1"] * 4), 1, "first")
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["completion_tokens"], 10)
        result = subset_metrics(candidates(["0"] * 4 + ["1"] * 4), 4, "first")
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertEqual(result["completion_tokens"], 40)

    def test_selection_does_not_use_correctness(self):
        group = candidates(["1", "1", "0", ""])
        for rule in ("first", "majority_first", "plurality", "majority_abstain"):
            before = selection_weights(group, rule)
            altered = [{**c, "correct": not c["correct"]} for c in group]
            self.assertEqual(before, selection_weights(altered, rule))

    def test_diversity_measures_answer_not_token_entropy(self):
        result = diversity(candidates(["1", "1", "0", ""]))
        self.assertEqual(result["unique_nonempty_answers"], 2)
        self.assertEqual(result["empty_answer_rate"], .25)
        self.assertEqual(result["correct_candidate_mass"], .5)
        self.assertTrue(result["mixed_correctness"])

    def test_rejects_bad_budget_and_rule(self):
        with self.assertRaises(ValueError):
            subset_metrics(candidates(["1"]), 2, "first")
        with self.assertRaises(ValueError):
            selection_weights([], "first")
        with self.assertRaises(ValueError):
            selection_weights(candidates(["1"]), "oracle")

    def test_paired_category_counts(self):
        rows = [{"source_index": i, "type": "Algebra", "level": "Level 3", "candidates": candidates([answer] * 8)}
                for i, answer in enumerate(("0", "1"))]
        controls = [{**r, "candidates": candidates(["0"] * 8)} for r in rows]
        result = paired_breakdown(rows, controls)["by_type"]["Algebra"]
        self.assertEqual(result["fixed"], [1])
        self.assertEqual(result["broken"], [])


class FollowupTests(unittest.TestCase):
    def test_design_rejects_external_tuning_or_missing_conditions(self):
        models = {label: f"/scratch/gilbreth/test/{label}" for label in ("base", "step128", "step256", "step512")}
        entries = [(label, "development", variant, "greedy", 0.0, 100)
                   for label in models for variant in ("reworded", "concise")]
        entries += [(label, benchmark, "original", "sampled", .7, n)
                    for label in ("base", "step512") for benchmark, n in (("math500", 500), ("gsm8k100", 100))]
        manifest = {"models": models, "conditions": [], "tasks": []}
        for i, (label, dataset, variant, mode, temperature, n) in enumerate(entries):
            size = 100 if mode == "greedy" else 20
            manifest["conditions"].append(dict(checkpoint=label, dataset=dataset, prompt_variant=variant,
                mode=mode, temperature=temperature, n=n, cap=3000, shard_size=size,
                model_path=models[label], shards=list(range(n // size))))
            manifest["tasks"].extend(dict(condition=i, shard=shard,
                phase="prompt" if dataset == "development" else "external") for shard in range(n // size))
        validate_design(manifest)
        manifest["conditions"][-1]["temperature"] = .4
        with self.assertRaises(ValueError):
            validate_design(manifest)
        manifest["conditions"][-1]["temperature"] = .7
        manifest["tasks"].pop()
        with self.assertRaises(ValueError):
            validate_design(manifest)

    def fixture(self, root):
        condition, cohort, rows = test_checkpoint_ablations.CheckpointAblationTests().fixture(root / "raw", n=2)
        condition.update(dataset="development", prompt_variant="reworded", n=2, shard_size=2)
        source = root / "source"
        write(source / "campaign.json", {"cohort": cohort})
        manifest = {"source_campaign": str(source), "prompts": PROMPTS}
        return manifest, condition, rows

    def test_prompt_only_change_and_no_reference_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, condition, rows = self.fixture(Path(directory))
            inputs = input_rows(manifest, condition)
            self.assertEqual(inputs[0]["prompt"][0]["content"], PROMPTS["reworded"])
            self.assertEqual(inputs[0]["prompt"][1]["content"], rows[0]["problem"])
            self.assertEqual(inputs[0]["answer"], rows[0]["reference_answer"])
            self.assertNotIn("solution", inputs[0])

    def test_validation_rejects_source_or_condition_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, condition, rows = self.fixture(Path(directory))
            config_path = Path(condition["root"]) / "sampled/grpo/shard_0_config.json"
            import json
            config = json.loads(config_path.read_text())
            config.update(condition_sha256=object_hash(condition), input_sha256=object_hash(input_rows(manifest, condition)))
            write(config_path, config)
            with patch("src.evaluation.checkpoint_followup.tokenizer_for"), patch(
                    "src.evaluation.checkpoint_followup.prompt_hashes", return_value=["0", "1"]):
                self.assertEqual(len(validate_condition(manifest, condition)[0]), 2)
                config["input_sha256"] = "changed"
                write(config_path, config)
                with self.assertRaises(ValueError):
                    validate_condition(manifest, condition)

    def test_missing_or_changed_smoke_cannot_release_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                verify_smoke(root, {"smoke_conditions": []})
            write(root / "smoke_passed.json", {"passed": False, "runs": []})
            with self.assertRaises(ValueError):
                verify_smoke(root, {"smoke_conditions": []})

    def test_summary_writes_all_conditions_and_rejects_partial_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, condition, rows = self.fixture(root)
            manifest.update(raw_root=str(root / "raw"), conditions=[{**condition, "key": str(i)} for i in range(12)])
            write(root / "campaign.json", manifest)
            with patch("src.evaluation.checkpoint_followup.verify_campaign"), patch(
                    "src.evaluation.checkpoint_followup.verify_smoke"), patch(
                    "src.evaluation.checkpoint_followup.validate_condition", return_value=(rows, {})), patch(
                    "src.evaluation.checkpoint_followup.control_rows", return_value=rows), patch(
                    "src.evaluation.checkpoint_followup.ensure_scratch_path", side_effect=Path):
                summarize(root)
                import json
                result = json.loads((root / "summary.json").read_text())
                self.assertEqual(len(result["runs"]), 12)
                self.assertEqual(result["runs"][0]["fixes_vs_control"], [0, 1])
                self.assertTrue((root / "raw/selection.jsonl").is_file())
                manifest["conditions"].pop()
                write(root / "campaign.json", manifest)
                with self.assertRaises(ValueError):
                    summarize(root)


if __name__ == "__main__":
    unittest.main()
