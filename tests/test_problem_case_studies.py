import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.problem_case_studies import independent_checks, load_policy, paired_metrics, render, tex_escape, validate_annotations


def row(index=0, correct=True):
    return {"source_index": index, "prompt_hash": str(index), "problem": "Question", "reference_answer": "5",
            "type": "Algebra", "level": "Level 3", "candidates": [{"correct": correct, "answer": "5",
            "raw_solution": "We obtain the answer 5.", "generated_tokens": 10, "truncated": False}]}


class CaseStudyTests(unittest.TestCase):
    def fixture(self, root):
        for shard in range(3):
            rows = [row(i) for i in range(100*shard, 100*(shard+1))]
            config = {"mode": "greedy", "temperature": 0, "max_new_tokens": 3000, "num_samples": 1,
                      "top_p": 1, "top_k": 0, "model_path": "model", "stop_ids": [1, 2],
                      "source_indices": [r["source_index"] for r in rows], "prompt_hashes": [r["prompt_hash"] for r in rows]}
            (root / f"shard_{shard}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (root / f"shard_{shard}_config.json").write_text(json.dumps(config))

    def test_complete_source_checked_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            sources = {}
            rows, _ = load_policy(root, sources)
            self.assertEqual(len(rows), 300)
            self.assertEqual(len(sources), 6)

    def test_partial_protocol_or_prompt_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in (("temperature", .7), ("prompt_hashes", ["wrong"] * 100), ("stop_ids", [3])):
                self.fixture(root)
                path = root / "shard_1_config.json"
                config = json.loads(path.read_text())
                config[field] = value
                path.write_text(json.dumps(config))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_policy(root, {})
            self.fixture(root)
            (root / "shard_0.jsonl").write_text(json.dumps(row()))
            with self.assertRaises(ValueError):
                load_policy(root, {})

    def test_annotations_require_verbatim_quotes_and_matching_outcomes(self):
        policies = {"base": {0: row()}}
        case = {"source_index": 0, "policies": ["base"], "expected": {"base": True},
                "evidence": [{"policy": "base", "quote": "answer 5"}]}
        validate_annotations({"cases": [case]}, policies)
        for change in ({"expected": {"base": False}}, {"evidence": [{"policy": "base", "quote": "invented reasoning"}]}):
            with self.assertRaises(ValueError):
                validate_annotations({"cases": [{**case, **change}]}, policies)
        with self.assertRaises(ValueError):
            validate_annotations({"cases": [case, copy.deepcopy(case)]}, policies)

    def test_fixed_and_broken_are_paired_not_accuracy_differences(self):
        before = {0: row(0, True), 1: row(1, False)}
        after = {0: row(0, False), 1: row(1, True)}
        result = paired_metrics(after, before)
        self.assertEqual(result["fixed"], [1])
        self.assertEqual(result["broken"], [0])
        with self.assertRaises(ValueError):
            paired_metrics(after, {0: before[0]})

    def test_independent_arithmetic_and_symbolic_checks(self):
        checks = independent_checks()
        self.assertEqual(len(checks), 17)
        self.assertTrue(all(checks.values()))

    def test_escaping_is_not_model_code_execution(self):
        self.assertEqual(tex_escape("a_b & 50%"), r"a\_b \& 50\%")

    def test_export_rejects_missing_evidence_by_not_fabricating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"full_dev300": {}, "cases": [], "sources": {}, "policy_configs": {}, "selection": "none"}
            with self.assertRaisesRegex(ValueError, "exactly 17"):
                render(payload, Path(directory))
            self.assertFalse((Path(directory) / "analysis.md").exists())


if __name__ == "__main__":
    unittest.main()
