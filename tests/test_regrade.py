import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.regrade import markdown, regrade


UNFINISHED = "Let me compute.\nWe get 70 so far, then subtract"
BOXED = r"So the answer is \boxed{70}."
HEADING = "#### Step 1\nWork.\n" + BOXED


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def candidate(text, correct, truncated=False):
    return {"raw_solution": text, "correct": correct, "truncated": truncated}


class RegradeTests(unittest.TestCase):
    def test_exploration_shards_and_baseline_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "sampled/baseline/shard_0.jsonl", [
                {"source_index": 1, "reference_answer": "70", "candidates": [candidate(UNFINISHED, True, True), candidate("no", False)]},
                {"source_index": 2, "reference_answer": "70", "candidates": [candidate(UNFINISHED, True, True), candidate("no", False)]},
            ])
            write_jsonl(root / "sampled/grpo/shard_0.jsonl", [
                {"source_index": 1, "reference_answer": "70", "candidates": [candidate(BOXED, True), candidate(BOXED, True)]},
                {"source_index": 2, "reference_answer": "70", "candidates": [candidate(r"\boxed{3} or \boxed{70}", True), candidate("no", False)]},
            ])
            write_jsonl(root / "sampled/notes.jsonl", [{"step": 1}])
            write_jsonl(root / "sampled/rollouts.jsonl", [{"reference_answer": "7", "raw_solution": "x"}])
            (root / "sampled/broken.jsonl").write_text("{not json\n")
            report = regrade(root)
            base, grpo = report["groups"]["sampled/baseline"], report["groups"]["sampled/grpo"]
            self.assertEqual((base["samples"], base["lenient_correct"], base["strict_correct"], base["lenient_only"]), (4, 2, 0, 2))
            self.assertEqual((base["truncated"], base["truncated_lenient_correct"], base["truncated_strict_correct"]), (2, 2, 0))
            self.assertEqual((grpo["lenient_correct"], grpo["strict_correct"], grpo["multiple_distinct_boxes"]), (3, 3, 1))
            self.assertEqual(grpo["vs_baseline"], {"lenient_change": 0.25, "strict_change": 0.75})
            self.assertNotIn("vs_baseline", base)
            self.assertEqual(report["skipped_files"], ["sampled/broken.jsonl", "sampled/notes.jsonl", "sampled/rollouts.jsonl"])
            self.assertIn("+25.0% → +75.0%", markdown(report))

    def test_pilot_dev_rows_and_stored_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Stored False for HEADING reproduces the pre-fix '####' bug; stored True for "no" is unexplained drift.
            write_jsonl(root / "grpo.jsonl", [
                {"source_index": 1, "reference_answer": "70", **candidate(HEADING, False)},
                {"source_index": 2, "reference_answer": "70", **candidate("no", True)},
            ])
            write_jsonl(root / "baseline.jsonl", [{"source_index": 1, "reference_answer": "70", **candidate("no", False)}])
            report = regrade(root)
            grpo = report["groups"]["grpo"]
            self.assertEqual((grpo["stored_correct"], grpo["lenient_correct"], grpo["strict_correct"]), (1, 1, 1))
            self.assertEqual((grpo["stored_mismatch_hash_fix"], grpo["stored_mismatch_unexplained"]), (1, 1))
            self.assertEqual(grpo["vs_baseline"], "skipped: problems or samples per problem differ from baseline")


if __name__ == "__main__":
    unittest.main()
