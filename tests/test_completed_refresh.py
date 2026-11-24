import json
from pathlib import Path
import tempfile
import unittest

from scripts.paper.refresh_completed import assess, measured_segments


class CompletedRefreshTests(unittest.TestCase):
    def fixture(self, root):
        reference = [{"source_index": i, "problem": str(i), "answer": str(i), "prompt_hash": str(i)} for i in range(2)]
        config = {"model_path": str(root / "model"), "source_indices": [0, 1], "mode": "greedy",
                  "num_samples": 1, "temperature": 0, "top_p": 1, "top_k": 0, "max_new_tokens": 3000,
                  "stop_ids": [151643, 151645], "prompt_hashes": ["0", "1"], "benchmark_sha256": "frozen"}
        rows = [{**r, "reference_answer": r["answer"], "runtime_seconds": 1, "input_tokens": 2,
                 "candidates": [{"correct": True, "truncated": False, "generated_tokens": 3}]} for r in reference]
        (root / "shard_0_config.json").write_text(json.dumps(config))
        self.save(root, rows)
        return reference, rows

    def save(self, root, rows):
        (root / "shard_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    def test_complete_cohort_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, _ = self.fixture(root)
            sources = {}
            status, score = assess(root, root / "model", reference, 2, sources, "frozen")
            self.assertEqual(status["status"], "complete")
            self.assertEqual(score["accuracy"], 1)
            self.assertEqual(len(sources), 2)

    def test_partial_does_not_get_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, rows = self.fixture(root)
            self.save(root, rows[:1])
            status, score = assess(root, root / "model", reference, 2, {})
            self.assertEqual(status["observed"], 1)
            self.assertEqual(status["missing_or_partial_shards"], [0])
            self.assertIsNone(score)

    def test_mismatch_fails_instead_of_becoming_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, rows = self.fixture(root)
            for field, value in (("problem", "different"), ("prompt_hash", "changed"), ("reference_answer", "99")):
                self.save(root, [{**rows[0], field: value}])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    assess(root, root / "model", reference, 2, {})
            self.save(root, rows)
            with self.assertRaises(ValueError):
                assess(root, root / "model", reference, 2, {}, "changed-frozen-hash")

    def test_no_line_across_missing_scheduled_checkpoints(self):
        curve = {str(s): {"dev300": 0.5} for s in [0, 16, 32, 128, 144, 512]}
        self.assertEqual([[s for s, _ in part] for part in measured_segments(curve, "dev300", 16)],
                         [[0, 16, 32], [128, 144], [512]])
        self.assertEqual(measured_segments(curve, "math500", 64), [])


if __name__ == "__main__":
    unittest.main()
