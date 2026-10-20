import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation import pilot_benchmarks as module


class BenchmarkTests(unittest.TestCase):
    def test_hash_normalizes_whitespace(self):
        self.assertEqual(module.problem_hash("a  b\n"), module.problem_hash("a b"))

    def test_summary_and_incomplete_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "toy_frozen.json"
            frozen.write_text(json.dumps({"rows": [{"source_index": i} for i in range(100)]}))
            for name in ("baseline", "grpo"):
                target = root / "toy/greedy" / name
                target.mkdir(parents=True)
                config = {"benchmark_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "prompt_hashes": [str(i) for i in range(100)]}
                (target / "shard_0_config.json").write_text(json.dumps(config))
                rows = [{"source_index": i, "prompt_hash": str(i), "type": "Algebra", "level": "Level 3", "input_tokens": 10, "runtime_seconds": 2, "candidates": [{"correct": name == "grpo", "generated_tokens": 20, "truncated": False}]} for i in range(100)]
                (target / "shard_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with patch.object(module, "BENCHMARKS", {"toy": 100}), patch.object(module, "NAMES", ("baseline", "grpo")):
                module.summarize(root)
                summary = json.loads((root / "summary.json").read_text())
                self.assertEqual(summary["runs"][1]["accuracy"], 1)
                self.assertEqual(len(summary["runs"][1]["fixed"]), 100)
                (root / "toy/greedy/grpo/shard_0.jsonl").write_text("")
                with self.assertRaises(ValueError):
                    module.summarize(root)


if __name__ == "__main__":
    unittest.main()
