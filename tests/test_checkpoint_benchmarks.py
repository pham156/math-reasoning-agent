import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.checkpoint_benchmarks import benchmark_rows, compare_replay, summarize


class CheckpointBenchmarkTests(unittest.TestCase):
    def test_replay_must_match_entire_training_prefix(self):
        original = [{"problem": str(i), "reward": 1} for i in range(4096)]
        compare_replay(original, original[:3072], 192)
        with self.assertRaises(ValueError):
            compare_replay(original, original[:3071], 192)
        changed = list(original[:3072])
        changed[-1] = {"problem": "different", "reward": 1}
        with self.assertRaises(ValueError):
            compare_replay(original, changed, 192)

    def test_benchmark_rejects_incomplete_or_changed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({"rows": [{"source_index": i} for i in range(100)]}))
            folder = root / "eval/greedy/grpo"
            folder.mkdir(parents=True)
            model = root / "model"
            config = {"benchmark_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                      "model_path": str(model), "max_new_tokens": 3000, "temperature": 0,
                      "prompt_hashes": [str(i) for i in range(100)]}
            (folder / "shard_0_config.json").write_text(json.dumps(config))
            rows = [{"source_index": i, "prompt_hash": str(i), "candidates": [{"correct": True}]} for i in range(100)]
            path = folder / "shard_0.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(len(benchmark_rows(root / "eval", "grpo", model, frozen)), 100)
            with self.assertRaises(ValueError):
                benchmark_rows(root / "eval", "grpo", root / "wrong", frozen)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
            with self.assertRaises(ValueError):
                benchmark_rows(root / "eval", "grpo", model, frozen)

    def test_summary_combines_reused_baseline_and_all_three_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            source.mkdir()
            frozen = source / "math500_frozen.json"
            frozen.write_text(json.dumps({"rows": [{"source_index": i} for i in range(100)]}))
            models = {f"step{step}": str(root / f"model{step}") for step in (128, 256, 512)}
            manifest = {"benchmark_source": str(source), "baseline_model": str(root / "base"), "models": models}
            (root / "submission.json").write_text(json.dumps(manifest))
            for label, model in [("base_no_rl", manifest["baseline_model"]), *models.items()]:
                folder = source / "math500/greedy/baseline" if label == "base_no_rl" else root / "evaluations" / label / "math500/greedy/grpo"
                folder.mkdir(parents=True)
                config = {"benchmark_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                          "model_path": model, "max_new_tokens": 3000, "temperature": 0,
                          "prompt_hashes": [str(i) for i in range(100)]}
                (folder / "shard_0_config.json").write_text(json.dumps(config))
                rows = [{"source_index": i, "prompt_hash": str(i), "type": "Algebra", "level": "Level 3",
                         "input_tokens": 5, "runtime_seconds": 1,
                         "candidates": [{"correct": label != "base_no_rl", "truncated": False, "generated_tokens": 10}]} for i in range(100)]
                (folder / "shard_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with patch("src.evaluation.checkpoint_benchmarks.BENCHMARKS", {"math500": 100}), patch("builtins.print"):
                summarize(root)
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual(result["runs"][0]["accuracy"], 0)
            self.assertEqual(result["runs"][2]["model"], "step256")
            self.assertEqual(len(result["runs"][2]["fixed_vs_base"]), 100)


if __name__ == "__main__":
    unittest.main()
