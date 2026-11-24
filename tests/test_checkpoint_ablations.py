import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.checkpoint_ablations import load_condition, metrics, selection, summarize, write
from src.evaluation.pilot_exploration import decoding_settings


class CheckpointAblationTests(unittest.TestCase):
    def test_default_decoding_remains_compatible(self):
        self.assertEqual(decoding_settings("greedy"), (0, 3000))
        self.assertEqual(decoding_settings("sampled"), (1, 3000))
        self.assertEqual(decoding_settings("sampled", 0.4, 512), (0.4, 512))

    def test_invalid_decoding_rejected(self):
        for mode, temp, cap in (("greedy", 0.4, 512), ("sampled", 0, 3000),
                                ("sampled", float("nan"), 3000), ("sampled", 1, 0), ("sampled", 1, 3001)):
            with self.assertRaises(ValueError):
                decoding_settings(mode, temp, cap)

    def test_vote_requires_strict_majority(self):
        candidates = [{"answer": answer} for answer in ["0", "1", "1", "1", "1", "2", "2", "2"]]
        self.assertEqual(selection(candidates)["selected_index"], 0)
        self.assertFalse(selection(candidates)["majority"])
        candidates[-1]["answer"] = "1"
        self.assertEqual(selection(candidates)["selected_index"], 1)
        self.assertTrue(selection(candidates)["majority"])

    def test_empty_answer_cannot_win(self):
        result = selection([{"answer": "2"}] + [{"answer": ""}] * 7)
        self.assertFalse(result["majority"])
        self.assertEqual(result["selected_index"], 0)

    def test_labels_do_not_affect_selection(self):
        candidates = [{"answer": "0", "correct": True}] + [{"answer": "1", "correct": False}] * 7
        before = selection(candidates)
        for candidate in candidates:
            candidate["correct"] = not candidate["correct"]
        self.assertEqual(before, selection(candidates))

    def fixture(self, root, mode="sampled", n=2):
        count = 8 if mode == "sampled" else 1
        config = {"name": "grpo", "mode": mode, "shard": 0, "model_path": str(root / "model"),
                  "source_indices": list(range(n)), "prompt_hashes": [str(i) for i in range(n)],
                  "stop_ids": [151643, 151645], "max_new_tokens": 3000, "num_samples": count,
                  "temperature": 1.0 if mode == "sampled" else 0, "top_p": 1.0, "top_k": 0,
                  "seed_rule": "100000 + source_index for each sampled group",
                  "precision": "FP32 parameters/BF16 compute", "grader": "pilot exact/Math-Verify", "sampling_batch": count}
        rows = [{"source_index": i, "prompt_hash": str(i), "problem": str(i), "reference_answer": "1",
                 "type": "Algebra", "level": "Level 3", "num_correct": count - 1,
                 "runtime_seconds": 2, "candidates": [
                     {"sample_index": j, "answer": "0" if j == 0 else "1", "correct": j > 0,
                      "generated_tokens": 10, "truncated": False, "ending_token_id": 151645}
                     for j in range(count)]} for i in range(n)]
        folder = root / mode / "grpo"
        folder.mkdir(parents=True)
        write(folder / "shard_0_config.json", config)
        (folder / "shard_0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        condition = {"key": mode, "checkpoint": "step128", "mode": mode, "temperature": config["temperature"],
                     "cap": 3000, "root": str(root), "name": "grpo", "model_path": config["model_path"], "shards": [0], "reused": True}
        cohort = [{key: row[key] for key in ("source_index", "prompt_hash", "problem", "reference_answer", "type", "level")} for row in rows]
        return condition, cohort, rows

    def test_condition_checks_model_protocol_cohort_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, cohort, rows = self.fixture(root)
            self.assertEqual(len(load_condition(condition, cohort, 2)[0]), 2)
            for key, value in (("cap", 512), ("temperature", 0.7), ("model_path", str(root / "wrong"))):
                with self.assertRaises(ValueError):
                    load_condition({**condition, key: value}, cohort, 2)
            cohort[0]["prompt_hash"] = "changed"
            with self.assertRaises(ValueError):
                load_condition(condition, cohort, 2)
            with self.assertRaises(ValueError):
                load_condition(condition, n=3)
            rows[-1]["source_index"] = rows[0]["source_index"]
            (root / "sampled/grpo/shard_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                load_condition(condition, n=2)

    def test_metrics_distinguish_sampled_first_selected_and_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, rows = self.fixture(Path(directory))
            result = metrics(rows)
            self.assertEqual(result["first_accuracy"], 0)
            self.assertEqual(result["selected_accuracy"], 1)
            self.assertEqual(result["pass_at_k"]["1"], 7 / 8)
            self.assertEqual(result["pass_at_k"]["8"], 1)
            self.assertEqual(result["fixed_by_consensus"], 2)
            self.assertEqual(result["total_generated_tokens"], 160)

    def test_complete_offline_summary_and_missing_smoke_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, cohort, _ = self.fixture(root / "raw", n=100)
            manifest = {"source_hashes": {}, "model_files": {}, "conditions": [condition],
                        "cohort": cohort, "raw_root": str(root / "raw")}
            write(root / "campaign.json", manifest)
            summarize(root, existing=True)
            result = json.loads((root / "existing_summary.json").read_text())
            self.assertEqual(result["runs"][0]["n"], 100)
            self.assertTrue((root / "existing_summary.csv").exists())
            with self.assertRaises(FileNotFoundError):
                summarize(root)
            write(root / "smoke_passed.json", {"passed": True})
            summarize(root)
            self.assertTrue((root / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
