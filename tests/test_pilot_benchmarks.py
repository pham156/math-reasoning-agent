import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation import pilot_benchmarks as module
from src.evaluation.benchmarks import matharena_aime_rows


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


def raw_aime(answer_override=None):
    rows = [{"problem_idx": i, "problem": f"Problem {i}", "answer": i * 7, "problem_type": ["Algebra"]} for i in range(30, 0, -1)]
    if answer_override is not None:
        rows[0]["answer"] = answer_override
    return rows


def write_competition_run(root, benchmark, mode, name, correct_counts, samples):
    frozen = root / f"{benchmark}_frozen.json"
    directory = root / benchmark / mode / name
    directory.mkdir(parents=True)
    config = {"mode": mode, "num_samples": samples, "benchmark_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "prompt_hashes": [f"{benchmark}-{i}" for i in range(1, 31)]}
    (directory / "shard_0_config.json").write_text(json.dumps(config))
    # Correct sample 0 is a truncated, unboxed answer: lenient-correct but strict-incorrect.
    texts = {True: r"So \boxed{7}.", False: "No answer."}
    rows = [{"source_index": i, "prompt_hash": f"{benchmark}-{i}", "reference_answer": "7", "candidates": [{"correct": j < correct_counts[i - 1], "raw_solution": "We get 7 so far" if j == 0 < correct_counts[i - 1] else texts[j < correct_counts[i - 1]], "truncated": j == 0, "generated_tokens": 100} for j in range(samples)]} for i in range(1, 31)]
    (directory / "shard_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


class AimeTests(unittest.TestCase):
    def test_matharena_rows_are_validated_and_sorted(self):
        rows = matharena_aime_rows(raw_aime(), "aime2025")
        self.assertEqual([r["problem_index"] for r in rows], list(range(1, 31)))
        self.assertEqual(rows[0]["ground_truth_answer_override"], "7")
        self.assertEqual(rows[0]["type"], "AIME 2025")
        for bad in (1000, -1, 3.5, "70", True):
            with self.assertRaises(ValueError):
                matharena_aime_rows(raw_aime(bad), "aime2025")
        with self.assertRaises(ValueError):
            matharena_aime_rows(raw_aime()[:29], "aime2025")
        duplicate = raw_aime()
        duplicate[1]["problem_idx"] = duplicate[0]["problem_idx"]
        with self.assertRaises(ValueError):
            matharena_aime_rows(duplicate, "aime2025")

    def test_freeze_rejects_pilot_overlap(self):
        source = matharena_aime_rows(raw_aime(), "aime2026")
        frozen = module.freeze_aime_rows(source, "aime2026", set())
        self.assertEqual(frozen[0]["answer"], "7")
        self.assertEqual(frozen[0]["prompt"][1], {"role": "user", "content": "Problem 1"})
        with self.assertRaises(ValueError):
            module.freeze_aime_rows(source, "aime2026", {module.problem_hash("Problem  5")})

    def test_summary_pools_years_and_guards_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benchmark in module.AIME_BENCHMARKS:
                (root / f"{benchmark}_frozen.json").write_text(json.dumps({"metadata": {"shard_size": 30}, "rows": [{"source_index": i} for i in range(1, 31)]}))
                write_competition_run(root, benchmark, "sampled", "baseline", [0] * 30, 16)
                write_competition_run(root, benchmark, "sampled", "grpo", [4] * 15 + [0] * 15, 16)
            module.summarize_aime(root, ["baseline", "grpo"], ["sampled"])
            runs = {(r["benchmark"], r["name"]): r for r in json.loads((root / "aime_summary.json").read_text())["runs"]}
            grpo = runs[("aime_pooled", "grpo")]
            self.assertEqual(grpo["problems"], 60)
            self.assertAlmostEqual(grpo["mean_accuracy"], 0.125)
            self.assertEqual(list(grpo["pass_at_k"]), ["1", "4", "8", "16"])
            self.assertEqual(grpo["pass_at_k"]["16"], 0.5)
            self.assertEqual(grpo["solved_any"], 30)
            self.assertAlmostEqual(grpo["truncated_fraction"], 1 / 16)
            self.assertAlmostEqual(grpo["vs_baseline"]["mean_difference"], 0.125)
            low, high = grpo["vs_baseline"]["bootstrap_95ci"]
            self.assertTrue(0 < low < 0.125 < high)
            self.assertEqual(len(grpo["vs_baseline"]["improved"]), 30)
            self.assertNotIn("vs_baseline", runs[("aime2025", "baseline")])
            self.assertIn("pass@16", (root / "aime_summary.md").read_text())
            self.assertEqual(grpo["correct_without_boxed"], 30)
            self.assertAlmostEqual(grpo["strict_mean_accuracy"], 0.09375)
            self.assertAlmostEqual(grpo["vs_baseline_strict"]["mean_difference"], 0.09375)
            self.assertIn("| 12.5% | 9.4% |", (root / "aime_summary.md").read_text())
            write_competition_run(root, "aime2025", "sampled", "dr_grpo", [1] * 30, 8)
            write_competition_run(root, "aime2026", "sampled", "dr_grpo", [1] * 30, 8)
            with self.assertRaises(ValueError):
                module.summarize_aime(root, ["baseline", "dr_grpo"], ["sampled"])
            (root / "aime2026/sampled/grpo/shard_0.jsonl").write_text("")
            with self.assertRaises(ValueError):
                module.summarize_aime(root, ["baseline", "grpo"], ["sampled"])


if __name__ == "__main__":
    unittest.main()
