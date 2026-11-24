import contextlib
import io
from itertools import islice
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.rl.reward_design import batch_metrics, format_score, reward_components
from src.training.rl.study_controls import BASE, profiles, updates


class RewardDesignTests(unittest.TestCase):
    def test_twenty_format_cases(self):
        examples = [(r"\boxed{2}", 1), (r"\boxed{-2}", 1), (r"\boxed{0}", 1),
                    (r"\boxed{\frac{1}{2}}", 1), (r"\boxed{25\%}", 1),
                    (r"\boxed{5,600}", 1), (r"\boxed{\{1,2\}}", 1),
                    (r"\boxed{1, 2}", 1), (r"\boxed{\sqrt{2}}", 1),
                    (r"\boxed{0.5}", 1), (r"Answer: $\boxed{3}$", 1),
                    (r"\boxed {3}", 1), (r"\boxed{(1,2)}", 1),
                    (r"\boxed{\frac{2}{\sqrt{3}}}", 1),
                    ("", 0), ("Answer: 2", 0), (r"\boxed{}", 0),
                    (r"\boxed{2", 0), (r"\boxed{1}\boxed{2}", 0),
                    (r"\fbox{2}", 0), (r"\boxed{2}\boxed{", 0),
                    (r"\boxed nonsense {2}", 0)]
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(format_score(text, lambda value: bool(value.strip())), expected)
        self.assertEqual(format_score(r"\boxed{nonsense}", lambda _: False), 0)

    def test_reward_components_and_bounds(self):
        def parse(_):
            return True
        control = reward_components(r"\boxed{2}", 1, 50, 100, "correctness", parse)
        self.assertEqual(control["reward"], 1)
        self.assertAlmostEqual(reward_components(r"\boxed{2}", 0, 50, 100, "format_bonus", parse)["reward"], 0.1)
        self.assertEqual(reward_components("Answer: 2", 1, 50, 100, "format_bonus", parse)["reward"], 1)
        for correct in (0, 1):
            short = reward_components(r"\boxed{2}", correct, 10, 100, "length_penalty", parse)
            long = reward_components(r"\boxed{2}", correct, 1000, 100, "length_penalty", parse)
            self.assertGreater(short["reward"], long["reward"])
            self.assertAlmostEqual(long["reward"], correct - 0.05)
        with self.assertRaises(ValueError):
            reward_components("", 0, -1, 100, "correctness", parse)

    def test_mixed_groups_use_correctness_not_shaped_reward(self):
        records = [{"problem": "p", **reward_components(text, 0, n, 100, "length_penalty", lambda _: True)}
                   for n, text in [(1, "wrong"), (10, "wrong"), (20, "wrong"), (30, "wrong")]]
        metrics = batch_metrics(records, 4)
        self.assertEqual(metrics["mixed_group_fraction"], 0)
        self.assertEqual(metrics["correctness_group_variance"], 0)
        self.assertGreater(metrics["total_group_variance"], 0)
        records[0]["problem"] = "different"
        with self.assertRaises(ValueError):
            batch_metrics(records, 4)


class StudyControlTests(unittest.TestCase):
    def test_studies_change_only_their_own_factor(self):
        for label, config in profiles("reward").items():
            diff = {key for key in BASE if config[key] != BASE[key]}
            self.assertTrue(diff <= {"seed", "reward_design"})
        for label, config in profiles("exploration").items():
            diff = {key for key in BASE if config[key] != BASE[key]}
            self.assertEqual(config["reward_design"], "correctness")
            self.assertTrue(diff <= ({"temperature"} if label.startswith("temperature") else {"num_generations"}))
        self.assertEqual([updates({**BASE, "num_generations": n}) for n in (2, 4, 8)], [64, 128, 256])

    def test_actual_trl_rollout_boundaries(self):
        from trl import GRPOConfig
        for n in (2, 4, 8):
            with tempfile.TemporaryDirectory() as tmp:
                c = GRPOConfig(output_dir=tmp, use_cpu=True, bf16=False, report_to="none",
                               per_device_train_batch_size=1, gradient_accumulation_steps=16,
                               generation_batch_size=16 * n, num_generations=n)
                self.assertEqual(c.steps_per_generation, 16 * n)
                self.assertEqual(c.steps_per_generation // c.gradient_accumulation_steps, n)

    def test_actual_sampler_matches_problem_exposure_across_group_sizes(self):
        from trl.trainer.utils import RepeatSampler

        order = None
        for n in (2, 4, 8):
            batch = 16 * n
            sampler = iter(RepeatSampler(list(range(512)), mini_repeat_count=n, batch_size=16,
                                        repeat_count=batch, shuffle=True, seed=0))
            generated = []
            for step in range(updates({**BASE, "num_generations": n}) * 16):
                ids = list(islice(sampler, batch))
                if step % batch == 0:
                    self.assertEqual(len(ids), batch)
                    generated.extend(ids)
            self.assertEqual(len(generated), 512 * n)
            prompts = generated[::n]
            self.assertEqual(len(set(prompts)), 512)
            if order is not None:
                self.assertEqual(prompts, order)
            order = prompts

    def test_resume_rejects_changed_reward_temperature_and_rollouts(self):
        from src.training.rl.pilot import main
        from src.core.config import data_dir, model_dir
        for extra in (["--reward_design", "format_bonus"], ["--temperature", "0.7"], ["--num_generations", "8"], ["--seed", "1"]):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                previous = {"algorithm": "grpo", "model_path": str(model_dir("Qwen2.5-Math-1.5B")), "split_dir": str(data_dir("rl_1p5b_pilot_v1")), "seed": 0, "max_steps": 128, "train_samples": 512, "eval_samples": 20, "max_completion_length": 3000, "max_prompt_length": 1024, "deterministic_training": True}
                (root / "progress.json").write_text(json.dumps({"status": "completed", "checkpoint": str(root / "checkpoint-128")}))
                (root / "run_config.json").write_text(json.dumps({"arguments": previous}))
                with patch("sys.argv", ["pilot", "--output_dir", tmp, "--eval_samples", "20", "--resume", *extra]), patch("src.training.rl.pilot.ensure_scratch_path", side_effect=lambda p, _: Path(p)), patch("src.training.rl.pilot.run") as run, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main()
                    run.assert_not_called()

    def test_unchanged_completed_legacy_run_is_noop(self):
        from src.training.rl.pilot import main
        from src.core.config import data_dir, model_dir
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = {"algorithm": "grpo", "model_path": str(model_dir("Qwen2.5-Math-1.5B")), "split_dir": str(data_dir("rl_1p5b_pilot_v1")), "seed": 0, "max_steps": 128, "train_samples": 512, "eval_samples": 20, "max_completion_length": 3000, "max_prompt_length": 1024, "deterministic_training": True}
            (root / "progress.json").write_text(json.dumps({"status": "completed", "checkpoint": "unused"}))
            (root / "run_config.json").write_text(json.dumps({"arguments": previous}))
            with patch("sys.argv", ["pilot", "--output_dir", tmp, "--eval_samples", "20", "--resume"]), patch("src.training.rl.pilot.ensure_scratch_path", side_effect=lambda p, _: Path(p)), patch("src.training.rl.pilot.run") as run, contextlib.redirect_stdout(io.StringIO()):
                main()
                run.assert_not_called()


class StudySummaryTests(unittest.TestCase):
    def test_reward_summary_and_changed_checkpoint_rejection(self):
        from src.evaluation.summarize_grpo_studies import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for label, config in profiles("reward").items():
                run, evaluation = root / "runs" / label, root / "evaluations" / label
                (run / "final_model").mkdir(parents=True)
                def write(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
                write(run / "final_model/config.json", {})
                write(run / "smoke_result.json", {"arguments": {**config, "algorithm": "grpo", "train_samples": 512}, "global_step": 128, "policy_update_observed": True})
                write(run / "training_args.json", {"epsilon": 0.2, "epsilon_high": 0.2, "num_generations": 4,
                      "temperature": 1.0, "learning_rate": 3e-6, "beta": 0.0, "loss_type": "grpo",
                      "scale_rewards": "group", "importance_sampling_level": "token", "lr_scheduler_type": "constant",
                      "gradient_accumulation_steps": 16, "max_completion_length": 3000, "generation_batch_size": 64, "max_steps": 128})
                write(run / "dataset_manifest.json", {"training": list(range(512))})
                write(run / "train_history.json", {"log_history": [{"step": 1 + i * 4, "reward": 0.5, "entropy": 0.5, "completions/mean_length": 10} for i in range(32)]})
                records = [{"problem": str(i), **reward_components("Answer: 2", i % 2, 10, 3000, config["reward_design"], lambda _: True)} for i in range(512) for _ in range(4)]
                (run / "rollouts.jsonl").write_text("\n".join(json.dumps(v) for v in records))
                for mode, n, shard_size, candidates in (("greedy", 300, 100, 1), ("sampled", 100, 20, 8)):
                    rows = [{"source_index": i, "prompt_hash": str(i), "input_tokens": 10, "runtime_seconds": 1,
                             "candidates": [{"correct": bool(i % 2), "truncated": False, "generated_tokens": 10} for _ in range(candidates)]} for i in range(n)]
                    for shard in range(n // shard_size):
                        folder = evaluation / mode / "grpo"
                        write(folder / f"shard_{shard}_config.json", {"model_path": str(run / "final_model"), "max_new_tokens": 3000, "temperature": 0 if mode == "greedy" else 1.0})
                        (folder / f"shard_{shard}.jsonl").write_text("\n".join(json.dumps(v) for v in rows[shard * shard_size:(shard + 1) * shard_size]))
                runs.append({"label": label, "settings": config, "run_root": str(run), "greedy_root": str(evaluation), "sampled_root": str(evaluation), "reused": label.startswith("correctness"), "training_jobs": []})
            write(root / "submission.json", {"study": "reward", "runs": runs})
            with patch("sys.argv", ["summary", tmp]), patch("src.evaluation.summarize_grpo_studies.ensure_scratch_path", side_effect=lambda p: Path(p)), patch("src.evaluation.summarize_grpo_studies.validate_gate"), contextlib.redirect_stdout(io.StringIO()):
                main()
                result = json.loads((root / "summary.json").read_text())
                self.assertEqual(len(result["runs"]), 6)
                self.assertEqual(result["seed_aggregates"]["length_penalty"]["n_seeds"], 2)
                self.assertEqual(result["runs"][0]["training"]["completion_tokens"], 20480)
                cfg = Path(runs[-1]["sampled_root"]) / "sampled/grpo/shard_0_config.json"
                value = json.loads(cfg.read_text())
                value["model_path"] = "wrong-checkpoint"
                write(cfg, value)
                with self.assertRaises(ValueError):
                    main()


if __name__ == "__main__":
    unittest.main()
