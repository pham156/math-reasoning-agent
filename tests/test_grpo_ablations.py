from collections import defaultdict
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.training.rl.ablation_controls import CASES, CONTROL, flags, settings
from src.training.rl.methods import check_method_config, method_settings


class AblationControlTests(unittest.TestCase):
    def test_variants_change_only_intended_factors(self):
        self.assertEqual(len(CASES), 6)
        for label in CASES:
            value = settings(label)
            self.assertEqual(len(flags(label)), 10)
            if label.startswith("kl_"):
                self.assertEqual({k for k in CONTROL if value[k] != CONTROL[k]}, {"kl_beta"})
            if label.startswith("clip_"):
                self.assertEqual(value["clip_lower"], value["clip_upper"])
                self.assertEqual({k for k in CONTROL if value[k] != CONTROL[k]}, {"clip_lower", "clip_upper"})
        self.assertEqual(settings("seed1_lr3e-06")["seed"], 1)

    def test_explicit_grpo_bounds_preserve_objective(self):
        profile = method_settings("grpo", 0.1, 0.3)
        self.assertEqual(profile["loss_type"], "grpo")
        check_method_config("grpo", SimpleNamespace(**profile), 0.1, 0.3)
        self.assertEqual(method_settings("grpo")["epsilon"], 0.2)
        for value in (0, -0.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                method_settings("grpo", value, 0.2)
        with self.assertRaises(ValueError):
            method_settings("gspo", 0.1, 0.1)

    def test_resume_rejects_changed_penalty_or_bounds(self):
        from src.training.rl.pilot import main
        from src.core.config import data_dir, model_dir

        for extra in (["--kl_beta", "0.01"], ["--clip_lower", "0.1", "--clip_upper", "0.1"]):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                previous = {"algorithm": "grpo", "model_path": str(model_dir("Qwen2.5-Math-1.5B")), "split_dir": str(data_dir("rl_1p5b_pilot_v1")), "seed": 0, "max_steps": 128, "train_samples": 512, "eval_samples": 20, "max_completion_length": 3000, "max_prompt_length": 1024, "deterministic_training": True}
                (root / "progress.json").write_text(json.dumps({"status": "paused", "checkpoint": str(root / "checkpoint-4")}))
                (root / "run_config.json").write_text(json.dumps({"arguments": previous}))
                argv = ["pilot", "--output_dir", str(root), "--eval_samples", "20", "--resume", *extra]
                with patch("sys.argv", argv), patch("src.training.rl.pilot.ensure_scratch_path", side_effect=lambda p, _: Path(p)), patch("src.training.rl.pilot.run") as run, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main()
                    run.assert_not_called()


def trainer_loss(current, beta=0.0, clip=0.2, advantages=None):
    import torch
    from src.training.rl.smoke import smoke_trainer_class

    trainer = object.__new__(smoke_trainer_class())
    trainer._get_per_token_logps_and_entropies = lambda *a, **kw: (current, torch.zeros_like(current))
    trainer.top_entropy_quantile, trainer.beta, trainer.use_vllm = 1.0, beta, False
    trainer.importance_sampling_level, trainer.loss_type = "token", "grpo"
    trainer.epsilon_low = trainer.epsilon_high = clip
    trainer.current_gradient_accumulation_steps = 1
    trainer.args = SimpleNamespace(delta=None, steps_per_generation=4)
    trainer.accelerator = SimpleNamespace(num_processes=1, gather=lambda x: x)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.model = SimpleNamespace(training=True)
    inputs = {"prompt_ids": torch.zeros(2, 1, dtype=torch.long), "prompt_mask": torch.ones(2, 1), "completion_ids": torch.zeros(2, 1, dtype=torch.long), "completion_mask": torch.ones(2, 1), "old_per_token_logps": torch.zeros_like(current), "ref_per_token_logps": torch.zeros_like(current), "advantages": advantages if advantages is not None else torch.zeros(2, dtype=current.dtype), "num_items_in_batch": torch.tensor(2.0)}
    return trainer._compute_loss(SimpleNamespace(training=True), inputs), trainer


class ActualTrainerControlTests(unittest.TestCase):
    def test_kl_penalty_loss_and_gradient(self):
        import torch
        current = torch.tensor([[0.2], [-0.3]], dtype=torch.float64, requires_grad=True)
        actual, trainer = trainer_loss(current, beta=0.01)
        independent = current.detach().clone().requires_grad_(True)
        expected = 0.01 * (torch.exp(-independent) + independent - 1).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, current)[0], torch.autograd.grad(expected, independent)[0])
        self.assertGreater(trainer._metrics["train"]["kl"][0], 0)

    def test_clipping_bounds_change_actual_gradient(self):
        import torch
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
        gradients = []
        for clip in (0.1, 0.3):
            current = torch.log(torch.tensor([[1.25], [0.75]], dtype=torch.float64)).requires_grad_()
            loss, trainer = trainer_loss(current, clip=clip, advantages=advantages)
            gradients.append(torch.autograd.grad(loss, current)[0])
            self.assertEqual(trainer._metrics["train"]["clip_ratio/region_mean"][0], 1 if clip == 0.1 else 0)
        self.assertEqual(float(gradients[0].abs().sum()), 0)
        self.assertGreater(float(gradients[1].abs().sum()), 0)

    def test_exact_fixed_prefix_divergence(self):
        import torch
        from src.evaluation.grpo_drift import categorical_kl
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(categorical_kl(logits, logits), torch.zeros(1))
        self.assertGreater(float(categorical_kl(logits, logits.flip(-1))), 0)


class AblationSummaryTests(unittest.TestCase):
    def test_complete_grid_summary_and_changed_probe_guard(self):
        from src.evaluation.summarize_grpo_ablations import main
        from src.training.grpo.train import write_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "probes.json"
            probe.write_text("{}")
            digest = hashlib.sha256(probe.read_bytes()).hexdigest()
            configs = {"seed0_lr1e-06": CONTROL, "seed0_lr3e-06": {**CONTROL, "learning_rate": 3e-6}, **{label: settings(label) for label in CASES}}
            runs = []
            for label, config in configs.items():
                run, evaluation = root / "runs" / label, root / "eval" / label
                training = {"learning_rate": config["learning_rate"], "beta": config["kl_beta"], "epsilon": config["clip_lower"], "epsilon_high": config["clip_upper"], "num_generations": 4, "temperature": 1.0, "max_completion_length": 3000, "loss_type": "grpo", "scale_rewards": "group", "importance_sampling_level": "token", "lr_scheduler_type": "constant", "warmup_steps": 0}
                write_json(run / "training_args.json", training)
                write_json(run / "dataset_manifest.json", {"training": [1, 2]})
                write_json(run / "smoke_result.json", {"arguments": config, "global_step": 128, "peak_gpu_allocated_bytes": 100})
                log = {"step": 1, "reward": 0.5, "completions/mean_length": 10, "clip_ratio/region_mean": 0.0}
                if config["kl_beta"]:
                    log["kl"] = 0.01
                write_json(run / "train_history.json", {"log_history": [log]})
                write_json(root / "diagnostics" / (label + ".json"), {"probe_sha256": digest, "model_path": str(run / "final_model"), "mean_kl": 0.1, "rows": [{}] * 8})
                rows = [{"source_index": i, "prompt_hash": str(i), "runtime_seconds": 1, "candidates": [{"correct": i < (210 if config["learning_rate"] == 3e-6 else 200), "truncated": False, "generated_tokens": 10}]} for i in range(300)]
                for shard in range(3):
                    target = evaluation / "greedy/grpo" / f"shard_{shard}.jsonl"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("\n".join(json.dumps(row) for row in rows[shard * 100:(shard + 1) * 100]))
                runs.append({"label": label, "settings": config, "run_root": str(run), "evaluation_root": str(evaluation), "reused": label.startswith("seed0"), "training_jobs": []})
            write_json(root / "submission.json", {"probe_sha256": digest, "runs": runs})
            with patch("sys.argv", ["summary", str(root)]), patch("src.evaluation.summarize_grpo_ablations.ensure_scratch_path", side_effect=lambda p: Path(p)), patch("src.evaluation.summarize_grpo_ablations.validate_gate"), contextlib.redirect_stdout(io.StringIO()):
                main()
                summary = json.loads((root / "summary.json").read_text())
                self.assertEqual(len(summary["runs"]), 8)
                self.assertEqual(len(summary["lr_seed_pairs"]), 2)
                self.assertAlmostEqual(summary["lr_seed_pairs"][1]["paired_delta"], 10 / 300)
                probe.write_text("changed")
                with self.assertRaises(ValueError):
                    main()


if __name__ == "__main__":
    unittest.main()
