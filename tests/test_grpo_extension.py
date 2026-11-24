import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.rl.extend import fork_run, validate_extension_gate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        patcher = patch("src.training.rl.extend.OUTPUTS", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        checkpoint = self.source / "checkpoints/checkpoint-8"
        self.args = {"algorithm": "grpo", "learning_rate": 3e-6, "seed": 0, "max_steps": 8, "train_samples": 32}
        write(self.source / "run_config.json", {"arguments": self.args})
        write(self.source / "progress.json", {"global_step": 8, "status": "completed", "checkpoint": str(checkpoint)})
        write(self.source / "smoke_result.json", {"global_step": 8, "policy_update_observed": True})
        write(self.source / "training_args.json", {"lr_scheduler_type": "constant", "learning_rate": 3e-6})
        for name in ("baseline.json", "train_history.json", "dataset_manifest.json"):
            write(self.source / name, {})
        write(checkpoint / "trainer_state.json", {"global_step": 8})
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
            (checkpoint / name).write_bytes(b"checkpoint")
        (self.source / "rollouts.jsonl").write_text("{}\n" * 128)

    def test_fork_retains_optimizer_and_copies_appendable_logs(self):
        original = (self.source / "run_config.json").read_bytes()
        fork_run(self.source, self.target, 16, 2, 1)
        config = json.loads((self.target / "run_config.json").read_text())["arguments"]
        self.assertEqual((config["max_steps"], config["training_passes"], config["checkpoint_limit"]), (16, 2, 1))
        left = self.source / "checkpoints/checkpoint-8/optimizer.pt"
        right = self.target / "checkpoints/checkpoint-8/optimizer.pt"
        self.assertEqual(left.stat().st_ino, right.stat().st_ino)
        with (self.target / "rollouts.jsonl").open("a") as stream:
            stream.write("new\n")
        self.assertEqual(len((self.source / "rollouts.jsonl").read_text().splitlines()), 128)
        self.assertEqual((self.source / "run_config.json").read_bytes(), original)
        with self.assertRaises(ValueError):
            fork_run(self.source, self.target, 16, 2, 1)

    def test_invalid_budget_optimizer_or_objective_rejected(self):
        with self.assertRaises(ValueError):
            fork_run(self.source, self.target, 12, 2, 1)
        (self.source / "checkpoints/checkpoint-8/optimizer.pt").unlink()
        with self.assertRaises(ValueError):
            fork_run(self.source, self.target, 16, 2, 1)
        self.assertFalse(self.target.exists())
        self.args["learning_rate"] = 1e-6
        write(self.source / "run_config.json", {"arguments": self.args})
        with self.assertRaises(ValueError):
            fork_run(self.source, self.target, 16, 2, 1)

    def test_changed_gate_boundary_or_failed_proof_rejected(self):
        gate = self.root / "gate"
        proof = {"passed": True, "same_rollouts": True, "same_parameter_keys": True,
                 "max_parameter_difference": 0.0, "deterministic_training": True,
                 "boundary_step": 8, "target_step": 16}
        for name in ("boundary_check.json", "resume_check.json"):
            write(gate / name, proof)
        for name in ("continuous", "resumed"):
            write(gate / name / "smoke_result.json", {"global_step": 16, "arguments": {"training_passes": 2}})
        validate_extension_gate(gate)
        write(gate / "resume_check.json", {**proof, "passed": False})
        with self.assertRaises(ValueError):
            validate_extension_gate(gate)

    def test_multi_pass_requires_explicit_budget_and_preserves_legacy_default(self):
        from src.training.rl.pilot import main

        for flags, valid in (([], False), (["--training_passes", "4", "--checkpoint_limit", "1"], True),
                             (["--training_passes", "4", "--algorithm", "dr_grpo"], False)):
            destination = self.root / "new-pilot"
            argv = ["pilot", "--output_dir", str(destination), "--max_steps", "512", *flags]
            with patch("sys.argv", argv), patch("src.training.rl.pilot.run") as run, \
                    patch("src.training.rl.pilot.ensure_scratch_path"), contextlib.redirect_stderr(io.StringIO()):
                if valid:
                    main()
                    self.assertEqual(run.call_args.args[0].training_passes, 4)
                else:
                    with self.assertRaises(SystemExit):
                        main()
                    run.assert_not_called()

    def test_actual_constant_lr_trainer_extends_completed_budget_across_epoch(self):
        import torch
        from transformers import Trainer, TrainingArguments, set_seed

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, input_ids, labels):
                return {"loss": ((self.linear(input_ids.float()).squeeze(-1) - labels.float()) ** 2).mean()}

        dataset = [{"input_ids": [i], "labels": i * 2} for i in range(4)]

        def trainer(directory, steps):
            set_seed(42)
            args = TrainingArguments(output_dir=str(directory), use_cpu=True, max_steps=steps,
                                     per_device_train_batch_size=1, learning_rate=3e-6,
                                     lr_scheduler_type="constant", save_strategy="steps", save_steps=4,
                                     save_total_limit=2, report_to="none", disable_tqdm=True)
            return Trainer(model=Model(), args=args, train_dataset=dataset)

        with contextlib.redirect_stdout(io.StringIO()):
            uninterrupted = trainer(self.root / "continuous", 8)
            uninterrupted.train()
            trainer(self.root / "initial", 4).train()
            resumed = trainer(self.root / "resumed", 8)
            resumed.train(resume_from_checkpoint=str(self.root / "initial/checkpoint-4"))
        self.assertEqual(resumed.state.global_step, 8)
        self.assertEqual(resumed.lr_scheduler.get_last_lr(), [3e-6, 3e-6])
        for key, value in uninterrupted.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]))

    def test_actual_trl_sampler_matches_epochs_and_accelerate_resume(self):
        import torch
        from accelerate import Accelerator, skip_first_batches
        from torch.utils.data import DataLoader
        from trl.trainer.utils import RepeatSampler
        from src.training.rl.smoke import epoch_repeat_sampler_class

        settings = dict(data_source=list(range(32)), mini_repeat_count=4,
                        batch_size=16, repeat_count=64, shuffle=True, seed=0)
        legacy = RepeatSampler(**settings)
        expected = [list(legacy) for _ in range(4)]
        sampler_class = epoch_repeat_sampler_class()
        sampler = sampler_class(**settings)
        for epoch in range(4):
            sampler.set_epoch(epoch)
            rng = torch.get_rng_state().clone()
            self.assertEqual(list(sampler), expected[epoch])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            resumed = sampler_class(**settings)
            resumed.set_epoch(epoch)
            self.assertEqual(list(resumed), expected[epoch])
        resumed = sampler_class(**settings)
        loader = Accelerator(cpu=True).prepare(DataLoader(list(range(32)), sampler=resumed, batch_size=64))
        loader.set_epoch(1)
        observed = [i for batch in loader for i in batch.tolist()]
        self.assertEqual(observed, expected[1])
        # A mid-epoch resume must reproduce the skipped prefix as well as its suffix.
        resumed = sampler_class(**settings, resume_epoch=2)
        loader = Accelerator(cpu=True).prepare(DataLoader(list(range(32)), sampler=resumed, batch_size=64))
        loader.set_epoch(2)
        skipped = skip_first_batches(loader, num_batches=64)
        observed = [i for batch in skipped for i in batch.tolist()]
        self.assertEqual(observed, expected[2][4096:])


class ExtensionSummaryTests(unittest.TestCase):
    def test_final_summary_uses_saved_256_evaluations_after_checkpoint_rotation(self):
        from src.evaluation.summarize_grpo_extension import summarize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            source = root / "source"
            write(root / "submission.json", {"source": str(source), "control": {
                "greedy_root": str(root / "control"), "sampled_root": str(root / "control")}})
            for path in (source, run):
                write(path / "dataset_manifest.json", {"frozen": True})
            write(run / "progress.json", {"status": "completed", "global_step": 512})
            checkpoint = run / "checkpoints/checkpoint-512"
            checkpoint.mkdir(parents=True)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
                (checkpoint / name).write_bytes(b"state")
            (run / "rollouts.jsonl").write_text("".join(json.dumps({"problem": str(i % 512), "reward": 0.5}) + "\n" for i in range(8192)))
            write(run / "train_history.json", {"log_history": [
                {"step": i, "completions/mean_length": 10} for i in range(1, 512, 4)]})
            for step in (256, 512):
                model = run / ("checkpoints/checkpoint-256" if step == 256 else "final_model")
                for mode, count in (("greedy", 3), ("sampled", 5)):
                    for shard in range(count):
                        write(root / "evaluations" / f"step{step}" / mode / "grpo" / f"shard_{shard}_config.json",
                              {"model_path": str(model), "max_new_tokens": 3000, "temperature": 0 if mode == "greedy" else 1})

            def rows(path, mode, name, n, count):
                return {i: {"prompt_hash": str(i), "candidates": [
                    {"correct": i % 2 == 0, "truncated": False, "generated_tokens": 10}
                    for _ in range(count)], "input_tokens": 3, "runtime_seconds": 1} for i in range(n)}

            with patch("src.evaluation.summarize_grpo_extension.ensure_scratch_path", side_effect=Path), \
                    patch("src.evaluation.summarize_grpo_extension.validate_extension_gate"), \
                    patch("src.evaluation.summarize_grpo_extension.read_rows", side_effect=rows), \
                    contextlib.redirect_stdout(io.StringIO()):
                summarize(root, 512)
            summary = json.loads((root / "summary_512.json").read_text())
            self.assertEqual([r["step"] for r in summary["rows"]], [128, 256, 512])
            self.assertEqual([r["training_completion_tokens"] for r in summary["rows"]], [20480, 40960, 81920])
            self.assertFalse((run / "checkpoints/checkpoint-256").exists())


if __name__ == "__main__":
    unittest.main()
