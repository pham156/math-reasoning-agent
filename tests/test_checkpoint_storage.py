import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.rl.checkpoint_storage import checkpoint_due, cleanup_completed_studies, cleanup_gate, save_limit
from src.training.rl.study_controls import BASE, updates


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def export(path):
    write(path / "config.json", {})
    (path / "model-00001.safetensors").write_bytes(b"first")
    (path / "model-00002.safetensors").write_bytes(b"second")
    write(path / "model.safetensors.index.json", {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}})


def run_files(path, config, n):
    write(path / "run_config.json", {"arguments": config})
    write(path / "smoke_result.json", {"arguments": {**config, "algorithm": "grpo", "train_samples": n}, "global_step": updates(config, n), "policy_update_observed": True})
    write(path / "training_args.json", {"epsilon": 0.2, "epsilon_high": 0.2, "num_generations": 4,
          "temperature": config["temperature"], "learning_rate": 3e-6, "beta": 0.0, "loss_type": "grpo",
          "scale_rewards": "group", "importance_sampling_level": "token", "lr_scheduler_type": "constant",
          "gradient_accumulation_steps": 16, "max_completion_length": 3000, "generation_batch_size": 64, "max_steps": updates(config, n)})
    write(path / "progress.json", {"status": "completed", "global_step": updates(config, n)})
    (path / "rollouts.jsonl").write_text("preserved rollouts\n")
    for step in (4, updates(config, n)):
        write(path / "checkpoints" / f"checkpoint-{step}" / "trainer_state.json", {"global_step": step})
        (path / "checkpoints" / f"checkpoint-{step}" / "optimizer.pt").write_bytes(b"optimizer")
        (path / "checkpoints" / f"checkpoint-{step}" / "scheduler.pt").write_bytes(b"scheduler")
        (path / "checkpoints" / f"checkpoint-{step}" / "rng_state.pth").write_bytes(b"rng")
        export(path / "checkpoints" / f"checkpoint-{step}")
    export(path / "final_model")


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outputs = Path(self.tmp.name)
        self.patch = patch("src.training.rl.checkpoint_storage.OUTPUTS", self.outputs)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.root = self.outputs / "study"
        self.cases = {}
        manifest = {"studies": {}}
        for study, label, config in (("reward", "format_bonus_seed0", {**BASE, "reward_design": "format_bonus"}),
                                     ("exploration", "temperature_0p7", {**BASE, "temperature": 0.7})):
            run = self.root / study / "runs" / label
            case = {"label": label, "reused": False, "settings": config, "run_root": str(run)}
            self.cases[study] = case
            manifest["studies"][study] = {"runs": [case]}
            run_files(run, config, 512)
            gate = self.root / study / "gates" / label
            for mode in ("continuous", "resumed"):
                run_files(gate / mode, config, 32)
            write(gate / "boundary_check.json", {"passed": True})
            write(gate / "resume_check.json", {"passed": True, "same_rollouts": True, "same_parameter_keys": True, "max_parameter_difference": 0.0, "deterministic_training": True})
        write(self.root / "submission.json", manifest)
        write(self.root / "retention_policy.json", {"version": 1, "root": str(self.root), "active_pilot_checkpoints": 1,
              "cleanup_gates_after_validation": True, "cleanup_pilots_after": "both_study_summaries", "keep_final_models": True,
              "completed_pilot_checkpoints": 1})

    def summaries(self):
        for study, case in self.cases.items():
            write(self.root / study / "summary.json", {"runs": [{"label": case["label"], "settings": case["settings"], "n": 300, "pass_at_k": {"1": 0.5, "4": 0.7, "8": 0.8}}]})
            (self.root / study / "summary.md").write_text("complete summary\n")

    def test_limits_are_scoped_and_gates_keep_boundary_snapshot(self):
        case = self.cases["reward"]
        self.assertEqual(save_limit(case["run_root"]), 1)
        self.assertEqual(save_limit(self.root / "reward/gates" / case["label"] / "continuous"), 2)
        self.assertEqual(save_limit(self.outputs / "old-study/runs/control"), 2)

    def test_save_only_at_complete_stop_or_final_boundaries(self):
        self.assertFalse(checkpoint_due(16, 128, 4, False))
        self.assertFalse(checkpoint_due(31, 128, 4, True))
        self.assertTrue(checkpoint_due(32, 128, 4, True))
        self.assertTrue(checkpoint_due(128, 128, 4, False))
        self.assertTrue(checkpoint_due(256, 256, 8, False))

    def test_trainer_model_card_is_preserved_during_cleanup(self):
        self.summaries()
        case = self.cases["reward"]
        run = Path(case["run_root"])
        gate = self.root / "reward/gates" / case["label"]
        cards = [run / "checkpoints/README.md", gate / "continuous/checkpoints/README.md"]
        for card in cards:
            card.write_text("Trainer-generated model card\n")
        cleanup_completed_studies(self.root)
        for card in cards:
            self.assertEqual(card.read_text(), "Trainer-generated model card\n")
        self.assertTrue((run / "checkpoints/checkpoint-128/optimizer.pt").exists())

    def test_symlinked_model_card_is_rejected(self):
        case = self.cases["reward"]
        gate = self.root / "reward/gates" / case["label"]
        outside = self.outputs / "protected.md"
        outside.write_text("keep")
        (gate / "continuous/checkpoints/README.md").symlink_to(outside)
        with self.assertRaises(ValueError):
            cleanup_gate(gate, case["settings"])
        self.assertEqual(outside.read_text(), "keep")

    def test_gate_cleanup_requires_proof_and_is_repeatable(self):
        case = self.cases["reward"]
        gate = self.root / "reward/gates" / case["label"]
        write(gate / "boundary_check.json", {"passed": False})
        with self.assertRaises(ValueError):
            cleanup_gate(gate, case["settings"])
        self.assertTrue((gate / "continuous/checkpoints/checkpoint-4").exists())
        write(gate / "boundary_check.json", {"passed": True})
        cleanup_gate(gate, case["settings"])
        cleanup_gate(gate, case["settings"])
        self.assertFalse(list((gate / "continuous/checkpoints").iterdir()))
        self.assertFalse(list((gate / "continuous/final_model").glob("*.safetensors")))
        self.assertTrue((gate / "continuous/final_model/config.json").exists())
        self.assertTrue((gate / "continuous/rollouts.jsonl").exists())
        write(gate / "boundary_check.json", {"passed": True, "changed": True})
        with self.assertRaises(ValueError):
            cleanup_gate(gate, case["settings"])

    def test_checkpoint_cleanup_waits_for_both_evaluations(self):
        run = Path(self.cases["reward"]["run_root"])
        cleanup_completed_studies(self.root)
        self.assertTrue((run / "checkpoints/checkpoint-128").exists())
        self.summaries()
        cleanup_completed_studies(self.root)
        cleanup_completed_studies(self.root)
        for case in self.cases.values():
            path = Path(case["run_root"])
            self.assertEqual([p.name for p in (path / "checkpoints").iterdir()], ["checkpoint-128"])
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                self.assertTrue((path / "checkpoints/checkpoint-128" / name).is_file())
            self.assertEqual(len(list((path / "final_model").glob("*.safetensors"))), 2)
            self.assertTrue((path / "rollouts.jsonl").exists())

    def test_missing_final_shard_preserves_all_resume_state(self):
        self.summaries()
        path = Path(self.cases["exploration"]["run_root"])
        (path / "final_model/model-00002.safetensors").unlink()
        with self.assertRaises(ValueError):
            cleanup_completed_studies(self.root)
        for case in self.cases.values():
            self.assertTrue((Path(case["run_root"]) / "checkpoints/checkpoint-128").exists())

    def test_incomplete_latest_checkpoint_preserves_all_older_checkpoints(self):
        self.summaries()
        run = Path(self.cases["exploration"]["run_root"])
        (run / "checkpoints/checkpoint-128/optimizer.pt").unlink()
        with self.assertRaises(ValueError):
            cleanup_completed_studies(self.root)
        for case in self.cases.values():
            self.assertTrue((Path(case["run_root"]) / "checkpoints/checkpoint-4").exists())

    def test_reused_control_checkpoints_are_not_removed(self):
        self.summaries()
        old = self.outputs / "existing-control/checkpoints/checkpoint-128/optimizer.pt"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"preserved")
        manifest = json.loads((self.root / "submission.json").read_text())
        manifest["studies"]["reward"]["runs"].append({"label": "correctness_seed0", "settings": BASE, "reused": True, "run_root": str(old.parents[2])})
        write(self.root / "submission.json", manifest)
        summary = json.loads((self.root / "reward/summary.json").read_text())
        summary["runs"].append({"label": "correctness_seed0", "settings": BASE, "n": 300, "pass_at_k": {"1": 0.5, "4": 0.7, "8": 0.8}})
        write(self.root / "reward/summary.json", summary)
        cleanup_completed_studies(self.root)
        self.assertTrue(old.exists())

    def test_cleanup_rejects_symlinked_checkpoint(self):
        case = self.cases["reward"]
        gate = self.root / "reward/gates" / case["label"]
        outside = self.outputs / "do-not-touch"
        outside.mkdir()
        (outside / "protected").write_text("keep")
        (gate / "continuous/checkpoints/checkpoint-99").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup_gate(gate, case["settings"])
        self.assertTrue((outside / "protected").exists())

    def test_cleanup_rejects_symlinked_gate_run_ancestor(self):
        import shutil
        case = self.cases["reward"]
        gate = self.root / "reward/gates" / case["label"]
        outside = self.outputs / "outside"
        (gate / "continuous").rename(outside)
        (gate / "continuous").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup_gate(gate, case["settings"])
        self.assertTrue((outside / "checkpoints/checkpoint-8/optimizer.pt").exists())
        (gate / "continuous").unlink()
        shutil.move(str(outside), str(gate / "continuous"))

    def test_actual_trainer_boundary_only_save_and_resume(self):
        import torch
        from transformers import Trainer, TrainerCallback, TrainingArguments

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, input_ids, labels):
                return {"loss": ((self.linear(input_ids.float()).squeeze(-1) - labels.float()) ** 2).mean()}

        class SaveAtBoundary(TrainerCallback):
            def __init__(self, stop=0):
                self.stop = stop

            def on_step_end(self, args, state, control, **kwargs):
                stop = self.stop and state.global_step >= self.stop
                if checkpoint_due(state.global_step, args.max_steps, 2, stop):
                    control.should_save = True
                    if stop:
                        control.should_training_stop = True
                return control

        folder = self.outputs / "tiny-trainer"
        args = TrainingArguments(output_dir=str(folder), use_cpu=True, bf16=False, fp16=False,
                                 max_steps=4, per_device_train_batch_size=1, save_strategy="no",
                                 save_total_limit=1, report_to="none", disable_tqdm=True)
        dataset = [{"input_ids": [i], "labels": i * 2} for i in range(4)]
        with contextlib.redirect_stdout(io.StringIO()):
            Trainer(model=TinyModel(), args=args, train_dataset=dataset, callbacks=[SaveAtBoundary(2)]).train()
            self.assertTrue((folder / "checkpoint-2/trainer_state.json").exists())
            self.assertEqual(len(list(folder.glob("checkpoint-*"))), 1)
            Trainer(model=TinyModel(), args=args, train_dataset=dataset, callbacks=[SaveAtBoundary()]).train(resume_from_checkpoint=str(folder / "checkpoint-2"))
        self.assertTrue((folder / "checkpoint-4/trainer_state.json").exists())
        self.assertFalse((folder / "checkpoint-2").exists())


if __name__ == "__main__":
    unittest.main()
