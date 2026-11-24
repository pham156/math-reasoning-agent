"""Explicit constant-LR continuation into a new scratch run; originals stay immutable."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from src.training.rl.checkpoint_storage import OUTPUTS, checkpoint_dirs, weights, write_receipt


def read(path):
    return json.loads(Path(path).read_text())


def fork_run(source, destination, target, passes, limit):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    for path in (source, destination):
        if path.resolve() != path or not path.is_relative_to(OUTPUTS.resolve()):
            raise ValueError("Continuation must use canonical scratch output paths")
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Use a new, separate continuation directory")
    progress = read(source / "progress.json")
    result = read(source / "smoke_result.json")
    metadata = read(source / "run_config.json")
    previous = metadata["arguments"]
    trainer_args = read(source / "training_args.json")
    step = progress["global_step"]
    expected = {"algorithm": "grpo", "learning_rate": 3e-6, "seed": 0,
                "reward_design": "correctness", "num_generations": 4, "temperature": 1.0, "kl_beta": 0.0}
    defaults = {"reward_design": "correctness", "num_generations": 4, "temperature": 1.0, "kl_beta": 0.0}
    if any(previous.get(k, defaults.get(k)) != v for k, v in expected.items()):
        raise ValueError("Extension requires the fixed seed0 GRPO control settings")
    if (progress["status"] != "completed" or step != previous["max_steps"] or
            result["global_step"] != step or not result["policy_update_observed"] or
            trainer_args["lr_scheduler_type"] != "constant" or trainer_args["learning_rate"] != 3e-6 or
            not 1 <= passes <= 4 or target != previous["train_samples"] * 4 // 16 * passes or
            target <= step or step % 4 or limit not in (1, 2)):
        raise ValueError("Invalid completed source or continuation budget")
    checkpoint = source / "checkpoints" / f"checkpoint-{step}"
    if Path(progress["checkpoint"]).resolve() != checkpoint or not weights(checkpoint):
        raise ValueError("Source checkpoint path/weights mismatch")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    if any(not (checkpoint / f).is_file() or (checkpoint / f).stat().st_size == 0 for f in required):
        raise ValueError("Source lacks resumable optimizer/scheduler/RNG state")
    if read(checkpoint / "trainer_state.json")["global_step"] != step:
        raise ValueError("Checkpoint step mismatch")
    if any(p.is_symlink() or not p.is_file() for p in checkpoint.iterdir()):
        raise ValueError("Refusing unexpected/symlinked checkpoint contents")
    originals = ["baseline.json", "dataset_manifest.json", "train_history.json", "rollouts.jsonl"]
    if len((source / "rollouts.jsonl").read_text().splitlines()) != step * 16:
        raise ValueError("Source rollout count mismatch")
    # Only immutable checkpoint files are hard-linked. All appendable logs are copied.
    destination.mkdir(parents=True)
    for name in originals + ["reward_groups.jsonl"]:
        path = source / name
        if name in originals or path.exists():
            shutil.copy2(path, destination / name)
    local_checkpoint = destination / "checkpoints" / checkpoint.name
    shutil.copytree(checkpoint, local_checkpoint, copy_function=os.link)
    arguments = {**previous, "output_dir": str(destination), "max_steps": target,
                 "training_passes": passes, "checkpoint_limit": limit, "resume": True,
                 "resume_from_checkpoint": str(local_checkpoint)}
    write_receipt(destination / "run_config.json", {**metadata, "arguments": arguments})
    write_receipt(destination / "progress.json", {"status": "paused", "global_step": step,
                  "target_steps": target, "checkpoint": str(local_checkpoint)})
    evidence = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("run_config.json", "smoke_result.json", "dataset_manifest.json", "progress.json")}
    write_receipt(destination / "continuation.json", {"source": str(source), "source_step": step,
                  "target_steps": target, "training_passes": passes, "source_evidence_sha256": evidence,
                  "checkpoint_copy": "hard-linked immutable files; appendable logs copied independently"})


def validate_extension_gate(root):
    root = Path(root)
    for file in ("boundary_check.json", "resume_check.json"):
        r = read(root / file)
        if not (r["passed"] and r["same_rollouts"] and r["same_parameter_keys"] and
                r["max_parameter_difference"] <= 1e-7 and r["deterministic_training"] and
                r["boundary_step"] == 8 and r["target_step"] == 16):
            raise ValueError("Repeated-pass extension gate failed")
    for name in ("continuous", "resumed"):
        result = read(root / name / "smoke_result.json")
        if result["global_step"] != 16 or result["arguments"]["training_passes"] != 2:
            raise ValueError("Gate did not cross the training pass boundary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fork = sub.add_parser("fork")
    fork.add_argument("--source", type=Path, required=True)
    fork.add_argument("--destination", type=Path, required=True)
    fork.add_argument("--target", type=int, required=True)
    fork.add_argument("--passes", type=int, required=True)
    fork.add_argument("--limit", type=int, default=1)
    gate = sub.add_parser("gate")
    gate.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "fork":
        fork_run(args.source, args.destination, args.target, args.passes, args.limit)
    else:
        validate_extension_gate(args.root)
        # Cleanup is limited to this new gate after successful numerical proof.
        root = args.root.absolute()
        if root.resolve() != root or not root.is_relative_to(OUTPUTS.resolve()) or root.name != "gate":
            raise ValueError("Unexpected gate cleanup scope")
        for name in ("continuous", "initial", "resumed"):
            run = root / name
            for checkpoint in checkpoint_dirs(run):
                shutil.rmtree(checkpoint)
            for path in weights(run / "final_model", require_complete=False):
                path.unlink()
            for index in (run / "final_model").glob("*.index.json"):
                index.unlink()
        write_receipt(root / "cleanup.json", {"status": "complete", "scope": "Validated extension gate weights/checkpoints only"})
        print("Repeated-pass extension gate passed; temporary gate weights removed")


if __name__ == "__main__":
    main()
