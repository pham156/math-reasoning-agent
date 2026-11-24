"""Independent reward and exploration pilots around the paired-seed GRPO control."""

import argparse
import json
from pathlib import Path


BASE = {"seed": 0, "learning_rate": 3e-6, "kl_beta": 0.0,
        "clip_lower": 0.2, "clip_upper": 0.2, "reward_design": "correctness",
        "num_generations": 4, "temperature": 1.0}


def profiles(study):
    if study == "reward":
        return {f"{design}_seed{seed}": {**BASE, "seed": seed, "reward_design": design}
                for seed in (0, 1) for design in ("correctness", "format_bonus", "length_penalty")}
    if study == "exploration":
        return {"correctness_seed0": dict(BASE),
                "temperature_0p7": {**BASE, "temperature": 0.7},
                "temperature_1p3": {**BASE, "temperature": 1.3},
                "rollouts_2": {**BASE, "num_generations": 2},
                "rollouts_8": {**BASE, "num_generations": 8}}
    raise ValueError("Unknown study")


def updates(config, problems=512):
    if problems % 16 or config["num_generations"] not in (2, 4, 8):
        raise ValueError("Require complete 16-problem rollout batches")
    return problems * config["num_generations"] // 16


def validate_run(path, config, problems=512):
    path = Path(path)
    result = json.loads((path / "smoke_result.json").read_text())
    args = result["arguments"]
    training = json.loads((path / "training_args.json").read_text())
    for key, default in BASE.items():
        if key not in ("clip_lower", "clip_upper") and args.get(key, default) != config[key]:
            raise ValueError(f"Changed run setting: {key}")
    for key, value in {"epsilon": config["clip_lower"], "epsilon_high": config["clip_upper"],
                       "num_generations": config["num_generations"], "temperature": config["temperature"],
                       "learning_rate": config["learning_rate"], "beta": 0.0,
                       "loss_type": "grpo", "scale_rewards": "group", "importance_sampling_level": "token",
                       "lr_scheduler_type": "constant", "gradient_accumulation_steps": 16,
                       "max_completion_length": 3000, "generation_batch_size": 16 * config["num_generations"],
                       "max_steps": updates(config, problems)}.items():
        if training.get(key) != value:
            raise ValueError(f"Changed trainer setting: {key}")
    if args.get("algorithm") != "grpo" or args["train_samples"] != problems or result["global_step"] != updates(config, problems) or not result["policy_update_observed"]:
        raise ValueError("Incomplete or mismatched training")
    if not (path / "final_model/config.json").exists():
        raise ValueError("Missing full policy checkpoint")
    return result


def validate_gate(root, config):
    root = Path(root)
    result = json.loads((root / "resume_check.json").read_text())
    if not result["passed"] or not result["same_rollouts"]:
        raise ValueError("Objective/restart gate failed")
    for label in ("continuous", "resumed"):
        validate_run(root / label, config, 32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["reward", "exploration"])
    parser.add_argument("label")
    parser.add_argument("--field", choices=list(BASE) + ["max_steps"])
    parser.add_argument("--validate_gate")
    args = parser.parse_args()
    config = profiles(args.study)[args.label]
    if args.validate_gate:
        validate_gate(args.validate_gate, config)
        from src.training.rl.checkpoint_storage import cleanup_gate
        cleanup_gate(args.validate_gate, config)
        print("Objective and exact restart gate passed")
    elif args.field:
        print(updates(config) if args.field == "max_steps" else config[args.field])
    else:
        print("\n".join(str(v) for key, value in config.items() for v in ("--" + key, value)))


if __name__ == "__main__":
    main()
