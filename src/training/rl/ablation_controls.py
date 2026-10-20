"""Fixed GRPO one-factor controls shared by submission, workers, and summaries."""

import argparse
import math


CONTROL = {"seed": 0, "learning_rate": 1e-6, "kl_beta": 0.0, "clip_lower": 0.2, "clip_upper": 0.2}
CASES = {
    "seed1_lr1e-06": {"seed": 1},
    "seed1_lr3e-06": {"seed": 1, "learning_rate": 3e-6},
    "kl_1e-03": {"kl_beta": 0.001},
    "kl_1e-02": {"kl_beta": 0.01},
    "clip_0p1": {"clip_lower": 0.1, "clip_upper": 0.1},
    "clip_0p3": {"clip_lower": 0.3, "clip_upper": 0.3},
}


def settings(label):
    return {**CONTROL, **CASES[label]}


def flags(label):
    return [item for key, value in settings(label).items() for item in ("--" + key, str(value))]


def check_saved(label, arguments, training_args, expected_settings=None):
    expected_settings = expected_settings or settings(label)
    for key, expected in expected_settings.items():
        saved = arguments.get(key, CONTROL[key])
        if saved != expected:
            raise ValueError(f"Changed {key} in {label}: {saved} != {expected}")
    expected = expected_settings
    for key, value in {"learning_rate": expected["learning_rate"], "beta": expected["kl_beta"], "epsilon": expected["clip_lower"], "epsilon_high": expected["clip_upper"], "num_generations": 4, "temperature": 1.0, "max_completion_length": 3000, "loss_type": "grpo", "scale_rewards": "group", "importance_sampling_level": "token", "lr_scheduler_type": "constant", "warmup_steps": 0}.items():
        if training_args.get(key) != value:
            raise ValueError(f"Trainer setting mismatch: {label}/{key}")


def validate_gate(root, label):
    import json
    from pathlib import Path

    root = Path(root)
    if not json.loads((root / "resume_check.json").read_text())["passed"]:
        raise ValueError("Restart gate did not pass")
    for name in ("continuous", "resumed"):
        result = json.loads((root / name / "smoke_result.json").read_text())
        training = json.loads((root / name / "training_args.json").read_text())
        check_saved(label, result["arguments"], training)
        if result["global_step"] != 8 or not result["policy_update_observed"]:
            raise ValueError("Smoke did not update the policy for eight steps")
        if settings(label)["kl_beta"]:
            kl = result["history_summary"]["kl_mean"]
            if kl is None or not math.isfinite(kl) or kl <= 0 or not result["reference"]["frozen"]:
                raise ValueError("KL smoke needs measured positive divergence and a frozen reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=list(CASES))
    parser.add_argument("--validate_gate")
    args = parser.parse_args()
    if args.validate_gate:
        validate_gate(args.validate_gate, args.label)
        print("Objective/reference/restart smoke gate passed")
    else:
        print("\n".join(flags(args.label)))


if __name__ == "__main__":
    main()
