"""Compare uninterrupted and rollout-boundary resumed runs before pilot launch."""

import argparse
import json
from pathlib import Path

from src.training.grpo.train import write_json


def main():
    from safetensors import safe_open
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--boundary_only", action="store_true", help="Compare both checkpoint-4 snapshots before restarting")
    parser.add_argument("--boundary_step", type=int, default=4)
    parser.add_argument("--target_step", type=int, default=8)
    args = parser.parse_args()
    if args.boundary_step < 1 or args.target_step != 2 * args.boundary_step:
        parser.error("Gate requires two complete rollout batches")
    continuous, resumed = args.root / "continuous", args.root / "resumed"
    def records(path):
        return [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    resumed_rollouts = records(resumed)
    continuous_rollouts = records(continuous)
    if args.boundary_only:
        continuous_rollouts = continuous_rollouts[:len(resumed_rollouts)]
    same_rollouts = continuous_rollouts == resumed_rollouts
    maximum = 0.0
    same_keys = True
    checkpoint_dir = f"checkpoints/checkpoint-{args.boundary_step}" if args.boundary_only else "final_model"
    left_files = sorted((continuous / checkpoint_dir).glob("*.safetensors"))
    right_files = sorted((resumed / checkpoint_dir).glob("*.safetensors"))
    if not left_files or [p.name for p in left_files] != [p.name for p in right_files]:
        raise RuntimeError("Missing or mismatched checkpoint shards")
    for left, right in zip(left_files, right_files, strict=True):
        with safe_open(left, framework="pt", device="cpu") as a, safe_open(right, framework="pt", device="cpu") as b:
            same_keys = same_keys and a.keys() == b.keys()
            if not same_keys:
                break
            for key in a.keys():
                delta = float((a.get_tensor(key) - b.get_tensor(key)).abs().max())
                if not torch.isfinite(torch.tensor(delta)):
                    raise FloatingPointError("Nonfinite checkpoint difference")
                maximum = max(maximum, delta)
    if args.boundary_only:
        steps_valid = json.loads((resumed / "progress.json").read_text())["global_step"] == args.boundary_step and len(resumed_rollouts) == args.boundary_step * 16
    else:
        results = [json.loads((path / "smoke_result.json").read_text()) for path in (continuous, resumed)]
        steps_valid = all(row["global_step"] == args.target_step and row["policy_update_observed"] for row in results) and len(resumed_rollouts) == args.target_step * 16
    deterministic_configs = [json.loads((path / "run_config.json").read_text())["determinism"] for path in (continuous, resumed)]
    deterministic = all(row["requested"] and row["algorithms_enabled"] for row in deterministic_configs)
    passed = same_rollouts and same_keys and maximum <= 1e-7 and steps_valid and deterministic
    payload = {"passed": passed, "same_rollouts": same_rollouts, "same_parameter_keys": same_keys, "max_parameter_difference": maximum, "tolerance": 1e-7, "deterministic_training": deterministic, "boundary_step": args.boundary_step, "target_step": args.target_step, "scope": "Rollout-boundary checkpoint comparison" if args.boundary_only else "Full weights and sampled completions compared across the requested restart boundary"}
    write_json(args.root / ("boundary_check.json" if args.boundary_only else "resume_check.json"), payload)
    print(json.dumps(payload))
    if not passed:
        raise SystemExit("Resume equivalence failed; dependent pilots must remain blocked")


if __name__ == "__main__":
    main()
