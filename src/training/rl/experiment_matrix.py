from __future__ import annotations

import argparse
import json

from src.training.rl.common import RLExperimentConfig, algorithm_capabilities, save_configs


def stage2_matrix() -> list[RLExperimentConfig]:
    common = dict(train_samples=500, max_steps=50, num_rollouts=4, temperature=0.7, seed=0)
    return [
        RLExperimentConfig(name="reinforce_common50", algorithm="reinforce", **common),
        RLExperimentConfig(name="ppo_common50", algorithm="ppo", **common),
        RLExperimentConfig(name="grpo_common50", algorithm="grpo", **common),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or save the staged RLVR experiment matrix.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--capabilities", action="store_true")
    args = parser.parse_args()
    payload = algorithm_capabilities() if args.capabilities else [item.to_dict() for item in stage2_matrix()]
    if args.output and not args.capabilities:
        save_configs(args.output, stage2_matrix())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
