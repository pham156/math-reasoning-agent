from __future__ import annotations

import argparse
import json

from src.training.rl.common import algorithm_capabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO RLVR capability and smoke interface.")
    parser.add_argument("--capability_only", action="store_true")
    args = parser.parse_args()
    capability = algorithm_capabilities()["ppo"]
    print(json.dumps(capability, indent=2))
    if not args.capability_only:
        raise RuntimeError(capability["blocker"])


if __name__ == "__main__":
    main()
