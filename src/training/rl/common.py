from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Algorithm = Literal["reinforce", "ppo", "grpo"]
Subset = Literal["math_random", "math_level_1_2", "math_level_3_5", "gsm8k"]


@dataclass(frozen=True)
class RLExperimentConfig:
    name: str
    algorithm: Algorithm
    dataset_subset: Subset = "math_random"
    model_path: str = "$MODEL_DIR/Qwen2.5-Math-7B-Instruct"
    train_samples: int = 500
    learning_rate: float = 2e-7
    max_steps: int = 50
    num_rollouts: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    max_prompt_tokens: int = 1024
    max_completion_tokens: int = 512
    seed: int = 0
    kl_coef: float = 0.05
    clip_range: float = 0.2
    output_dir: str = "$OUTPUT_DIR/rl_experiments"

    def validate(self) -> None:
        if self.algorithm not in {"reinforce", "ppo", "grpo"}:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        if self.num_rollouts < 1 or self.max_steps < 1 or self.train_samples < 1:
            raise ValueError("rollouts, steps, and train_samples must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("sampled RL requires temperature > 0 and 0 < top_p <= 1")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def load_config(path: str | Path) -> RLExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = RLExperimentConfig(**payload)
    config.validate()
    return config


def save_configs(path: str | Path, configs: list[RLExperimentConfig]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([config.to_dict() for config in configs], indent=2) + "\n",
        encoding="utf-8",
    )


def algorithm_capabilities() -> dict:
    """Describe installed backends without importing TRL's heavy lazy modules."""
    import importlib.metadata
    trl_version = importlib.metadata.version("trl")
    distribution = importlib.metadata.distribution("trl")
    package_root = Path(distribution.locate_file("trl"))
    grpo = (package_root / "trainer" / "grpo_trainer.py").is_file()
    ppo = (package_root / "experimental" / "ppo" / "ppo_trainer.py").is_file()
    return {
        "trl_version": trl_version,
        "grpo": {
            "backend": "trl.GRPOTrainer",
            "available": grpo,
            "callable_verifiable_reward": True,
        },
        "ppo": {
            "backend": "trl.experimental.ppo.PPOTrainer",
            "available": ppo,
            "callable_verifiable_reward": False,
            "blocker": "TRL PPO requires reward_model and value_model objects; an adapter is needed for exact-answer rewards.",
        },
        "reinforce": {
            "backend": "project implementation required",
            "available": False,
            "blocker": "TRL has no REINFORCE trainer; a memory-safe on-policy loop is required.",
        },
    }
