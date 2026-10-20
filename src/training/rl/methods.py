"""Explicit objective settings shared by the four reasoning smoke runs."""

import math

METHODS = {
    "grpo": dict(loss_type="grpo", scale_rewards="group", importance_sampling_level="token", epsilon=0.2, epsilon_high=0.2),
    "dr_grpo": dict(loss_type="dr_grpo", scale_rewards="none", importance_sampling_level="token", epsilon=0.2, epsilon_high=0.2),
    "dapo_style": dict(loss_type="dapo", scale_rewards="group", importance_sampling_level="token", epsilon=0.2, epsilon_high=0.28),
    "gspo": dict(loss_type="grpo", scale_rewards="group", importance_sampling_level="sequence", epsilon=3e-4, epsilon_high=4e-4),
}


def method_settings(name: str, clip_lower=None, clip_upper=None) -> dict:
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    settings = dict(METHODS[name])
    if clip_lower is not None or clip_upper is not None:
        if name != "grpo":
            raise ValueError("Clipping ablations are restricted to GRPO")
        for key, value in (("epsilon", clip_lower), ("epsilon_high", clip_upper)):
            if value is not None:
                if not math.isfinite(value) or not 0 < value < 1:
                    raise ValueError("Clip distances must be finite and between zero and one")
                settings[key] = value
    return settings


def check_method_config(name: str, config, clip_lower=None, clip_upper=None) -> None:
    for key, expected in method_settings(name, clip_lower, clip_upper).items():
        actual = getattr(config, key, None)
        if key == "scale_rewards" and expected == "none" and actual is False:
            continue
        if actual != expected:
            raise ValueError(f"{name}: {key} must be {expected!r}, got {actual!r}")
