"""Small, auditable outcome-reward variants; correctness remains the evaluation target."""

import math
import re
from statistics import mean, pvariance

from src.evaluation.answer import _extract_balanced_braced


REWARDS = ("correctness", "format_bonus", "length_penalty")


def format_score(text, parseable):
    if len(re.findall(r"\\boxed\b", text)) != 1 or r"\fbox" in text:
        return 0.0
    if not re.search(r"\\boxed\s*\{", text):
        return 0.0
    boxes = _extract_balanced_braced(text, r"\boxed")
    return float(len(boxes) == 1 and bool(boxes[0]) and parseable(boxes[0]))


def reward_components(text, correctness, completion_tokens, cap, design, parseable):
    if design not in REWARDS or correctness not in (0, 1) or cap <= 0 or completion_tokens < 0:
        raise ValueError("Invalid reward inputs")
    formatting = format_score(text, parseable)
    bonus = 0.10 * formatting if design == "format_bonus" else 0.0
    penalty = 0.05 * min(completion_tokens / cap, 1.0) if design == "length_penalty" else 0.0
    return {"correctness": float(correctness), "format_score": formatting,
            "format_bonus": bonus, "length_penalty": penalty,
            "completion_tokens": completion_tokens, "reward": float(correctness) + bonus - penalty}


def batch_metrics(records, group_size):
    if not records or len(records) % group_size:
        raise ValueError("Reward batch must contain complete prompt groups")
    groups = [records[i:i + group_size] for i in range(0, len(records), group_size)]
    if any(len({r["problem"] for r in group}) != 1 for group in groups):
        raise ValueError("Reward groups contain different problems")
    for row in records:
        if not all(math.isfinite(float(row[k])) for k in ("correctness", "reward")):
            raise ValueError("Nonfinite reward")
    return {"correctness_mean": mean(r["correctness"] for r in records),
            "format_mean": mean(r["format_score"] for r in records),
            "length_penalty_mean": mean(r["length_penalty"] for r in records),
            "correctness_group_variance": mean(pvariance(r["correctness"] for r in g) for g in groups),
            "total_group_variance": mean(pvariance(r["reward"] for r in g) for g in groups),
            "mixed_group_fraction": mean(0 < sum(r["correctness"] for r in g) < group_size for g in groups),
            "format_only_failure_fraction": mean(r["correctness"] == 0 and r["format_bonus"] > 0 for r in records)}
