from __future__ import annotations

import torch


def reinforce_advantages(rewards: torch.Tensor, baseline: str = "batch_mean") -> torch.Tensor:
    """Sequence-return advantages for vanilla policy gradient."""
    if baseline == "none":
        return rewards
    if baseline == "batch_mean":
        return rewards - rewards.mean()
    raise ValueError(f"Unknown REINFORCE baseline: {baseline}")


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """GRPO advantages, normalized independently within each prompt group."""
    if rewards.ndim != 2:
        raise ValueError("GRPO rewards must have shape [prompts, rollouts]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)


def policy_gradient_loss(
    token_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Token-sum policy gradient loss; avoids response-length normalization bias."""
    sequence_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
    return -(sequence_log_probs * advantages.detach()).mean()


def ppo_clipped_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO policy objective over completion tokens."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages[:, None]
    clipped = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages[:, None]
    return -((torch.minimum(unclipped, clipped) * completion_mask).sum() / completion_mask.sum().clamp_min(1))
