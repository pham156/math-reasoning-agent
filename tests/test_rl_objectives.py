from __future__ import annotations

import unittest

import torch

from src.training.rl.objectives import group_relative_advantages, policy_gradient_loss, ppo_clipped_loss, reinforce_advantages


class RLObjectivesTest(unittest.TestCase):
    def test_reinforce_centered_baseline(self) -> None:
        advantages = reinforce_advantages(torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.allclose(advantages, torch.tensor([-0.5, 0.5])))

    def test_grpo_group_normalization(self) -> None:
        advantages = group_relative_advantages(torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
        self.assertTrue(torch.allclose(advantages[0], torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.equal(advantages[1], torch.zeros(2)))

    def test_policy_gradient_is_token_sum(self) -> None:
        logps = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        loss = policy_gradient_loss(logps, mask, torch.tensor([1.0, -1.0]))
        self.assertAlmostEqual(loss.item(), 0.5)

    def test_ppo_clips_large_ratio(self) -> None:
        new = torch.log(torch.tensor([[2.0]]))
        old = torch.zeros_like(new)
        loss = ppo_clipped_loss(new, old, torch.ones_like(new), torch.tensor([1.0]), 0.2)
        self.assertAlmostEqual(loss.item(), -1.2, places=5)


if __name__ == "__main__":
    unittest.main()
