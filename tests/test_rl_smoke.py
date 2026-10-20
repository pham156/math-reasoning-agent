import unittest
from collections import defaultdict
from types import SimpleNamespace

from src.training.rl.methods import METHODS, check_method_config, method_settings


class ObjectiveSettingsTest(unittest.TestCase):
    def test_four_distinct_objectives(self):
        self.assertEqual(set(METHODS), {"grpo", "dr_grpo", "dapo_style", "gspo"})
        signatures = {tuple(sorted(method_settings(name).items())) for name in METHODS}
        self.assertEqual(len(signatures), 4)

    def test_defaults_cannot_silently_replace_objective(self):
        for name in METHODS:
            settings = method_settings(name)
            check_method_config(name, SimpleNamespace(**settings))
            settings["loss_type"] = "wrong_default"
            with self.assertRaises(ValueError):
                check_method_config(name, SimpleNamespace(**settings))

    def test_profile_copies_are_independent(self):
        profile = method_settings("grpo")
        profile["epsilon"] = 0.9
        self.assertEqual(method_settings("grpo")["epsilon"], 0.2)

    def test_dr_grpo_scaling_and_gspo_clip_scale(self):
        self.assertEqual(method_settings("dr_grpo")["scale_rewards"], "none")
        self.assertEqual(method_settings("gspo")["importance_sampling_level"], "sequence")
        self.assertLess(method_settings("gspo")["epsilon"], method_settings("grpo")["epsilon"])

    def test_actual_trainer_loss_and_gradients(self):
        import torch
        from src.training.rl.smoke import smoke_trainer_class

        for name in METHODS:
            for delta in (0.00001, 0.01):
                with self.subTest(method=name, delta=delta):
                    cls = smoke_trainer_class()
                    trainer = object.__new__(cls)
                    settings = method_settings(name)
                    current = torch.tensor([[delta, delta * 2, -delta], [-delta, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
                    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
                    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
                    trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (current, torch.zeros_like(current))
                    trainer.top_entropy_quantile = 1.0
                    trainer.beta = 0.0
                    trainer.use_vllm = False
                    trainer.importance_sampling_level = settings["importance_sampling_level"]
                    trainer.loss_type = settings["loss_type"]
                    trainer.epsilon_low = settings["epsilon"]
                    trainer.epsilon_high = settings["epsilon_high"]
                    trainer.max_completion_length = 3
                    trainer.current_gradient_accumulation_steps = 2
                    trainer.args = SimpleNamespace(delta=None, steps_per_generation=8)
                    trainer.accelerator = SimpleNamespace(num_processes=1, gather=lambda tensor: tensor)
                    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
                    trainer.model = SimpleNamespace(training=True)
                    inputs = {"prompt_ids": torch.zeros(2, 1, dtype=torch.long), "prompt_mask": torch.ones(2, 1), "completion_ids": torch.zeros(2, 3, dtype=torch.long), "completion_mask": mask, "old_per_token_logps": torch.zeros_like(current), "advantages": advantages, "num_items_in_batch": torch.tensor(32.0)}
                    actual = trainer._compute_loss(SimpleNamespace(training=True), inputs)
                    actual_gradient = torch.autograd.grad(actual, current)[0]
                    independent = current.detach().clone().requires_grad_(True)
                    if name == "gspo":
                        ratios = torch.exp((independent * mask).sum(-1) / mask.sum(-1))
                        objective = torch.minimum(ratios * advantages, ratios.clamp(1 - settings["epsilon"], 1 + settings["epsilon_high"]) * advantages)
                        expected = -objective.mean() / 2
                    else:
                        ratios = independent.exp()
                        terms = -torch.minimum(ratios * advantages[:, None], ratios.clamp(1 - settings["epsilon"], 1 + settings["epsilon_high"]) * advantages[:, None]) * mask
                        if name == "grpo":
                            expected = (terms.sum(-1) / mask.sum(-1)).mean() / 2
                        elif name == "dr_grpo":
                            expected = terms.sum() / (2 * 3 * 2)
                        else:
                            expected = terms.sum() / (32 * 2 / 8)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, independent)[0])

    def test_math_verify_correctness(self):
        from src.training.rl.smoke import grade_solution

        for text, reference, expected in [
            (r"\boxed{2/4}", r"\frac{1}{2}", 1.0),
            (r"\boxed{\sqrt{4}}", "2", 1.0),
            (r"\boxed{3}", "4", 0.0),
            ("", "", 0.0),
            ("I cannot solve this problem.", "2", 0.0),
        ]:
            with self.subTest(text=text):
                self.assertEqual(grade_solution(text, reference), expected)


if __name__ == "__main__":
    unittest.main()
