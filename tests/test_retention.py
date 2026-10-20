import unittest

from src.evaluation.summarize_retention import paired_interval


class RetentionTests(unittest.TestCase):
    def test_unchanged_pairs_zero_interval(self):
        self.assertEqual(paired_interval([0] * 32), [0, 0])

    def test_uniform_loss_detected(self):
        self.assertEqual(paired_interval([-1] * 100), [-1, -1])

    def test_answer_logprob_scores_only_continuation(self):
        import torch
        from src.evaluation.retention import answer_logprob
        logits = torch.tensor([[[9.0, -9.0], [-9.0, 9.0], [0.0, 0.0]]])
        self.assertAlmostEqual(answer_logprob(logits, 2, [1]), 0, places=6)
        self.assertLess(answer_logprob(logits, 2, [0]), -17)
        self.assertAlmostEqual(answer_logprob(logits, 1, [0, 1]), 0, places=6)
        with self.assertRaises(ValueError):
            answer_logprob(logits, 0, [1])


if __name__ == "__main__":
    unittest.main()
