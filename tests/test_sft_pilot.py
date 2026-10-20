import unittest

from src.training.rl.sft_pilot import tokenize_solution


class Tokenizer:
    def apply_chat_template(self, turns, tokenize, add_generation_prompt):
        return [10, 11, 12] if add_generation_prompt else [10, 11, 12, 20, 21]


class SFTTests(unittest.TestCase):
    def test_only_response_is_supervised(self):
        record = tokenize_solution(Tokenizer(), "problem", "solution")
        self.assertEqual(record["labels"], [-100, -100, -100, 20, 21])
        self.assertEqual(len(record["labels"]), len(record["input_ids"]))

    def test_oversized_response_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds cap"):
            tokenize_solution(Tokenizer(), "problem", "solution", cap=4)

    def test_template_mismatch_rejected(self):
        class Mismatch(Tokenizer):
            def apply_chat_template(self, turns, tokenize, add_generation_prompt):
                return [99] if add_generation_prompt else [10, 20]
        with self.assertRaisesRegex(ValueError, "inference prefix"):
            tokenize_solution(Mismatch(), "problem", "solution")


if __name__ == "__main__":
    unittest.main()
