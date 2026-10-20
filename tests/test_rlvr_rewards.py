from __future__ import annotations

import unittest

from src.training.rewards.math_correctness import canonical_final_answer, final_answer_reward_one


class RLVRRewardTest(unittest.TestCase):
    def test_twenty_answer_formats(self) -> None:
        positive_cases = [
            (r"The final answer is \boxed{\frac{1}{2}}.", "1/2"),
            (r"Answer: \boxed{2/4}", r"\frac{1}{2}"),
            ("#### 5,600", "5600"),
            (r"We get \boxed{10\%}.", "0.1"),
            ("Final answer: 0.125", "1/8"),
            (r"\boxed{-3}", "-3"),
            (r"\fbox{42}", "42"),
            (r"Therefore, \boxed{\{3,1,2\}}", "{1,2,3}"),
            ("Final answer: [1, 2, 3]", "[1,2,3]"),
            (r"\boxed{\left(3,\frac{\pi}{2}\right)}", r"(3,\pi/2)"),
            (r"Answer: \text{Evelyn}", "Evelyn"),
            (r"\boxed{\sqrt{51}}", r"\sqrt{51}"),
            ("Final answer: 7 and 11", "11,7"),
            ("Answer: 1;2;3", "1,2,3"),
            (r"\boxed{ -\frac{3}{6} }", "-1/2"),
            ("Final answer: .5", "1/2"),
            ("Final answer: 50%", "1/2"),
            (r"\boxed{\mathrm{AB}}", "ab"),
            (r"\boxed{1,000,000}", "1000000"),
            ("Therefore, x=6", "x=6"),
        ]
        self.assertEqual(len(positive_cases), 20)
        for prediction, reference in positive_cases:
            with self.subTest(prediction=prediction, reference=reference):
                self.assertEqual(final_answer_reward_one(prediction, reference), 1.0)

    def test_negative_cases(self) -> None:
        negative_cases = [
            (r"\boxed{10\%}", "10"),
            (r"\boxed{3}", "4"),
            ("Final answer: [1,2,3]", "[3,2,1]"),
            (r"\boxed{p+q}", "p-q"),
        ]
        for prediction, reference in negative_cases:
            with self.subTest(prediction=prediction, reference=reference):
                self.assertEqual(final_answer_reward_one(prediction, reference), 0.0)

    def test_canonical_set_is_order_insensitive(self) -> None:
        self.assertEqual(canonical_final_answer(r"\boxed{\{b,a,c\}}"), "{a,b,c}")


if __name__ == "__main__":
    unittest.main()
