import unittest

from src.evaluation.answer import closed_boxed_answers, extract_committed_answer, extract_final_answer


class FinalAnswerExtractionTest(unittest.TestCase):
    def test_markdown_heading_does_not_override_boxed_answer(self):
        text = "#### Step 1: Expand\nSo the answer is \\boxed{7}."
        self.assertEqual(extract_final_answer(text), "7")

    def test_trailing_markdown_heading_does_not_override_boxed_answer(self):
        text = "Work.\n\\boxed{7}\n#### Final check"
        self.assertEqual(extract_final_answer(text), "7")

    def test_gsm8k_reference_marker_on_last_line(self):
        text = "Natalia sold 48/2 = 24 clips in May.\n48+24 = 72\n#### 72"
        self.assertEqual(extract_final_answer(text), "72")

    def test_gsm8k_marker_with_trailing_whitespace(self):
        self.assertEqual(extract_final_answer("Total 5,600\n#### 5,600\n\n"), "5,600")

    def test_heading_without_boxed_falls_through_to_final_answer_patterns(self):
        text = "#### Step 1\nCompute the sum.\nFinal answer: 12"
        self.assertEqual(extract_final_answer(text), "12")


class CommittedAnswerExtractionTest(unittest.TestCase):
    def test_last_closed_box_wins(self):
        self.assertEqual(extract_committed_answer(r"\boxed{4} or maybe \boxed{5}"), "5")

    def test_nested_braces(self):
        self.assertEqual(extract_committed_answer(r"so \boxed{\frac{1}{2}}."), r"\frac{1}{2}")

    def test_cut_off_box_is_not_an_answer(self):
        self.assertIsNone(extract_committed_answer(r"so x = \boxed{3"))
        self.assertEqual(extract_final_answer(r"so x = \boxed{3"), "3")

    def test_unfinished_sentence_is_not_an_answer(self):
        self.assertIsNone(extract_committed_answer("Let me compute.\nWe get 70 so far, then subtract"))

    def test_hash_marker_and_phrases_are_ignored(self):
        self.assertIsNone(extract_committed_answer("48+24 = 72\n#### 72"))
        self.assertIsNone(extract_committed_answer("Final answer: 12"))

    def test_closed_box_after_cut_off_box(self):
        self.assertEqual(closed_boxed_answers(r"\boxed{2} and y=\boxed{3}"), ["2", "3"])


class GraderTest(unittest.TestCase):
    """Lenient behavior is frozen (it is the training reward); strict is report-only."""

    CASES = [
        # text, reference, lenient, strict
        ("Let me compute.\nWe get 70 so far, then subtract", "70", 1.0, 0.0),
        (r"so x = \boxed{3", "3", 1.0, 0.0),
        (r"\boxed{2/4}", r"\frac{1}{2}", 1.0, 1.0),
        (r"\boxed{(-3, 2)}", "(2,-3)", 0.0, 0.0),
        (r"\boxed{[1,2)}", "(1,2]", 0.0, 0.0),
        (r"\boxed{4} or maybe \boxed{5}", "4", 0.0, 0.0),
        (r"x=\boxed{2} and y=\boxed{3}", "2, 3", 0.0, 0.0),
        ("#### Step 1\nWork.\nThe answer is \\boxed{7}.", "7", 1.0, 1.0),
        ("", "1", 0.0, 0.0),
    ]

    def test_lenient_and_strict(self):
        from src.training.rl.smoke import grade_solution, grade_solution_strict

        for text, reference, lenient, strict in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(grade_solution(text, reference), lenient)
                self.assertEqual(grade_solution_strict(text, reference), strict)


if __name__ == "__main__":
    unittest.main()
