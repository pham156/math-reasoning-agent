import unittest

from scripts.paper.prepare_poster import A0_MM, ecdf, transition_matrix


class PosterTests(unittest.TestCase):
    def test_exact_a0_landscape(self):
        self.assertEqual(A0_MM, (1189, 841))

    def test_transition_preserves_question_counts_and_direction(self):
        before = [{"source_index": i, "prompt_hash": str(i), "num_correct": c} for i, c in enumerate([0, 2, 8])]
        after = [{"source_index": i, "prompt_hash": str(i), "num_correct": c} for i, c in enumerate([8, 3, 8])]
        matrix = transition_matrix(before, after)
        self.assertEqual(sum(map(sum, matrix)), 3)
        self.assertEqual(matrix[0][8], 1)
        self.assertEqual(matrix[2][3], 1)
        self.assertEqual(matrix[8][8], 1)

    def test_transition_rejects_different_questions_prompts_or_sample_count(self):
        row = {"source_index": 1, "prompt_hash": "a", "num_correct": 4}
        for change in ({"source_index": 2}, {"prompt_hash": "b"}, {"num_correct": 9}):
            with self.assertRaises(ValueError):
                transition_matrix([row], [{**row, **change}])

    def test_ecdf_uses_observations_not_fitted_values(self):
        xs, ys = ecdf([4, 2, 4, 1])
        self.assertEqual(xs, [1, 2, 4, 4])
        self.assertEqual(ys, [0.25, 0.5, 0.75, 1])
        with self.assertRaises(ValueError):
            ecdf([])


if __name__ == "__main__":
    unittest.main()
