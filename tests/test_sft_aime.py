import unittest

from src.training.rl.rft_data import clean_ending, select_targets, shard_bounds


def candidate(index, text, correct=True, truncated=False, terminated=True):
    return {"sample_index": index, "raw_solution": text, "correct": correct, "truncated": truncated, "terminated": terminated, "generated_tokens": len(text)}


class RFTSelectionTest(unittest.TestCase):
    def test_filters_incorrect_truncated_and_duplicates(self):
        rows = [{"source_index": 1, "problem": "p", "candidates": [
            candidate(3, "b  answer"), candidate(0, "a"), candidate(1, "x", correct=False),
            candidate(2, "a "), candidate(4, "t", truncated=True, terminated=False), candidate(5, "b answer")]}]
        data, stats = select_targets(rows, max_per_problem=4)
        self.assertEqual([d["sample_index"] for d in data], [0, 3])
        self.assertEqual(stats["duplicates"], 2)
        self.assertEqual(stats["truncated_correct"], 1)
        self.assertEqual(stats["correct"], 5)
        self.assertEqual(stats["problems_with_target"], 1)

    def test_cap_and_unsolved_problem(self):
        rows = [{"source_index": 1, "problem": "p", "candidates": [candidate(i, f"s{i}") for i in range(8)]},
                {"source_index": 2, "problem": "q", "candidates": [candidate(i, "w", correct=False) for i in range(8)]}]
        data, stats = select_targets(rows, max_per_problem=4)
        self.assertEqual([d["sample_index"] for d in data], [0, 1, 2, 3])
        self.assertEqual(stats["problems_with_target"], 1)
        self.assertEqual(stats["targets"], 4)

    def test_shards_partition_training_set(self):
        spans = [shard_bounds(i, 16) for i in range(16)]
        self.assertEqual(spans[0], (0, 32))
        self.assertEqual(spans[-1], (480, 512))
        with self.assertRaises(ValueError):
            shard_bounds(0, 7)
        with self.assertRaises(ValueError):
            shard_bounds(16, 16)


class CleanEndingTest(unittest.TestCase):
    def test_clean_and_unclean_endings(self):
        self.assertTrue(clean_ending("So x = 2, giving \\(\\boxed{\\frac{1}{2}}\\)."))
        self.assertFalse(clean_ending("\\boxed{3}\nPropel\nWhat is the value of x? Let us solve it again from the start."))
        self.assertFalse(clean_ending("\\boxed{3} \u0631\u0648\u0645"))
        self.assertFalse(clean_ending("no boxed answer here"))

    def test_clean_filter_counts_rejections(self):
        rows = [{"source_index": 1, "problem": "p", "candidates": [candidate(0, "\\boxed{1}" + " junk" * 20), candidate(1, "\\boxed{1}.")]}]
        data, stats = select_targets(rows, 4, clean=True)
        self.assertEqual([d["sample_index"] for d in data], [1])
        self.assertEqual(stats["unclean_correct"], 1)


if __name__ == "__main__":
    unittest.main()
