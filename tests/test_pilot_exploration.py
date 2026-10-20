import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from src.evaluation.pilot_exploration import pass_at_k, stop_ids
from src.evaluation.summarize_pilot_exploration import read_rows


class ExplorationTests(unittest.TestCase):
    def test_pass_estimator(self):
        self.assertEqual(pass_at_k(8, 0, 8), 0)
        self.assertEqual(pass_at_k(8, 8, 1), 1)
        self.assertEqual(pass_at_k(8, 1, 1), 0.125)
        self.assertEqual(pass_at_k(8, 1, 4), 0.5)
        self.assertEqual(pass_at_k(8, 1, 8), 1)
        self.assertAlmostEqual(pass_at_k(8, 2, 4), 1 - 15 / 70)
        with self.assertRaises(ValueError):
            pass_at_k(8, 1, 9)

    def test_stop_union(self):
        tokenizer = SimpleNamespace(eos_token_id=151643, unk_token_id=None, convert_tokens_to_ids=lambda _: 151645)
        self.assertEqual(stop_ids(tokenizer, SimpleNamespace(eos_token_id=151643)), [151643, 151645])
        self.assertEqual(stop_ids(tokenizer, SimpleNamespace(eos_token_id=[151643, 151645])), [151643, 151645])

    def test_missing_chat_end_rejected(self):
        tokenizer = SimpleNamespace(eos_token_id=1, unk_token_id=0, convert_tokens_to_ids=lambda _: 0)
        with self.assertRaises(ValueError):
            stop_ids(tokenizer, SimpleNamespace(eos_token_id=1))

    def test_incomplete_and_duplicate_shards_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sampled/baseline"
            target.mkdir(parents=True)
            row = {"source_index": 7, "candidates": [{}] * 8}
            path = target / "shard_0.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                read_rows(root, "sampled", "baseline", 2, 8)
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                read_rows(root, "sampled", "baseline", 2, 8)
            second = {"source_index": 8, "candidates": [{}] * 8}
            path.write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n")
            self.assertEqual(set(read_rows(root, "sampled", "baseline", 2, 8)), {7, 8})


if __name__ == "__main__":
    unittest.main()
