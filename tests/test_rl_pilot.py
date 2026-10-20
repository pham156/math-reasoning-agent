import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from src.training.rl.pilot import split_rows


class SplitTest(unittest.TestCase):
    def test_development_reserved_reproducibly_and_balanced(self):
        rows = [{"source_index": i, "type": str(i % 2), "level": str(i % 3)} for i in range(60)]
        training, dev = split_rows(rows, 30, 12)
        self.assertEqual((training, dev), split_rows(rows, 30, 12))
        self.assertFalse({r["source_index"] for r in training} & {r["source_index"] for r in dev})
        self.assertEqual(len(training), 30)
        counts = {(r["type"], r["level"]): 0 for r in rows}
        for row in dev:
            counts[(row["type"], row["level"])] += 1
        self.assertEqual(set(counts.values()), {2})

    def test_insufficient_data_fails(self):
        with self.assertRaises(ValueError):
            split_rows([], 30, 12)


class ResumeCheckTest(unittest.TestCase):
    def test_boundary_check_rejects_pre_restart_divergence(self):
        import torch
        from safetensors.torch import save_file
        from src.evaluation.check_rl_resume import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in (("continuous", 128), ("resumed", 64)):
                run = root / name
                checkpoint = run / "checkpoints/checkpoint-4"
                checkpoint.mkdir(parents=True)
                save_file({"weight": torch.ones(2)}, str(checkpoint / "model.safetensors"))
                (run / "rollouts.jsonl").write_text((json.dumps({"completion": "same"}) + "\n") * count)
                (run / "progress.json").write_text(json.dumps({"global_step": 4}))
                (run / "run_config.json").write_text(json.dumps({"determinism": {"requested": True, "algorithms_enabled": True}}))
            with patch("sys.argv", ["check", str(root), "--boundary_only"]), contextlib.redirect_stdout(io.StringIO()):
                main()
                self.assertTrue(json.loads((root / "boundary_check.json").read_text())["passed"])
                save_file({"weight": torch.ones(2) + 1e-5}, str(root / "resumed/checkpoints/checkpoint-4/model.safetensors"))
                with self.assertRaises(SystemExit):
                    main()


if __name__ == "__main__":
    unittest.main()
