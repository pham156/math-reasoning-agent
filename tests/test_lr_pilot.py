import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.core.config import data_dir, model_dir
from src.training.rl.pilot import main


class LearningRateResumeTests(unittest.TestCase):
    def check_resume(self, saved_rate, requested_rate, should_fail):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = {"algorithm": "grpo", "model_path": str(model_dir("Qwen2.5-Math-1.5B")), "split_dir": str(data_dir("rl_1p5b_pilot_v1")), "seed": 0, "max_steps": 128, "train_samples": 512, "eval_samples": 20, "max_completion_length": 3000, "max_prompt_length": 1024, "deterministic_training": True}
            if saved_rate is not None:
                previous["learning_rate"] = saved_rate
            (root / "progress.json").write_text(json.dumps({"status": "paused", "checkpoint": str(root / "checkpoint-4")}))
            (root / "run_config.json").write_text(json.dumps({"arguments": previous}))
            argv = ["pilot", "--output_dir", str(root), "--eval_samples", "20", "--learning_rate", str(requested_rate), "--resume"]
            with patch("sys.argv", argv), patch("src.training.rl.pilot.ensure_scratch_path", side_effect=lambda p, _: Path(p)), patch("src.training.rl.pilot.run") as run, contextlib.redirect_stderr(io.StringIO()):
                if should_fail:
                    with self.assertRaises(SystemExit):
                        main()
                    run.assert_not_called()
                else:
                    main()
                    self.assertEqual(run.call_args.args[0].learning_rate, requested_rate)

    def test_old_default_checkpoint_compatible(self):
        self.check_resume(None, 1e-6, False)

    def test_learning_rate_change_rejected(self):
        self.check_resume(None, 2e-7, True)
        self.check_resume(2e-7, 5e-7, True)

    def test_same_tuned_rate_resumes(self):
        self.check_resume(2e-7, 2e-7, False)


if __name__ == "__main__":
    unittest.main()
