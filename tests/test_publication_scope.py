from pathlib import Path
import re
import tempfile
import unittest
from zipfile import ZipFile

from scripts.paper.package_final import package
from scripts.paper.prepare_final import PUBLICATION_METHODS, REPORT_FIGURES, initial_training_points, reference_reward_rows
from scripts.paper.prepare_poster import METHODS


class PublicationScopeTests(unittest.TestCase):
    def test_initial_loss_is_dense_and_can_be_negative(self):
        history = [{"step": step, "loss": -.001 * step} for step in range(1, 129)]
        self.assertEqual(len(initial_training_points(history, "loss")), 128)
        with self.assertRaises(ValueError):
            initial_training_points(history[:-1], "loss")

    def test_only_requested_methods_are_in_publication(self):
        self.assertEqual(PUBLICATION_METHODS, ("grpo", "dr_grpo"))
        self.assertEqual(METHODS, PUBLICATION_METHODS)

    def test_fixed_reference_reward_is_not_a_cross_run_mean(self):
        runs = [{"settings": {"reward_design": design, "seed": seed}, "accuracy": score,
                 "avg_generated_tokens": 700}
                for design in ("correctness", "format_bonus", "length_penalty")
                for seed, score in ((0, .75), (1, .1))]
        self.assertEqual([r[1] for r in reference_reward_rows(runs)], ["75.0"] * 3)
        with self.assertRaises(ValueError):
            reference_reward_rows([r for r in runs if r["settings"]["seed"] == 1])
        with self.assertRaises(ValueError):
            reference_reward_rows(runs + [runs[0]])

    def test_manuscript_has_no_excluded_algorithms_or_sparse_figures(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs/final/main.tex").read_text()
        self.assertNotRegex(text.lower(), r"dapo|gspo|seed[ -]?[01]|paired.seed")
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text)
        self.assertEqual(set(graphics), {f"{name}.pdf" for name in REPORT_FIGURES})
        self.assertNotIn("generated/seeds.tex", text)

    def test_archive_excludes_unreferenced_legacy_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "figures").mkdir()
            (root / "generated").mkdir()
            source = "\\input{generated/methods.tex}\n" + "\n".join(
                "\\includegraphics{" + name + ".pdf}" for name in REPORT_FIGURES)
            (root / "main.tex").write_text(source)
            (root / "generated/methods.tex").write_text("focused table")
            (root / "generated/seeds.tex").write_text("legacy comparison")
            for name in (*REPORT_FIGURES, "followup_ablations"):
                (root / "figures" / f"{name}.pdf").write_bytes(b"test figure")
            package(root)
            with ZipFile(root / "final_overleaf.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"main.tex", "generated/methods.tex",
                    "figures/method_dynamics.pdf", "figures/extended_training.pdf"})


if __name__ == "__main__":
    unittest.main()
