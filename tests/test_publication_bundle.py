import re
import shutil
import subprocess
from pathlib import Path
import unittest
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


class PublicationBundleTests(unittest.TestCase):
    def test_overleaf_archives_match_published_sources(self):
        for name, archive in (("midterm", "midterm_overleaf.zip"),
                              ("final", "final_overleaf.zip"),
                              ("results_compendium", "results_overleaf.zip")):
            directory = ROOT / "docs" / name
            with self.subTest(document=name), ZipFile(directory / archive) as bundle:
                self.assertIsNone(bundle.testzip())
                for member in bundle.namelist():
                    path = directory / member
                    self.assertTrue(path.resolve().is_relative_to(directory.resolve()))
                    self.assertEqual(bundle.read(member), path.read_bytes(), member)

    def test_manuscript_inputs_and_figures_are_self_contained(self):
        for name in ("midterm", "final", "results_compendium"):
            directory = ROOT / "docs" / name
            pending = [directory / "main.tex"]
            seen = set()
            while pending:
                path = pending.pop()
                if path in seen:
                    continue
                seen.add(path)
                text = path.read_text()
                for target in re.findall(r"\\input\{([^{}]+)\}", text):
                    pending.append(directory / (target if target.endswith(".tex") else target + ".tex"))
                for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text):
                    candidates = [directory / target, directory / "figures" / target]
                    candidates += [p.with_suffix(".pdf") for p in candidates if not p.suffix]
                    self.assertTrue(any(p.is_file() for p in candidates), f"Missing {name} figure: {target}")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "Requires a Git checkout and host Git")
    def test_private_artifacts_are_ignored_but_documents_are_publishable(self):
        for path in (".agent/CONTINUITY.md", "AGENTS.md", "scripts/rl/run_algorithm_smokes.slurm",
                     "outputs/example.jsonl", "logs/example.log", "checkpoints/weights.safetensors"):
            result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
            self.assertEqual(result.returncode, 0, path)
        for path in ("docs/midterm/midterm_update.pdf", "docs/final/final_report.pdf",
                     "docs/poster/research_poster.pdf", "docs/results_compendium/completed_results.pdf"):
            result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
            self.assertEqual(result.returncode, 1, path)


if __name__ == "__main__":
    unittest.main()
