import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_benchmarks import BENCHMARKS, validate_aime
from src.evaluation.benchmarks import _aime_rows_from_raw, load_benchmark_examples


class AimeSetupTests(unittest.TestCase):
    def rows(self):
        return [{"id": str(i), "problem": f"Problem {i}", "answer": str(i)} for i in range(30)]

    def test_validation_and_fingerprint(self):
        rows = self.rows()
        result = validate_aime({"test": rows}, 30)
        self.assertEqual(result["validated_rows"], 30)
        self.assertEqual(result, validate_aime({"test": rows}, 30))
        rows[0]["answer"] = "999"
        self.assertNotEqual(result, validate_aime({"test": rows}, 30))

    def test_rejects_invalid_data(self):
        for field, value in [("id", "1"), ("problem", "Problem 1"), ("answer", "1000"),
                             ("answer", "1.5"), ("problem", ""), ("answer", "")]:
            rows = self.rows()
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_aime({"test": rows}, 30)
        with self.assertRaises(ValueError):
            validate_aime({"test": self.rows()[:29]}, 30)

    def test_specs_pin_both_years(self):
        for year in (2025, 2026):
            spec = next(s for s in BENCHMARKS if s["target"] == f"AIME_{year}")
            self.assertEqual(spec["expected_rows"], 30)
            self.assertEqual(len(spec["kwargs"]["revision"]), 40)

    def test_rows_preserve_year_and_answer(self):
        row = _aime_rows_from_raw(self.rows()[:1], "aime2026", "2026")[0]
        self.assertEqual(row["ground_truth_answer_override"], "0")
        self.assertEqual(row["benchmark_example_metadata"]["year"], "2026")
        self.assertEqual(row["source_id"], "0")

    def test_missing_2026_gives_download_command(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.evaluation.benchmarks.data_dir", return_value=Path(directory) / "absent"):
                with self.assertRaisesRegex(RuntimeError, "--benchmarks AIME_2025 AIME_2026"):
                    load_benchmark_examples("aime2026", "test", None, 0, "first")

    def test_2026_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.evaluation.benchmarks.data_dir", return_value=Path(directory)), \
                    patch("src.evaluation.benchmarks.load_aime_year", return_value=([], {})) as loader:
                load_benchmark_examples("aime2026", "test", None, 0, "first")
                self.assertEqual(loader.call_args.kwargs["dataset_id"], "math-ai/aime26")
                self.assertEqual(loader.call_args.kwargs["default_year"], "2026")


if __name__ == "__main__":
    unittest.main()
