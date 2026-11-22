from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.core.config import data_dir


@dataclass
class BenchmarkMetadata:
    benchmark: str
    display_name: str
    dataset_id: str
    split: str
    limit: int
    sample_seed: int
    sampling: str
    source: str
    note: str = ""


def _as_int_level(value: Any) -> str:
    if value is None or value == "":
        return ""
    text = str(value)
    return text if text.lower().startswith("level") else f"Level {text}"


def _select_rows(
    rows: list[dict],
    limit: int | None,
    sample_seed: int,
    sampling: str,
    start_index: int = 0,
) -> list[dict]:
    if start_index < 0:
        raise ValueError(f"start_index must be nonnegative, got {start_index}")
    rows = rows[start_index:]
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows
    if sampling == "first":
        return rows[:limit]
    if sampling != "random":
        raise ValueError(f"Unknown sampling mode {sampling!r}; expected first or random.")
    indices = list(range(len(rows)))
    random.Random(sample_seed).shuffle(indices)
    selected = sorted(indices[:limit])
    return [rows[index] for index in selected]


def _standard_row(
    *,
    benchmark: str,
    index: int,
    source_id: str,
    problem: str,
    solution: str,
    answer: str,
    category: str = "",
    level: str = "",
    metadata: dict | None = None,
) -> dict:
    return {
        "benchmark": benchmark,
        "problem_id": f"{benchmark}-{index}",
        "problem_index": index,
        "source_id": source_id,
        "type": category,
        "category": category,
        "level": level,
        "problem": problem,
        "ground_truth_solution": solution,
        "ground_truth_answer_override": answer,
        "benchmark_example_metadata": metadata or {},
    }


def load_math_benchmark(
    split: str,
    limit: int | None,
    sample_seed: int,
    sampling: str,
    start_index: int = 0,
) -> tuple[list[dict], dict]:
    from datasets import load_from_disk

    dataset_path = data_dir("MATH")
    dataset = load_from_disk(str(dataset_path))
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found in {dataset_path}. Available splits: {list(dataset.keys())}")
    rows = []
    for index, row in enumerate(dataset[split]):
        rows.append(
            _standard_row(
                benchmark="math",
                index=index,
                source_id=f"{split}-{index}",
                problem=row.get("problem", ""),
                solution=row.get("solution", ""),
                answer="",
                category=row.get("type", row.get("category", "")),
                level=row.get("level", ""),
            )
        )
    selected = _select_rows(
        rows,
        limit=20 if limit is None else limit,
        sample_seed=sample_seed,
        sampling=sampling,
        start_index=start_index,
    )
    metadata = BenchmarkMetadata(
        benchmark="math",
        display_name="MATH local subset",
        dataset_id=str(dataset_path),
        split=split,
        limit=len(selected),
        sample_seed=sample_seed,
        sampling=sampling,
        source="local_disk",
        note="Uses the local MATH dataset in scratch; not the fixed MATH-500 preset.",
    )
    result_metadata = asdict(metadata)
    result_metadata["start_index"] = start_index
    return selected, result_metadata


def load_math500(
    limit: int | None,
    sample_seed: int,
    sampling: str,
    start_index: int = 0,
) -> tuple[list[dict], dict]:
    from datasets import load_dataset, load_from_disk

    dataset_id = "HuggingFaceH4/MATH-500"
    local_path = data_dir("MATH-500")
    source = "local_disk" if local_path.exists() else "huggingface"
    dataset = load_from_disk(str(local_path))["test"] if local_path.exists() else load_dataset(dataset_id, split="test")
    rows = []
    for index, row in enumerate(dataset):
        rows.append(
            _standard_row(
                benchmark="math500",
                index=index,
                source_id=str(row.get("unique_id", f"test-{index}")),
                problem=row.get("problem", ""),
                solution=row.get("solution", ""),
                answer=row.get("answer", ""),
                category=row.get("subject", ""),
                level=_as_int_level(row.get("level", "")),
                metadata={"unique_id": row.get("unique_id"), "subject": row.get("subject")},
            )
        )
    selected = _select_rows(
        rows,
        limit=500 if limit is None else limit,
        sample_seed=sample_seed,
        sampling=sampling,
        start_index=start_index,
    )
    metadata = BenchmarkMetadata(
        benchmark="math500",
        display_name="MATH-500",
        dataset_id=dataset_id,
        split="test",
        limit=len(selected),
        sample_seed=sample_seed,
        sampling=sampling,
        source=source,
        note="Fixed 500-problem subset from the MATH benchmark commonly reported in reasoning-model papers.",
    )
    result_metadata = asdict(metadata)
    result_metadata["start_index"] = start_index
    return selected, result_metadata


def load_gsm8k100(
    limit: int | None,
    sample_seed: int,
    sampling: str,
    start_index: int = 0,
) -> tuple[list[dict], dict]:
    from datasets import load_dataset, load_from_disk

    dataset_id = "openai/gsm8k"
    local_path = data_dir("GSM8K")
    source = "local_disk" if local_path.exists() else "huggingface"
    dataset = load_from_disk(str(local_path))["test"] if local_path.exists() else load_dataset(dataset_id, "main", split="test")
    rows = []
    for index, row in enumerate(dataset):
        answer = row.get("answer", "")
        final = answer.split("####")[-1].strip() if "####" in answer else ""
        rows.append(
            _standard_row(
                benchmark="gsm8k100",
                index=index,
                source_id=f"test-{index}",
                problem=row.get("question", ""),
                solution=answer,
                answer=final,
                category="GSM8K",
                level="grade_school",
            )
        )
    selected = _select_rows(
        rows,
        limit=100 if limit is None else limit,
        sample_seed=sample_seed,
        sampling=sampling,
        start_index=start_index,
    )
    metadata = BenchmarkMetadata(
        benchmark="gsm8k100",
        display_name="GSM8K-100",
        dataset_id=dataset_id,
        split="test",
        limit=len(selected),
        sample_seed=sample_seed,
        sampling=sampling,
        source=source,
        note="Deterministic 100-example subset of GSM8K test. Use the same sample_seed for reproducibility.",
    )
    result_metadata = asdict(metadata)
    result_metadata["start_index"] = start_index
    return selected, result_metadata


def _aime_rows_from_raw(raw_rows: list[dict], benchmark: str, default_year: str) -> list[dict]:
    rows = []
    for index, row in enumerate(raw_rows):
        problem = row.get("problem", row.get("question", row.get("prompt", "")))
        answer = str(row.get("answer", row.get("final_answer", row.get("target", ""))))
        solution = row.get("solution", row.get("answer_with_solution", answer))
        year = str(row.get("year", row.get("contest_year", default_year)))
        contest = row.get("contest", row.get("exam", f"AIME {year}".strip()))
        rows.append(
            _standard_row(
                benchmark=benchmark,
                index=index,
                source_id=str(row.get("id", row.get("unique_id", f"{benchmark}-{index}"))),
                problem=problem,
                solution=str(solution),
                answer=answer,
                category=str(contest),
                level="AIME",
                metadata={"year": year, "contest": contest},
            )
        )
    return rows


def load_aime_year(
    benchmark: str,
    target_dir: str,
    dataset_id: str,
    split: str,
    default_year: str,
    limit: int | None,
    sample_seed: int,
    sampling: str,
    start_index: int = 0,
) -> tuple[list[dict], dict]:
    from datasets import load_dataset, load_from_disk

    local_path = data_dir(target_dir)
    source = "local_disk" if local_path.exists() else "huggingface"
    if local_path.exists():
        dataset = load_from_disk(str(local_path))
        selected_split = split if hasattr(dataset, "keys") and split in dataset else next(iter(dataset.keys()))
        raw_rows = list(dataset[selected_split])
    else:
        dataset = load_dataset(dataset_id)
        selected_split = split if split in dataset else next(iter(dataset.keys()))
        raw_rows = list(dataset[selected_split])
    rows = _aime_rows_from_raw(raw_rows, benchmark=benchmark, default_year=default_year)
    selected = _select_rows(rows, limit=limit, sample_seed=sample_seed, sampling=sampling, start_index=start_index)
    metadata = BenchmarkMetadata(
        benchmark=benchmark,
        display_name=f"AIME {default_year}",
        dataset_id=dataset_id,
        split=selected_split,
        limit=len(selected),
        sample_seed=sample_seed,
        sampling=sampling,
        source=source,
        note=f"{default_year} AIME subset.",
    )
    result_metadata = asdict(metadata)
    result_metadata["start_index"] = start_index
    return selected, result_metadata


def _load_aime_from_disk(path: Path, split: str | None) -> list[dict]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    if hasattr(dataset, "keys"):
        selected_split = split if split in dataset else next(iter(dataset.keys()))
        return list(dataset[selected_split])
    return list(dataset)


def _load_aime_from_hf(dataset_id: str, split: str | None) -> list[dict]:
    from datasets import load_dataset

    loaded = load_dataset(dataset_id)
    if hasattr(loaded, "keys"):
        selected_split = split if split in loaded else next(iter(loaded.keys()))
        return list(loaded[selected_split])
    return list(loaded)


def load_aime(
    limit: int | None,
    sample_seed: int,
    sampling: str,
    split: str,
    aime_dataset_path: str | None = None,
    aime_hf_dataset: str | None = None,
    start_index: int = 0,
) -> tuple[list[dict], dict]:
    source = ""
    dataset_id = ""
    raw_rows: list[dict] = []
    local_candidates = []
    if aime_dataset_path:
        local_candidates.append(Path(aime_dataset_path).expanduser())
    local_candidates.append(data_dir("AIME"))

    for candidate in local_candidates:
        if candidate.exists():
            raw_rows = _load_aime_from_disk(candidate, split)
            source = "local_disk"
            dataset_id = str(candidate)
            break

    if not raw_rows and aime_hf_dataset:
        raw_rows = _load_aime_from_hf(aime_hf_dataset, split)
        source = "huggingface"
        dataset_id = aime_hf_dataset

    if not raw_rows:
        raise RuntimeError(
            "AIME benchmark requested, but no AIME dataset is available. "
            "Provide --aime_dataset_path under scratch or --aime_hf_dataset DATASET_ID."
        )

    rows = _aime_rows_from_raw(raw_rows, benchmark="aime", default_year="")

    selected = _select_rows(rows, limit=limit, sample_seed=sample_seed, sampling=sampling, start_index=start_index)
    metadata = BenchmarkMetadata(
        benchmark="aime",
        display_name="AIME subset",
        dataset_id=dataset_id,
        split=split,
        limit=len(selected),
        sample_seed=sample_seed,
        sampling=sampling,
        source=source,
        note="Optional AIME subset; exact composition depends on the provided local/HF dataset.",
    )
    result_metadata = asdict(metadata)
    result_metadata["start_index"] = start_index
    return selected, result_metadata


def load_benchmark_examples(
    benchmark: str,
    split: str,
    limit: int | None,
    sample_seed: int,
    sampling: str,
    aime_dataset_path: str | None = None,
    aime_hf_dataset: str | None = None,
    start_index: int = 0,
) -> tuple[list[dict], dict]:
    if benchmark == "math":
        return load_math_benchmark(
            split=split,
            limit=limit,
            sample_seed=sample_seed,
            sampling=sampling,
            start_index=start_index,
        )
    if benchmark == "math500":
        return load_math500(limit=limit or 500, sample_seed=sample_seed, sampling=sampling, start_index=start_index)
    if benchmark == "gsm8k100":
        return load_gsm8k100(
            limit=limit or 100,
            sample_seed=sample_seed,
            sampling=sampling,
            start_index=start_index,
        )
    if benchmark == "aime2024":
        return load_aime_year(
            benchmark="aime2024",
            target_dir="AIME_2024",
            dataset_id="HuggingFaceH4/aime_2024",
            split="train",
            default_year="2024",
            limit=limit,
            sample_seed=sample_seed,
            sampling=sampling,
            start_index=start_index,
        )
    if benchmark == "aime2025":
        return load_aime_year(
            benchmark="aime2025",
            target_dir="AIME_2025",
            dataset_id="math-ai/aime25",
            split="test",
            default_year="2025",
            limit=limit,
            sample_seed=sample_seed,
            sampling=sampling,
            start_index=start_index,
        )
    if benchmark == "aime":
        return load_aime(
            limit=limit,
            sample_seed=sample_seed,
            sampling=sampling,
            split=split,
            aime_dataset_path=aime_dataset_path,
            aime_hf_dataset=aime_hf_dataset,
            start_index=start_index,
        )
    raise ValueError(f"Unknown benchmark preset {benchmark!r}.")


# Post-release competitions for contamination checks. Qwen2.5-Math was released in
# September 2024; both contests were held afterwards. Revisions are pinned because
# MathArena may revise statements (its AIME 2025 problem 2 differs from math-ai/aime25).
MATHARENA_AIME = {
    "aime2025": {"year": "2025", "dataset_id": "MathArena/aime_2025", "revision": "c94da77eb22bbd6439e62a323bec18493a421302", "held": "February 2025"},
    "aime2026": {"year": "2026", "dataset_id": "MathArena/aime_2026", "revision": "d61f486805306e6e6b5913504c07763a99e1b112", "held": "February 2026"},
}


def matharena_aime_rows(raw_rows: list[dict], benchmark: str) -> list[dict]:
    """Validate one MathArena AIME year and convert it to standard benchmark rows."""
    if len(raw_rows) != 30:
        raise ValueError(f"{benchmark}: expected 30 AIME problems, got {len(raw_rows)}")
    rows = []
    for raw in raw_rows:
        index = int(raw["problem_idx"])
        problem = str(raw["problem"]).strip()
        answer = raw["answer"]
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer <= 999:
            raise ValueError(f"{benchmark} problem {index}: AIME answers are integers 0-999, got {answer!r}")
        if not problem:
            raise ValueError(f"{benchmark} problem {index}: empty statement")
        topics = raw.get("problem_type") or []
        rows.append(
            _standard_row(
                benchmark=benchmark,
                index=index,
                source_id=f"{benchmark}-{index}",
                problem=problem,
                solution="",
                answer=str(answer),
                category=f"AIME {MATHARENA_AIME[benchmark]['year']}",
                level="AIME",
                metadata={"problem_type": list(topics)},
            )
        )
    if sorted(row["problem_index"] for row in rows) != list(range(1, 31)):
        raise ValueError(f"{benchmark}: problem_idx must be exactly 1..30")
    return sorted(rows, key=lambda row: row["problem_index"])


def load_matharena_aime(benchmark: str) -> tuple[list[dict], dict]:
    """Load a pinned MathArena AIME revision, caching it under $DATA_DIR."""
    import json

    from datasets import load_dataset, load_from_disk

    spec = MATHARENA_AIME[benchmark]
    local_path = data_dir(f"MathArena_{benchmark}")
    record_path = local_path / "download_record.json"
    if local_path.exists():
        record = json.loads(record_path.read_text())
        if record["revision"] != spec["revision"]:
            raise ValueError(f"{local_path} holds revision {record['revision']}, expected {spec['revision']}")
        dataset = load_from_disk(str(local_path / "dataset"))
        source = "local_disk"
    else:
        dataset = load_dataset(spec["dataset_id"], revision=spec["revision"])
        local_path.mkdir(parents=True)
        dataset.save_to_disk(str(local_path / "dataset"))
        record_path.write_text(json.dumps({"dataset_id": spec["dataset_id"], "revision": spec["revision"]}, indent=2) + "\n")
        source = "huggingface"
    rows = matharena_aime_rows(list(dataset["train"]), benchmark)
    metadata = asdict(
        BenchmarkMetadata(
            benchmark=benchmark,
            display_name=f"AIME {spec['year']}",
            dataset_id=spec["dataset_id"],
            split="train",
            limit=len(rows),
            sample_seed=0,
            sampling="all",
            source=source,
            note=f"All 30 AIME {spec['year']} I/II problems (held {spec['held']}), after the Qwen2.5-Math release.",
        )
    )
    metadata.update(revision=spec["revision"], license="cc-by-nc-sa-4.0", answer_format="integer 0-999")
    return rows, metadata
