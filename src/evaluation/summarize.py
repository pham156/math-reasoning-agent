from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.core.config import output_dir


def resolve_input_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_absolute():
        return requested
    parts = requested.parts
    if parts and parts[0] == "outputs":
        return output_dir(Path(*parts[1:]) if len(parts) > 1 else "")
    return requested


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def load_jsonl(path: Path) -> tuple[list[dict], dict | None]:
    rows = []
    metadata = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "run_metadata":
                metadata = item
                continue
            rows.append(item)
    return rows, metadata


def load_rows(path: Path) -> list[dict]:
    rows, _ = load_jsonl(path)
    return rows


def accuracy(rows: list[dict]) -> float:
    return safe_mean([1.0 if row.get("exact_match") else 0.0 for row in rows])


def grouped_accuracy(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "") or "unknown")].append(row)
    return {
        group: {
            "count": len(group_rows),
            "accuracy": accuracy(group_rows),
        }
        for group, group_rows in sorted(groups.items())
    }


def verifier_summary(rows: list[dict]) -> dict:
    used = [row for row in rows if row.get("verifier_verdict") is not None]
    if not used:
        return {"verifier_used": False}
    scores = [float(row.get("verifier_score") or 0.0) for row in used]
    verdicts: dict[str, int] = defaultdict(int)
    for row in used:
        verdicts[str(row.get("verifier_verdict") or "unknown")] += 1
    return {
        "verifier_used": True,
        "num_verified": len(used),
        "average_verifier_score": safe_mean(scores),
        "verdict_counts": dict(sorted(verdicts.items())),
    }


def selection_summary(rows: list[dict]) -> dict:
    selected_rows = [row for row in rows if row.get("selection_metadata")]
    verifier_selected = [
        row
        for row in selected_rows
        if row.get("selection_metadata", {}).get("selection") in {"verifier", "consensus_verifier"}
    ]
    if not verifier_selected:
        return {"selection_used": False}

    selected_scores = [
        float(row.get("selection_metadata", {}).get("selected_verifier_score") or 0.0)
        for row in verifier_selected
    ]
    rows_with_available_correct = [
        row for row in verifier_selected if row.get("selection_metadata", {}).get("any_candidate_correct")
    ]
    success_rows = [
        row for row in rows_with_available_correct if row.get("selection_metadata", {}).get("selected_candidate_correct")
    ]
    correct_over_incorrect_rows = [
        row
        for row in verifier_selected
        if row.get("selection_metadata", {}).get("verifier_selected_correct_over_incorrect")
    ]
    fixed_first_rows = [
        row for row in verifier_selected if row.get("selection_metadata", {}).get("verifier_fixed_first_candidate")
    ]
    missed_available_rows = [
        row for row in verifier_selected if row.get("selection_metadata", {}).get("verifier_missed_available_correct")
    ]
    mixed_rows = [
        row
        for row in verifier_selected
        if any(candidate.get("exact_match") for candidate in row.get("candidate_solutions", []))
        and any(not candidate.get("exact_match") for candidate in row.get("candidate_solutions", []))
    ]
    selection_modes: dict[str, int] = defaultdict(int)
    for row in verifier_selected:
        selection_modes[str(row.get("selection_metadata", {}).get("selection") or "unknown")] += 1
    return {
        "selection_used": True,
        "selection_modes": dict(sorted(selection_modes.items())),
        "num_selected": len(verifier_selected),
        "average_selected_verifier_score": safe_mean(selected_scores),
        "rows_with_available_correct_candidate": len(rows_with_available_correct),
        "selection_success_rate": (
            len(success_rows) / len(rows_with_available_correct) if rows_with_available_correct else 0.0
        ),
        "mixed_candidate_cases": len(mixed_rows),
        "correct_over_incorrect_cases": len(correct_over_incorrect_rows),
        "correct_over_incorrect_rate": len(correct_over_incorrect_rows) / len(mixed_rows) if mixed_rows else 0.0,
        "fixed_first_candidate_cases": len(fixed_first_rows),
        "missed_available_correct_cases": len(missed_available_rows),
        "consensus_fallback_to_verifier_cases": sum(
            1 for row in verifier_selected if row.get("selection_metadata", {}).get("fallback_to_verifier")
        ),
        "average_answer_groups": safe_mean(
            [
                float(row.get("selection_metadata", {}).get("num_answer_groups") or 0.0)
                for row in verifier_selected
                if row.get("selection_metadata", {}).get("selection") == "consensus_verifier"
            ]
        ),
    }


def truncation_summary(rows: list[dict], run_metadata: dict | None = None) -> dict:
    max_new_tokens = (run_metadata or {}).get("max_new_tokens")
    truncated = []
    if max_new_tokens is not None:
        threshold = int(max_new_tokens)
        truncated = [
            row
            for row in rows
            if int(row.get("token_counts", {}).get("solver_generated_tokens") or 0) >= threshold
        ]
    non_truncated = [row for row in rows if row not in truncated]
    return {
        "max_new_tokens": max_new_tokens,
        "num_truncated": len(truncated),
        "truncation_rate": len(truncated) / len(rows) if rows else 0.0,
        "accuracy_non_truncated": accuracy(non_truncated),
        "num_non_truncated": len(non_truncated),
    }


def summarize_rows(rows: list[dict], run_metadata: dict | None = None) -> dict:
    generated_tokens = [
        float(row.get("token_counts", {}).get("solver_generated_tokens") or 0.0)
        for row in rows
    ]
    runtimes = [float(row.get("runtime", {}).get("total_sec", row.get("runtime_sec", 0.0)) or 0.0) for row in rows]
    return {
        "schema_version": "benchmark_summary_v1",
        "benchmark_metadata": (run_metadata or {}).get("benchmark_metadata"),
        "num_examples": len(rows),
        "accuracy": accuracy(rows),
        "accuracy_by_benchmark": grouped_accuracy(rows, "benchmark"),
        "accuracy_by_type": grouped_accuracy(rows, "type"),
        "accuracy_by_category": grouped_accuracy(rows, "category"),
        "accuracy_by_level": grouped_accuracy(rows, "level"),
        "average_runtime_sec": safe_mean(runtimes),
        "average_generated_tokens": safe_mean(generated_tokens),
        "truncation": truncation_summary(rows, run_metadata=run_metadata),
        "verifier": verifier_summary(rows),
        "selection": selection_summary(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an evaluation JSONL file.")
    parser.add_argument("jsonl_path")
    args = parser.parse_args()

    path = resolve_input_path(args.jsonl_path)
    rows, run_metadata = load_jsonl(path)
    summary = summarize_rows(rows, run_metadata=run_metadata)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
