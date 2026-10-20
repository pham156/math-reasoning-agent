from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.summarize import load_jsonl, resolve_input_path, safe_mean, summarize_rows


def mode_name(path: Path, metadata: dict | None) -> str:
    retrieval = (metadata or {}).get("retrieval_source", "none")
    style = (metadata or {}).get("retrieval_prompt_style", "none")
    num_candidates = int((metadata or {}).get("num_candidates") or 1)
    selection = (metadata or {}).get("selection", "first")
    if retrieval == "none" and num_candidates == 1:
        return "baseline"
    if retrieval == "none" and selection == "consensus_verifier":
        return f"no_retrieval_best_of_{num_candidates}_consensus_verifier"
    if retrieval == "none" and selection == "verifier":
        return f"no_retrieval_best_of_{num_candidates}_verifier"
    if style == "compact" and num_candidates == 1:
        return "compact_retrieval"
    if style == "compact" and selection == "verifier":
        return f"compact_retrieval_best_of_{num_candidates}_verifier"
    if style == "compact" and selection == "consensus_verifier":
        return f"compact_retrieval_best_of_{num_candidates}_consensus_verifier"
    return path.stem


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def run_row(path: Path) -> dict:
    rows, metadata = load_jsonl(path)
    summary = summarize_rows(rows, run_metadata=metadata)
    selection = summary.get("selection", {})
    verifier_scores = [
        float(candidate.get("verifier_score") or 0.0)
        for row in rows
        for candidate in row.get("candidate_solutions", [])
        if candidate.get("verifier_score") is not None
    ]
    return {
        "mode": mode_name(path, metadata),
        "n": len(rows),
        "accuracy": summary["accuracy"],
        "avg_verifier_score": safe_mean(verifier_scores) if verifier_scores else None,
        "selection_success_rate": selection.get("selection_success_rate"),
        "correct_over_incorrect_cases": selection.get("correct_over_incorrect_cases"),
        "fixed_first_candidate_cases": selection.get("fixed_first_candidate_cases"),
        "missed_available_correct_cases": selection.get("missed_available_correct_cases"),
        "consensus_fallback_to_verifier_cases": selection.get("consensus_fallback_to_verifier_cases"),
        "average_answer_groups": selection.get("average_answer_groups"),
        "avg_runtime": summary["average_runtime_sec"],
        "avg_generated_tokens": summary["average_generated_tokens"],
    }


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "mode",
        "n",
        "accuracy",
        "avg_verifier_score",
        "selection_success_rate",
        "correct_over_incorrect_cases",
        "fixed_first_candidate_cases",
        "missed_available_correct_cases",
        "consensus_fallback_to_verifier_cases",
        "average_answer_groups",
        "avg_runtime",
        "avg_generated_tokens",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    str(row["n"]),
                    format_float(row["accuracy"]),
                    format_float(row["avg_verifier_score"]),
                    format_float(row["selection_success_rate"]),
                    str(row["correct_over_incorrect_cases"] or ""),
                    str(row["fixed_first_candidate_cases"] or ""),
                    str(row["missed_available_correct_cases"] or ""),
                    str(row["consensus_fallback_to_verifier_cases"] or ""),
                    format_float(row["average_answer_groups"]),
                    format_float(row["avg_runtime"]),
                    format_float(row["avg_generated_tokens"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare verifier-guided best-of-N evaluation runs.")
    parser.add_argument("jsonl_paths", nargs="+")
    args = parser.parse_args()

    rows = [run_row(resolve_input_path(path)) for path in args.jsonl_paths]
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
