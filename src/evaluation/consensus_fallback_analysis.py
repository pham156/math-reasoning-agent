from __future__ import annotations

import argparse
import json
from statistics import mean

from src.agents.pipeline import resolve_output_path
from src.evaluation.summarize import load_jsonl, resolve_input_path


FALLBACKS = ("verifier", "first", "abstain")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def answer_groups(candidates: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.get("normalized_answer") or "", []).append(candidate)
    rows = [
        {
            "normalized_answer": answer,
            "display_answer": group[0].get("extracted_answer"),
            "count": len(group),
            "candidate_indices": [candidate["candidate_index"] for candidate in group],
            "max_verifier_score": max(float(candidate.get("verifier_score") or 0.0) for candidate in group),
            "any_candidate_correct": any(bool(candidate.get("exact_match")) for candidate in group),
        }
        for answer, group in groups.items()
    ]
    rows.sort(key=lambda group: (group["count"], group["max_verifier_score"], -min(group["candidate_indices"])), reverse=True)
    return rows


def select_with_fallback(candidates: list[dict], fallback: str) -> tuple[dict | None, dict]:
    groups = answer_groups(candidates)
    majority_threshold = len(candidates) // 2 + 1
    majority_groups = [group for group in groups if group["count"] >= majority_threshold]
    if majority_groups:
        selected_group = majority_groups[0]
        pool = [
            candidate
            for candidate in candidates
            if candidate["candidate_index"] in set(selected_group["candidate_indices"])
        ]
        selected = max(pool, key=lambda candidate: (float(candidate.get("verifier_score") or 0.0), -candidate["candidate_index"]))
        return selected, {
            "answer_groups": groups,
            "selected_group": selected_group,
            "fallback": fallback,
            "fallback_used": False,
            "abstained": False,
            "majority_threshold": majority_threshold,
        }

    if fallback == "abstain":
        return None, {
            "answer_groups": groups,
            "selected_group": None,
            "fallback": fallback,
            "fallback_used": True,
            "abstained": True,
            "majority_threshold": majority_threshold,
        }
    if fallback == "first":
        selected = candidates[0]
    elif fallback == "verifier":
        selected = max(candidates, key=lambda candidate: (float(candidate.get("verifier_score") or 0.0), -candidate["candidate_index"]))
    else:
        raise ValueError(f"Unknown fallback {fallback!r}")
    selected_group = next(group for group in groups if selected["candidate_index"] in group["candidate_indices"])
    return selected, {
        "answer_groups": groups,
        "selected_group": selected_group,
        "fallback": fallback,
        "fallback_used": True,
        "abstained": False,
        "majority_threshold": majority_threshold,
    }


def analyze(rows: list[dict], fallback: str) -> dict:
    selected_correct = []
    answered_correct = []
    majority_correct = []
    fallback_correct = []
    oracle_available = []
    missed_correct = 0
    abstained = 0
    majority_cases = 0
    fallback_cases = 0
    examples = []

    for row in rows:
        candidates = row.get("candidate_solutions", [])
        selected, metadata = select_with_fallback(candidates, fallback)
        any_correct = any(bool(candidate.get("exact_match")) for candidate in candidates)
        oracle_available.append(1.0 if any_correct else 0.0)
        if metadata["abstained"]:
            abstained += 1
            selected_correct.append(0.0)
            if any_correct:
                missed_correct += 1
        else:
            is_correct = bool(selected and selected.get("exact_match"))
            selected_correct.append(1.0 if is_correct else 0.0)
            answered_correct.append(1.0 if is_correct else 0.0)
            if any_correct and not is_correct:
                missed_correct += 1

        if metadata["fallback_used"]:
            fallback_cases += 1
            if not metadata["abstained"]:
                fallback_correct.append(1.0 if selected and selected.get("exact_match") else 0.0)
        else:
            majority_cases += 1
            majority_correct.append(1.0 if selected and selected.get("exact_match") else 0.0)

        examples.append(
            {
                "problem_id": row.get("problem_id"),
                "problem_index": row.get("problem_index"),
                "fallback": fallback,
                "selected_candidate_index": selected.get("candidate_index") if selected else None,
                "selected_answer": selected.get("extracted_answer") if selected else None,
                "selected_correct": bool(selected and selected.get("exact_match")),
                "oracle_candidate_available": any_correct,
                "fallback_used": metadata["fallback_used"],
                "abstained": metadata["abstained"],
                "answer_groups": metadata["answer_groups"],
                "selected_group": metadata["selected_group"],
            }
        )

    runtimes = [float(row.get("runtime", {}).get("total_sec", row.get("runtime_sec", 0.0)) or 0.0) for row in rows]
    generated_tokens = [float(row.get("token_counts", {}).get("solver_generated_tokens") or 0.0) for row in rows]
    input_tokens = [float(row.get("token_counts", {}).get("solver_input_tokens") or 0.0) for row in rows]
    return {
        "fallback": fallback,
        "n": len(rows),
        "selected_accuracy": safe_mean(selected_correct),
        "answered_accuracy": safe_mean(answered_correct),
        "abstain_count": abstained,
        "abstain_rate": abstained / len(rows) if rows else 0.0,
        "majority_cases": majority_cases,
        "majority_rate": majority_cases / len(rows) if rows else 0.0,
        "majority_accuracy": safe_mean(majority_correct),
        "fallback_cases": fallback_cases,
        "fallback_accuracy": safe_mean(fallback_correct),
        "oracle_candidate_accuracy": safe_mean(oracle_available),
        "missed_correct_cases": missed_correct,
        "avg_runtime": safe_mean(runtimes),
        "avg_generated_tokens": safe_mean(generated_tokens),
        "avg_input_tokens": safe_mean(input_tokens),
        "examples": examples,
    }


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "run",
        "fallback",
        "n",
        "selected_accuracy",
        "answered_accuracy",
        "abstain_rate",
        "majority_rate",
        "majority_accuracy",
        "fallback_accuracy",
        "oracle_candidate_accuracy",
        "missed_correct_cases",
        "avg_runtime",
        "avg_generated_tokens",
        "avg_input_tokens",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["fallback"],
                    str(row["n"]),
                    f"{row['selected_accuracy']:.4f}",
                    f"{row['answered_accuracy']:.4f}",
                    f"{row['abstain_rate']:.4f}",
                    f"{row['majority_rate']:.4f}",
                    f"{row['majority_accuracy']:.4f}",
                    f"{row['fallback_accuracy']:.4f}",
                    f"{row['oracle_candidate_accuracy']:.4f}",
                    str(row["missed_correct_cases"]),
                    f"{row['avg_runtime']:.2f}",
                    f"{row['avg_generated_tokens']:.1f}",
                    f"{row['avg_input_tokens']:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze consensus fallback strategies from best-of-N JSONLs.")
    parser.add_argument("--output_json", default="outputs/consensus_math500/fallback_analysis.json")
    parser.add_argument("--output_md", default="outputs/consensus_math500/fallback_analysis.md")
    parser.add_argument("jsonl_paths", nargs="+")
    args = parser.parse_args()

    summary_rows = []
    payload = {"schema_version": "consensus_fallback_analysis_v1", "runs": []}
    for input_arg in args.jsonl_paths:
        path = resolve_input_path(input_arg)
        rows, metadata = load_jsonl(path)
        run_name = path.stem
        run_payload = {"run": run_name, "path": str(path), "metadata": metadata, "fallbacks": []}
        for fallback in FALLBACKS:
            result = analyze(rows, fallback)
            result["run"] = run_name
            summary_rows.append({key: value for key, value in result.items() if key != "examples"})
            run_payload["fallbacks"].append(result)
        payload["runs"].append(run_payload)

    output_json = resolve_output_path(args.output_json)
    output_md = resolve_output_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = "# Consensus Fallback Analysis\n\n" + markdown_table(summary_rows) + "\n"
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
