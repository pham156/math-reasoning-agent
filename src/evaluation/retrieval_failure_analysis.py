from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.agents.pipeline import resolve_output_path


def resolve_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_absolute():
        return requested
    return resolve_output_path(requested)


def load_eval_rows(path: Path) -> tuple[list[dict], dict | None]:
    rows = []
    metadata = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "run_metadata":
                metadata = item
            else:
                rows.append(item)
    return rows, metadata


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def retrieved_categories(row: dict) -> list[str]:
    return [str(item.get("type") or item.get("category") or "") for item in row.get("retrieved_examples_metadata", [])]


def retrieved_levels(row: dict) -> list[str]:
    return [str(item.get("level") or "") for item in row.get("retrieved_examples_metadata", [])]


def similarity_scores(row: dict) -> list[float]:
    scores = []
    for item in row.get("retrieved_examples_metadata", []):
        score = item.get("similarity_score", item.get("score", item.get("semantic_score")))
        if score is not None:
            scores.append(float(score))
    return scores


def top1_similarity(row: dict) -> float | None:
    scores = similarity_scores(row)
    return scores[0] if scores else None


def is_category_match(row: dict) -> bool:
    category = str(row.get("type") or row.get("category") or "")
    return bool(category) and category in retrieved_categories(row)


def is_level_match(row: dict) -> bool:
    level = str(row.get("level") or "")
    return bool(level) and level in retrieved_levels(row)


def row_record(row: dict, change_type: str) -> dict:
    retrieved = row.get("retrieved_examples_metadata", [])
    return {
        "change_type": change_type,
        "problem_id": row.get("problem_id"),
        "source_id": row.get("source_id"),
        "category": row.get("type") or row.get("category") or "",
        "level": row.get("level") or "",
        "problem": row.get("problem"),
        "ground_truth_answer": row.get("ground_truth_answer"),
        "retrieval_model_answer": row.get("model_answer"),
        "similarity_scores": similarity_scores(row),
        "top1_similarity": top1_similarity(row),
        "average_similarity": safe_mean(similarity_scores(row)),
        "retrieved_categories": retrieved_categories(row),
        "retrieved_levels": retrieved_levels(row),
        "category_match_any": is_category_match(row),
        "level_match_any": is_level_match(row),
        "retrieved_examples": [
            {
                "id": item.get("id"),
                "type": item.get("type"),
                "level": item.get("level"),
                "similarity_score": item.get("similarity_score"),
                "rerank_score": item.get("rerank_score"),
                "faiss_rank": item.get("faiss_rank"),
                "topics": item.get("topics"),
                "strategies": item.get("strategies"),
                "theorem_patterns": item.get("theorem_patterns"),
                "problem": item.get("problem"),
            }
            for item in retrieved
        ],
    }


def changed_rows(base_rows: list[dict], retrieval_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    base_by_id = {row["problem_id"]: row for row in base_rows}
    retrieval_by_id = {row["problem_id"]: row for row in retrieval_rows}
    fixed = []
    broken = []
    for problem_id, base in base_by_id.items():
        retrieval = retrieval_by_id[problem_id]
        if not base.get("exact_match") and retrieval.get("exact_match"):
            fixed.append(retrieval)
        elif base.get("exact_match") and not retrieval.get("exact_match"):
            broken.append(retrieval)
    return fixed, broken


def match_summary(rows: list[dict]) -> dict:
    top1_scores = [score for row in rows if (score := top1_similarity(row)) is not None]
    all_scores = [score for row in rows for score in similarity_scores(row)]
    return {
        "n": len(rows),
        "category_match_rate_any": safe_mean([1.0 if is_category_match(row) else 0.0 for row in rows]),
        "level_match_rate_any": safe_mean([1.0 if is_level_match(row) else 0.0 for row in rows]),
        "average_top1_similarity": safe_mean(top1_scores),
        "average_all_retrieved_similarity": safe_mean(all_scores),
    }


def grouped_helpfulness(base_rows: list[dict], retrieval_rows: list[dict], key: str) -> list[dict]:
    base_by_id = {row["problem_id"]: row for row in base_rows}
    retrieval_by_id = {row["problem_id"]: row for row in retrieval_rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in base_rows:
        groups[str(row.get(key) or "unknown")].append(row["problem_id"])

    records = []
    for group, problem_ids in sorted(groups.items()):
        fixed = sum(
            (not base_by_id[pid].get("exact_match")) and retrieval_by_id[pid].get("exact_match")
            for pid in problem_ids
        )
        broken = sum(
            base_by_id[pid].get("exact_match") and (not retrieval_by_id[pid].get("exact_match"))
            for pid in problem_ids
        )
        base_correct = sum(base_by_id[pid].get("exact_match") for pid in problem_ids)
        retrieval_correct = sum(retrieval_by_id[pid].get("exact_match") for pid in problem_ids)
        records.append(
            {
                key: group,
                "n": len(problem_ids),
                "base_correct": base_correct,
                "retrieval_correct": retrieval_correct,
                "base_accuracy": base_correct / len(problem_ids),
                "retrieval_accuracy": retrieval_correct / len(problem_ids),
                "fixed": fixed,
                "broken": broken,
                "net_help": fixed - broken,
                "net_help_rate": (fixed - broken) / len(problem_ids),
            }
        )
    return records


def category_confusion(rows: list[dict]) -> list[dict]:
    counts = Counter()
    for row in rows:
        category = str(row.get("type") or row.get("category") or "")
        top_category = retrieved_categories(row)[0] if retrieved_categories(row) else ""
        counts[(category, top_category)] += 1
    return [
        {"category": category, "top_retrieved_category": top_category, "count": count}
        for (category, top_category), count in counts.most_common()
    ]


def build_analysis(base_path: Path, retrieval_path: Path) -> dict:
    base_rows, base_metadata = load_eval_rows(base_path)
    retrieval_rows, retrieval_metadata = load_eval_rows(retrieval_path)
    fixed, broken = changed_rows(base_rows, retrieval_rows)
    unchanged = len(base_rows) - len(fixed) - len(broken)
    return {
        "schema_version": "retrieval_failure_analysis_v1",
        "base_path": str(base_path),
        "retrieval_path": str(retrieval_path),
        "base_metadata": base_metadata,
        "retrieval_metadata": retrieval_metadata,
        "overview": {
            "n": len(base_rows),
            "base_correct": sum(row.get("exact_match") for row in base_rows),
            "retrieval_correct": sum(row.get("exact_match") for row in retrieval_rows),
            "base_accuracy": sum(row.get("exact_match") for row in base_rows) / len(base_rows),
            "retrieval_accuracy": sum(row.get("exact_match") for row in retrieval_rows) / len(retrieval_rows),
            "fixed_by_retrieval": len(fixed),
            "broken_by_retrieval": len(broken),
            "unchanged": unchanged,
            "net_help": len(fixed) - len(broken),
        },
        "changed_summary": {
            "fixed": match_summary(fixed),
            "broken": match_summary(broken),
            "all_changed": match_summary(fixed + broken),
        },
        "helpfulness_by_category": grouped_helpfulness(base_rows, retrieval_rows, "type"),
        "helpfulness_by_level": grouped_helpfulness(base_rows, retrieval_rows, "level"),
        "top1_category_confusion_fixed": category_confusion(fixed),
        "top1_category_confusion_broken": category_confusion(broken),
        "fixed_examples": [row_record(row, "fixed") for row in fixed],
        "broken_examples": [row_record(row, "broken") for row in broken],
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(headers: list[str], rows: list[dict], keys: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def render_markdown(analysis: dict) -> str:
    overview = analysis["overview"]
    changed = analysis["changed_summary"]
    lines = [
        "# Retrieval Failure Analysis: MATH-500",
        "",
        "## Overview",
        "",
        table(
            ["metric", "value"],
            [{"metric": key, "value": value} for key, value in overview.items()],
            ["metric", "value"],
        ),
        "",
        "## Changed Examples Summary",
        "",
        table(
            ["group", "n", "category_match_rate_any", "level_match_rate_any", "average_top1_similarity", "average_all_retrieved_similarity"],
            [{"group": key, **value} for key, value in changed.items()],
            ["group", "n", "category_match_rate_any", "level_match_rate_any", "average_top1_similarity", "average_all_retrieved_similarity"],
        ),
        "",
        "## Retrieval Helpfulness By Category",
        "",
        table(
            ["category", "n", "base_acc", "retrieval_acc", "fixed", "broken", "net_help", "net_help_rate"],
            analysis["helpfulness_by_category"],
            ["type", "n", "base_accuracy", "retrieval_accuracy", "fixed", "broken", "net_help", "net_help_rate"],
        ),
        "",
        "## Retrieval Helpfulness By Level",
        "",
        table(
            ["level", "n", "base_acc", "retrieval_acc", "fixed", "broken", "net_help", "net_help_rate"],
            analysis["helpfulness_by_level"],
            ["level", "n", "base_accuracy", "retrieval_accuracy", "fixed", "broken", "net_help", "net_help_rate"],
        ),
        "",
        "## Top-1 Category Confusion: Fixed",
        "",
        table(
            ["category", "top_retrieved_category", "count"],
            analysis["top1_category_confusion_fixed"][:20],
            ["category", "top_retrieved_category", "count"],
        ),
        "",
        "## Top-1 Category Confusion: Broken",
        "",
        table(
            ["category", "top_retrieved_category", "count"],
            analysis["top1_category_confusion_broken"][:20],
            ["category", "top_retrieved_category", "count"],
        ),
        "",
        "## Interpretation",
        "",
        f"Retrieval fixed {overview['fixed_by_retrieval']} examples but broke {overview['broken_by_retrieval']}, for net help {overview['net_help']}.",
        "The tables above show where retrieval is useful versus harmful, and whether retrieved examples match the target problem category/level.",
        "",
        "## Changed Examples",
        "",
    ]
    for label, examples in (("Fixed By Retrieval", analysis["fixed_examples"]), ("Broken By Retrieval", analysis["broken_examples"])):
        lines.extend([f"### {label}", ""])
        for example in examples:
            lines.extend(
                [
                    f"- `{example['problem_id']}`: {example['category']} / {example['level']}",
                    f"  - top1_similarity: {fmt(example['top1_similarity'])}; category_match_any: {example['category_match_any']}; level_match_any: {example['level_match_any']}",
                    f"  - retrieved_categories: {', '.join(example['retrieved_categories'])}",
                    f"  - retrieved_levels: {', '.join(example['retrieved_levels'])}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MATH-500 retrieval fixed/broken examples.")
    parser.add_argument("--base", default="outputs/benchmarks_deterministic/math500_no_retrieval.jsonl")
    parser.add_argument("--retrieval", default="outputs/benchmarks_deterministic/math500_retrieval.jsonl")
    parser.add_argument("--json_output", default="outputs/retrieval_failure_analysis.json")
    parser.add_argument("--markdown_output", default="outputs/retrieval_failure_analysis.md")
    args = parser.parse_args()

    analysis = build_analysis(resolve_path(args.base), resolve_path(args.retrieval))
    json_path = resolve_output_path(args.json_output)
    markdown_path = resolve_output_path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
