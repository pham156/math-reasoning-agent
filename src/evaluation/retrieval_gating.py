from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Callable

from src.agents.pipeline import resolve_output_path


HELPFUL_CATEGORIES = {"Intermediate Algebra", "Counting & Probability", "Algebra"}
SIMILARITY_THRESHOLDS = (0.70, 0.73, 0.76, 0.79)
CATEGORY_CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80)


def resolve_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_absolute():
        return requested
    return resolve_output_path(requested)


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
            else:
                rows.append(item)
    return rows, metadata


def load_prediction_rows(path: Path) -> tuple[dict[str, dict], dict | None]:
    rows, metadata = load_jsonl(path)
    return {row["problem_id"]: row for row in rows}, metadata


def write_jsonl(path: Path, metadata: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def top1_similarity(row: dict) -> float | None:
    examples = row.get("retrieved_examples_metadata") or []
    if not examples:
        return None
    score = examples[0].get("similarity_score", examples[0].get("score", examples[0].get("semantic_score")))
    return float(score) if score is not None else None


def predict_category_keyword(problem: str) -> tuple[str, float, list[str]]:
    text = problem.lower()
    scores = Counter()
    evidence: list[str] = []

    rules = {
        "Geometry": (
            "triangle",
            "circle",
            "angle",
            "radius",
            "diameter",
            "area",
            "perimeter",
            "parallel",
            "perpendicular",
            "quadrilateral",
            "polygon",
        ),
        "Counting & Probability": (
            "probability",
            "random",
            "ways",
            "arrange",
            "choose",
            "permutation",
            "combination",
            "committee",
            "color",
            "cards",
            "dice",
        ),
        "Number Theory": (
            "integer",
            "positive integer",
            "prime",
            "divisible",
            "remainder",
            "modulo",
            "congruent",
            "gcd",
            "lcm",
            "digits",
        ),
        "Precalculus": (
            "sin",
            "cos",
            "tan",
            "trigonometric",
            "log",
            "exponential",
            "polar",
            "complex plane",
            "sequence",
            "series",
            "function",
        ),
        "Intermediate Algebra": (
            "polynomial",
            "roots",
            "coefficient",
            "quadratic",
            "cubic",
            "rational function",
            "system of equations",
            "inequality",
            "absolute value",
        ),
        "Algebra": (
            "equation",
            "expression",
            "solve",
            "factor",
            "linear",
            "variable",
            "real numbers",
        ),
        "Prealgebra": (
            "percent",
            "ratio",
            "average",
            "mean",
            "fraction",
            "decimal",
            "units",
            "price",
        ),
    }
    for category, keywords in rules.items():
        for keyword in keywords:
            if keyword in text:
                scores[category] += 1
                evidence.append(keyword)

    if re.search(r"\b[a-z]\^2\b|\b[a-z]\^3\b", text):
        scores["Intermediate Algebra"] += 1
        evidence.append("power variable")
    if not scores:
        return "Algebra", 0.20, []

    ordered = scores.most_common()
    predicted, score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    confidence = min(0.95, 0.35 + 0.15 * score + 0.10 * max(score - runner_up, 0))
    return predicted, confidence, evidence[:8]


def base_metadata(policy: str, source_metadata: dict | None) -> dict:
    return {
        "kind": "run_metadata",
        "schema_version": "retrieval_gating_eval_v1",
        "benchmark": "math500",
        "policy": policy,
        "source_metadata": source_metadata,
    }


def selected_row(base_row: dict, retrieval_row: dict, use_retrieval: bool, policy: str, policy_metadata: dict) -> dict:
    row = dict(retrieval_row if use_retrieval else base_row)
    row["policy"] = policy
    row["retrieval_policy_metadata"] = {
        "policy": policy,
        "use_retrieval": use_retrieval,
        **policy_metadata,
    }
    if not use_retrieval:
        row["retrieved_examples_metadata"] = retrieval_row.get("retrieved_examples_metadata", [])
        row["retrieval_metadata"] = {
            **(row.get("retrieval_metadata") or {}),
            "actual_source": "none",
            "available_retrieval_source": "math",
        }
    return row


def build_policy_rows(
    policy: str,
    base_rows: list[dict],
    retrieval_rows: list[dict],
    chooser: Callable[[dict, dict], tuple[bool, dict]],
) -> list[dict]:
    retrieval_by_id = {row["problem_id"]: row for row in retrieval_rows}
    rows = []
    for base_row in base_rows:
        retrieval_row = retrieval_by_id[base_row["problem_id"]]
        use_retrieval, metadata = chooser(base_row, retrieval_row)
        rows.append(selected_row(base_row, retrieval_row, use_retrieval, policy, metadata))
    return rows


def choose_llm_category_gate(
    base_row: dict,
    predictions_by_id: dict[str, dict],
    threshold: float,
) -> tuple[bool, dict]:
    prediction = predictions_by_id.get(base_row["problem_id"], {})
    predicted_category = prediction.get("predicted_category")
    confidence = float(prediction.get("confidence") or 0.0)
    helpful_category = predicted_category in HELPFUL_CATEGORIES
    confident = confidence >= threshold
    use_retrieval = helpful_category and confident

    if not prediction:
        fallback_reason = "missing_prediction"
    elif not helpful_category:
        fallback_reason = "unhelpful_predicted_category"
    elif not confident:
        fallback_reason = "low_confidence"
    else:
        fallback_reason = ""

    return use_retrieval, {
        "predicted_category": predicted_category,
        "predicted_category_confidence": confidence,
        "category_confidence_threshold": threshold,
        "true_category": prediction.get("true_category", base_row.get("type")),
        "category_correct": prediction.get("category_correct"),
        "rationale": prediction.get("rationale"),
        "allowed_categories": sorted(HELPFUL_CATEGORIES),
        "classifier": "few_shot_llm",
        "fallback_reason": fallback_reason,
    }


def summarize_policy(policy: str, rows: list[dict], base_rows: list[dict]) -> dict:
    base_by_id = {row["problem_id"]: row for row in base_rows}
    fixes = sum((not base_by_id[row["problem_id"]].get("exact_match")) and row.get("exact_match") for row in rows)
    breaks = sum(base_by_id[row["problem_id"]].get("exact_match") and (not row.get("exact_match")) for row in rows)
    input_tokens = [float(row.get("token_counts", {}).get("solver_input_tokens") or 0.0) for row in rows]
    runtimes = [float(row.get("runtime", {}).get("total_sec", row.get("runtime_sec", 0.0)) or 0.0) for row in rows]
    return {
        "policy": policy,
        "n": len(rows),
        "accuracy": safe_mean([1.0 if row.get("exact_match") else 0.0 for row in rows]),
        "fixes_vs_baseline": fixes,
        "breaks_vs_baseline": breaks,
        "net": fixes - breaks,
        "avg_input_tokens": safe_mean(input_tokens),
        "avg_runtime": safe_mean(runtimes),
        "retrieval_use_rate": safe_mean(
            [1.0 if row.get("retrieval_policy_metadata", {}).get("use_retrieval") else 0.0 for row in rows]
        ),
    }


def table(rows: list[dict]) -> str:
    headers = [
        "policy",
        "accuracy",
        "fixes_vs_baseline",
        "breaks_vs_baseline",
        "net",
        "avg_input_tokens",
        "avg_runtime",
        "retrieval_use_rate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["policy"]),
                    f"{row['accuracy']:.4f}",
                    str(row["fixes_vs_baseline"]),
                    str(row["breaks_vs_baseline"]),
                    str(row["net"]),
                    f"{row['avg_input_tokens']:.1f}",
                    f"{row['avg_runtime']:.2f}",
                    f"{row['retrieval_use_rate']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective retrieval gating policies from MATH-500 runs.")
    parser.add_argument("--base", default="outputs/benchmarks_deterministic/math500_no_retrieval.jsonl")
    parser.add_argument("--always", default="outputs/benchmarks_deterministic/math500_retrieval.jsonl")
    parser.add_argument("--compact", default="outputs/retrieval_gating/compact_retrieval.jsonl")
    parser.add_argument(
        "--category_predictions",
        default="outputs/retrieval_gating/category_predictions_llm.jsonl",
        help="JSONL from python -m src.evaluation.category_classifier. Missing file skips LLM-predicted gates.",
    )
    parser.add_argument("--output_dir", default="outputs/retrieval_gating")
    args = parser.parse_args()

    base_rows, base_meta = load_jsonl(resolve_path(args.base))
    retrieval_rows, retrieval_meta = load_jsonl(resolve_path(args.always))
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policies: dict[str, list[dict]] = {
        "none": build_policy_rows(
            "none",
            base_rows,
            retrieval_rows,
            lambda base, retrieval: (False, {"reason": "baseline_no_retrieval"}),
        ),
        "always": build_policy_rows(
            "always",
            base_rows,
            retrieval_rows,
            lambda base, retrieval: (True, {"reason": "always_retrieve"}),
        ),
        "category_gate_oracle": build_policy_rows(
            "category_gate_oracle",
            base_rows,
            retrieval_rows,
            lambda base, retrieval: (
                str(base.get("type")) in HELPFUL_CATEGORIES,
                {"oracle_category": base.get("type"), "allowed_categories": sorted(HELPFUL_CATEGORIES)},
            ),
        ),
        "category_gate_keyword": build_policy_rows(
            "category_gate_keyword",
            base_rows,
            retrieval_rows,
            lambda base, retrieval: (
                (pred := predict_category_keyword(base.get("problem", "")))[0] in HELPFUL_CATEGORIES,
                {
                    "predicted_category": pred[0],
                    "predicted_category_confidence": pred[1],
                    "prediction_evidence": pred[2],
                    "allowed_categories": sorted(HELPFUL_CATEGORIES),
                    "classifier": "keyword_legacy",
                },
            ),
        ),
    }
    for threshold in SIMILARITY_THRESHOLDS:
        policy = f"similarity_gate_{threshold:.2f}"
        policies[policy] = build_policy_rows(
            policy,
            base_rows,
            retrieval_rows,
            lambda base, retrieval, threshold=threshold: (
                (top1_similarity(retrieval) or 0.0) >= threshold,
                {"threshold": threshold, "top1_similarity": top1_similarity(retrieval)},
            ),
        )

    prediction_metadata = None
    prediction_path = resolve_path(args.category_predictions)
    if prediction_path.exists():
        predictions_by_id, prediction_metadata = load_prediction_rows(prediction_path)
        for threshold in CATEGORY_CONFIDENCE_THRESHOLDS:
            policy = f"category_gate_predicted_{threshold:.2f}"
            policies[policy] = build_policy_rows(
                policy,
                base_rows,
                retrieval_rows,
                lambda base, retrieval, threshold=threshold: choose_llm_category_gate(
                    base,
                    predictions_by_id,
                    threshold,
                ),
            )

    compact_path = resolve_path(args.compact)
    if compact_path.exists():
        compact_rows, compact_meta = load_jsonl(compact_path)
        policies["compact_retrieval"] = [
            {**row, "policy": "compact_retrieval", "retrieval_policy_metadata": {"policy": "compact_retrieval", "use_retrieval": True}}
            for row in compact_rows
        ]
    else:
        compact_meta = None

    summary = []
    for policy, rows in policies.items():
        if policy == "compact_retrieval":
            source_meta = compact_meta
        elif policy.startswith("category_gate_predicted_"):
            source_meta = {"retrieval": retrieval_meta, "category_predictions": prediction_metadata}
        else:
            source_meta = retrieval_meta if policy != "none" else base_meta
        write_jsonl(output_dir / f"{policy}.jsonl", base_metadata(policy, source_meta), rows)
        summary.append(summarize_policy(policy, rows, base_rows))

    summary.sort(key=lambda row: (-row["accuracy"], row["policy"]))
    summary_payload = {
        "schema_version": "retrieval_gating_summary_v1",
        "baseline_path": str(resolve_path(args.base)),
        "always_path": str(resolve_path(args.always)),
        "compact_path": str(compact_path) if compact_path.exists() else None,
        "category_predictions_path": str(prediction_path) if prediction_path.exists() else None,
        "helpful_categories": sorted(HELPFUL_CATEGORIES),
        "similarity_thresholds": list(SIMILARITY_THRESHOLDS),
        "category_confidence_thresholds": list(CATEGORY_CONFIDENCE_THRESHOLDS),
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = "# Retrieval Gating Summary\n\n" + table(summary) + "\n"
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
