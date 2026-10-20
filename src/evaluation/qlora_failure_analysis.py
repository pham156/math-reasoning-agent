from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from src.evaluation.answer import normalize_answer
from src.evaluation.summarize import load_jsonl, resolve_input_path


def _index_rows(rows: list[dict]) -> dict[str, dict]:
    return {str(row["problem_id"]): row for row in rows}


def _tokens(row: dict) -> int:
    return int((row.get("token_counts") or {}).get("solver_generated_tokens") or 0)


def _answer(row: dict) -> str:
    return str(row.get("extracted_model_answer") or row.get("model_answer") or "")


def _excerpt(text: str, limit: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _max_tokens(metadata: dict | None) -> int:
    return int((metadata or {}).get("max_new_tokens") or 0)


def diagnose(row: dict, reference_answer: str, max_new_tokens: int, comparison_tokens: int) -> set[str]:
    raw = str(row.get("model_raw_solution") or "")
    predicted = _answer(row)
    generated_tokens = _tokens(row)
    labels: set[str] = set()
    has_box = "\\boxed" in raw or "\\fbox" in raw

    if not has_box:
        labels.add("no_boxed_answer")
    if max_new_tokens and generated_tokens >= max_new_tokens:
        labels.add("truncation")
    if generated_tokens < 64:
        labels.add("too_short")
    if generated_tokens >= 1024 or (comparison_tokens > 0 and generated_tokens > 1.75 * comparison_tokens and generated_tokens > 512):
        labels.add("too_verbose")

    normalized_reference = normalize_answer(reference_answer)
    normalized_prediction = normalize_answer(predicted)
    normalized_raw = normalize_answer(raw)
    possible_extraction_failure = (
        not predicted
        or (
            bool(normalized_reference)
            and normalized_prediction != normalized_reference
            and normalized_reference in normalized_raw
        )
    )
    if possible_extraction_failure:
        labels.add("answer_extraction_failure")

    if not row.get("exact_match") and "answer_extraction_failure" not in labels and "truncation" not in labels:
        labels.add("wrong_reasoning")
    return labels


def _stats(rows: list[dict]) -> dict:
    token_values = [_tokens(row) for row in rows]
    return {
        "n": len(rows),
        "correct": sum(bool(row.get("exact_match")) for row in rows),
        "accuracy": sum(bool(row.get("exact_match")) for row in rows) / len(rows) if rows else 0.0,
        "average_generated_tokens": sum(token_values) / len(token_values) if token_values else 0.0,
        "boxed_answer_rate": (
            sum("\\boxed" in str(row.get("model_raw_solution") or "") or "\\fbox" in str(row.get("model_raw_solution") or "") for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
    }


def _format_rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _case_table(title: str, cases: list[tuple[dict, dict]], limit: int = 8) -> list[str]:
    lines = [f"## {title}", "", f"Count: **{len(cases)}**", ""]
    if not cases:
        lines.append("No cases in this bucket.")
        lines.append("")
        return lines
    lines.extend(
        [
            "| ID | Type | Level | Ground truth | Instruct answer | QLoRA answer |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for instruct, qlora in cases[:limit]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(instruct.get("problem_id")),
                    str(instruct.get("type") or ""),
                    str(instruct.get("level") or ""),
                    _excerpt(str(instruct.get("extracted_ground_truth_answer") or ""), 100),
                    _excerpt(_answer(instruct), 100),
                    _excerpt(_answer(qlora), 100),
                ]
            )
            + " |"
        )
    lines.append("")
    for instruct, qlora in cases[:3]:
        lines.extend(
            [
                f"### {instruct.get('problem_id')}",
                "",
                f"**Problem:** {_excerpt(str(instruct.get('problem') or ''), 600)}",
                "",
                f"**Instruct:** {_excerpt(str(instruct.get('model_raw_solution') or ''), 500)}",
                "",
                f"**QLoRA:** {_excerpt(str(qlora.get('model_raw_solution') or ''), 500)}",
                "",
            ]
        )
    return lines


def build_report(
    instruct_rows: list[dict],
    instruct_meta: dict | None,
    qlora_rows: list[dict],
    qlora_meta: dict | None,
    base_rows: list[dict],
    base_meta: dict | None,
    limit: int,
) -> str:
    qlora_by_id = _index_rows(qlora_rows)
    base_by_id = _index_rows(base_rows)
    aligned_ids = [
        str(row["problem_id"])
        for row in instruct_rows
        if str(row["problem_id"]) in qlora_by_id and str(row["problem_id"]) in base_by_id
    ][:limit]
    instruct_by_id = _index_rows(instruct_rows)
    instruct = [instruct_by_id[item] for item in aligned_ids]
    qlora = [qlora_by_id[item] for item in aligned_ids]
    base = [base_by_id[item] for item in aligned_ids]
    if len(aligned_ids) < limit:
        raise ValueError(f"Only {len(aligned_ids)} aligned rows found; requested {limit}.")

    both_correct = [(i, q) for i, q in zip(instruct, qlora) if i.get("exact_match") and q.get("exact_match")]
    instruct_only = [(i, q) for i, q in zip(instruct, qlora) if i.get("exact_match") and not q.get("exact_match")]
    qlora_only = [(i, q) for i, q in zip(instruct, qlora) if not i.get("exact_match") and q.get("exact_match")]
    both_wrong = [(i, q) for i, q in zip(instruct, qlora) if not i.get("exact_match") and not q.get("exact_match")]

    failure_labels: Counter[str] = Counter()
    failure_examples: dict[str, list[str]] = {}
    qlora_max_tokens = _max_tokens(qlora_meta)
    for instruct_row, qlora_row in zip(instruct, qlora):
        if qlora_row.get("exact_match"):
            continue
        labels = diagnose(
            qlora_row,
            str(qlora_row.get("extracted_ground_truth_answer") or ""),
            qlora_max_tokens,
            _tokens(instruct_row),
        )
        failure_labels.update(labels)
        for label in labels:
            failure_examples.setdefault(label, []).append(str(qlora_row["problem_id"]))

    model_rows = [("Base", base), ("QLoRA", qlora), ("Instruct", instruct)]
    lines = [
        "# QLoRA Failure Analysis",
        "",
        "## Scope",
        "",
        f"This report compares the first **{limit} aligned MATH-500 examples** under deterministic decoding.",
        "Failure labels are non-exclusive heuristics; `wrong_reasoning` means the final answer is wrong after excluding likely extraction failures and truncation.",
        "",
        "## Prompt Audit",
        "",
        "- QLoRA training format: plain `### Problem:\n{problem}\n\n### Solution:\n{solution}`.",
        "- Base, QLoRA, and Instruct evaluation format: a user chat message containing a step-by-step instruction plus `Problem:` and `Solution:` fields.",
        "- Therefore the adapter was evaluated under a format it never saw during SFT. This is a genuine prompt-format mismatch and a plausible contributor, but it does not explain the full gap by itself because the unfine-tuned base control uses the identical evaluation prompt.",
        "",
        "## Three-Way Comparison",
        "",
        "| Model | Correct | Accuracy | Avg generated tokens | Boxed-answer rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in model_rows:
        stats = _stats(rows)
        lines.append(
            f"| {name} | {stats['correct']}/{stats['n']} | {_format_rate(stats['accuracy'])} | "
            f"{stats['average_generated_tokens']:.1f} | {_format_rate(stats['boxed_answer_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Instruct vs QLoRA Outcome Buckets",
            "",
            f"- Both correct: **{len(both_correct)}**",
            f"- Instruct correct / QLoRA wrong: **{len(instruct_only)}**",
            f"- QLoRA correct / Instruct wrong: **{len(qlora_only)}**",
            f"- Both wrong: **{len(both_wrong)}**",
            "",
            "## QLoRA Failure Diagnoses",
            "",
            f"Diagnosed QLoRA-wrong cases: **{sum(not row.get('exact_match') for row in qlora)}**",
            "",
            "| Diagnosis | Count | Example IDs |",
            "| --- | ---: | --- |",
        ]
    )
    for label in (
        "answer_extraction_failure",
        "no_boxed_answer",
        "too_verbose",
        "too_short",
        "wrong_reasoning",
        "truncation",
    ):
        examples = ", ".join(failure_examples.get(label, [])[:10])
        lines.append(f"| {label.replace('_', ' ')} | {failure_labels[label]} | {examples} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Missing `\\boxed{}` is a formatting symptom, not automatically a mathematical error; fallback answer extraction is still attempted.",
            "- `answer extraction failure` is conservative and marks empty extraction or cases where the normalized reference appears in the raw output but was not selected as the final answer.",
            "- `too short` is fewer than 64 generated tokens. `too verbose` is at least 1,024 tokens or substantially longer than the aligned Instruct response.",
            "- Prompt-format mismatch is global and is reported above rather than assigned to individual rows.",
            "",
        ]
    )
    lines.extend(_case_table("Both Correct", both_correct))
    lines.extend(_case_table("Instruct Correct / QLoRA Wrong", instruct_only))
    lines.extend(_case_table("QLoRA Correct / Instruct Wrong", qlora_only))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze QLoRA failures against base and Instruct controls.")
    parser.add_argument("--instruct", required=True)
    parser.add_argument("--qlora", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    instruct_rows, instruct_meta = load_jsonl(resolve_input_path(args.instruct))
    qlora_rows, qlora_meta = load_jsonl(resolve_input_path(args.qlora))
    base_rows, base_meta = load_jsonl(resolve_input_path(args.base))
    report = build_report(instruct_rows, instruct_meta, qlora_rows, qlora_meta, base_rows, base_meta, args.limit)

    output = Path(args.output).expanduser()
    if not output.is_absolute() and output.parts and output.parts[0] == "outputs":
        from src.core.config import output_dir

        output = output_dir(Path(*output.parts[1:]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote QLoRA failure analysis to {output}")


if __name__ == "__main__":
    main()
