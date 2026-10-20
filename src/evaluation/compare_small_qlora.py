from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.pipeline import resolve_output_path
from src.evaluation.summarize import summarize_rows


def load_rows(path: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    metadata: dict = {}
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


def row_key(row: dict) -> str:
    return str(row.get("problem_id") or row.get("index") or "")


def build_report(baseline_path: Path, adapter_path: Path, output_json: Path | None = None) -> str:
    baseline_rows, baseline_meta = load_rows(baseline_path)
    adapter_rows, adapter_meta = load_rows(adapter_path)
    baseline_by_id = {row_key(row): row for row in baseline_rows}
    adapter_by_id = {row_key(row): row for row in adapter_rows}
    aligned_ids = [key for key in baseline_by_id if key in adapter_by_id]
    if not aligned_ids:
        raise ValueError("No aligned problem IDs found between baseline and adapter runs.")

    baseline_correct = sum(bool(baseline_by_id[key].get("exact_match")) for key in aligned_ids)
    adapter_correct = sum(bool(adapter_by_id[key].get("exact_match")) for key in aligned_ids)
    n = len(aligned_ids)
    fixed = [
        key
        for key in aligned_ids
        if not baseline_by_id[key].get("exact_match") and adapter_by_id[key].get("exact_match")
    ]
    broken = [
        key
        for key in aligned_ids
        if baseline_by_id[key].get("exact_match") and not adapter_by_id[key].get("exact_match")
    ]
    both_correct = [
        key
        for key in aligned_ids
        if baseline_by_id[key].get("exact_match") and adapter_by_id[key].get("exact_match")
    ]
    both_wrong = [
        key
        for key in aligned_ids
        if not baseline_by_id[key].get("exact_match") and not adapter_by_id[key].get("exact_match")
    ]

    baseline_summary = summarize_rows(baseline_rows, run_metadata=baseline_meta)
    adapter_summary = summarize_rows(adapter_rows, run_metadata=adapter_meta)
    status = "improved" if adapter_correct > baseline_correct else "worse_or_equal"
    should_stop = adapter_correct <= baseline_correct

    payload = {
        "schema_version": "small_qlora_validation_v1",
        "n": n,
        "status": status,
        "stop_before_scaling": should_stop,
        "baseline": {
            "path": str(baseline_path),
            "correct": baseline_correct,
            "accuracy": baseline_correct / n,
            "summary": baseline_summary,
            "metadata": baseline_meta,
        },
        "adapter": {
            "path": str(adapter_path),
            "correct": adapter_correct,
            "accuracy": adapter_correct / n,
            "summary": adapter_summary,
            "metadata": adapter_meta,
        },
        "delta_correct": adapter_correct - baseline_correct,
        "fixed_vs_baseline": fixed,
        "broken_vs_baseline": broken,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Small Prompt-Matched QLoRA Validation",
        "",
        f"Aligned examples: **{n}**",
        "",
        "| Run | Correct | Accuracy | Avg runtime | Avg generated tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Instruct baseline | {baseline_correct}/{n} | {baseline_correct / n:.1%} | "
            f"{baseline_summary.get('average_runtime_sec', 0.0):.2f}s | "
            f"{baseline_summary.get('average_generated_tokens', 0.0):.1f} |"
        ),
        (
            f"| Instruct + QLoRA | {adapter_correct}/{n} | {adapter_correct / n:.1%} | "
            f"{adapter_summary.get('average_runtime_sec', 0.0):.2f}s | "
            f"{adapter_summary.get('average_generated_tokens', 0.0):.1f} |"
        ),
        "",
        f"Delta correct: **{adapter_correct - baseline_correct:+d}**",
        f"Fixed vs baseline: **{len(fixed)}**",
        f"Broken vs baseline: **{len(broken)}**",
        "",
        (
            "**Decision:** adapter improved on first-50; scaling can be considered."
            if adapter_correct > baseline_correct
            else "**Decision:** stop before AIME/full MATH-500 because the adapter is not better on first-50."
        ),
        "",
        "## Changed Cases",
        "",
        f"Fixed: {', '.join(fixed) if fixed else 'none'}",
        "",
        f"Broken: {', '.join(broken) if broken else 'none'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare small Instruct+QLoRA validation against Instruct baseline.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--fail_if_not_improved",
        action="store_true",
        help="Exit with status 3 when adapter accuracy is not strictly better than baseline.",
    )
    args = parser.parse_args()

    baseline_path = resolve_output_path(args.baseline)
    adapter_path = resolve_output_path(args.adapter)
    output_md = resolve_output_path(args.output_md)
    output_json = resolve_output_path(args.output_json)
    report = build_report(baseline_path, adapter_path, output_json=output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(report)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if args.fail_if_not_improved and payload["stop_before_scaling"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
