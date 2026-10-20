from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.agents.pipeline import resolve_output_path
from src.evaluation.compare_small_qlora import load_rows, row_key
from src.evaluation.summarize import summarize_rows


def _boxed_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum("\\boxed" in str(row.get("model_raw_solution") or "") for row in rows) / len(rows)


def _read_curve_tail(path: Path, limit: int = 8) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:]


def _example_block(title: str, ids: list[str], baseline_by_id: dict[str, dict], rl_by_id: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", ""]
    if not ids:
        return lines + ["none", ""]
    for key in ids[:5]:
        base = baseline_by_id[key]
        rl = rl_by_id[key]
        lines.extend(
            [
                f"### {key}",
                "",
                f"- Type/level: {base.get('type')} / {base.get('level')}",
                f"- Ground truth: `{base.get('extracted_ground_truth_answer')}`",
                f"- Baseline answer: `{base.get('extracted_model_answer')}`",
                f"- RL answer: `{rl.get('extracted_model_answer')}`",
                "",
                "Baseline excerpt:",
                "",
                str(base.get("model_raw_solution") or "")[:700].replace("\n", " "),
                "",
                "RL excerpt:",
                "",
                str(rl.get("model_raw_solution") or "")[:700].replace("\n", " "),
                "",
            ]
        )
    return lines


def build_report(
    baseline_path: Path,
    rl_path: Path,
    train_dir: Path,
    output_json: Path | None = None,
    collapse_token_threshold: int = 64,
    substantial_drop: float = 0.05,
    min_rl_accuracy: float | None = None,
) -> str:
    baseline_rows, baseline_meta = load_rows(baseline_path)
    rl_rows, rl_meta = load_rows(rl_path)
    baseline_by_id = {row_key(row): row for row in baseline_rows}
    rl_by_id = {row_key(row): row for row in rl_rows}
    aligned_ids = [key for key in baseline_by_id if key in rl_by_id]
    if not aligned_ids:
        raise ValueError("No aligned problem IDs found between baseline and RLVR runs.")

    baseline_correct = sum(bool(baseline_by_id[key].get("exact_match")) for key in aligned_ids)
    rl_correct = sum(bool(rl_by_id[key].get("exact_match")) for key in aligned_ids)
    n = len(aligned_ids)
    fixed = [key for key in aligned_ids if not baseline_by_id[key].get("exact_match") and rl_by_id[key].get("exact_match")]
    broken = [key for key in aligned_ids if baseline_by_id[key].get("exact_match") and not rl_by_id[key].get("exact_match")]
    baseline_summary = summarize_rows(baseline_rows, run_metadata=baseline_meta)
    rl_summary = summarize_rows(rl_rows, run_metadata=rl_meta)

    baseline_acc = baseline_correct / n
    rl_acc = rl_correct / n
    avg_rl_tokens = float(rl_summary.get("average_generated_tokens") or 0.0)
    avg_base_tokens = float(baseline_summary.get("average_generated_tokens") or 0.0)
    boxed_rate = _boxed_rate(rl_rows)
    train_summary_path = train_dir / "train_summary.json"
    train_curve_path = train_dir / "train_curves.csv"
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8")) if train_summary_path.exists() else {}
    reward_mean = (
        train_summary.get("history_summary", {}).get("reward_mean")
        if isinstance(train_summary.get("history_summary"), dict)
        else None
    )

    stop_reasons = []
    if rl_acc + 1e-12 < baseline_acc:
        stop_reasons.append("evaluation_dropped")
    if baseline_acc - rl_acc >= substantial_drop:
        stop_reasons.append("evaluation_drops_substantially")
    if min_rl_accuracy is not None and rl_acc < min_rl_accuracy:
        stop_reasons.append(f"validation_below_{min_rl_accuracy:.0%}")
    if reward_mean == 0.0:
        stop_reasons.append("reward_collapsed")
    if avg_rl_tokens < collapse_token_threshold or (avg_base_tokens and avg_rl_tokens < 0.25 * avg_base_tokens):
        stop_reasons.append("response_length_collapsed")
    if boxed_rate < 0.5:
        stop_reasons.append("generations_malformed_low_boxed_rate")
    stop_before_scaling = bool(stop_reasons) or rl_acc < baseline_acc

    payload = {
        "schema_version": "rlvr_first50_report_v1",
        "n": n,
        "baseline_accuracy": baseline_acc,
        "rl_accuracy": rl_acc,
        "delta_correct": rl_correct - baseline_correct,
        "fixed_vs_baseline": fixed,
        "broken_vs_baseline": broken,
        "stop_before_scaling": stop_before_scaling,
        "stop_reasons": stop_reasons,
        "baseline_summary": baseline_summary,
        "rl_summary": rl_summary,
        "rl_boxed_rate": boxed_rate,
        "train_summary": train_summary,
        "train_curve_path": str(train_curve_path),
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RLVR/GRPO First-50 Report",
        "",
        f"Training directory: `{train_dir}`",
        f"Train summary: `{train_summary_path}`",
        f"Train curves: `{train_curve_path}`",
        "",
        "| Run | Correct | Accuracy | Avg runtime | Avg generated tokens | Boxed rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Instruct baseline | {baseline_correct}/{n} | {baseline_acc:.1%} | "
            f"{baseline_summary.get('average_runtime_sec', 0.0):.2f}s | "
            f"{baseline_summary.get('average_generated_tokens', 0.0):.1f} | {_boxed_rate(baseline_rows):.1%} |"
        ),
        (
            f"| RLVR adapter | {rl_correct}/{n} | {rl_acc:.1%} | "
            f"{rl_summary.get('average_runtime_sec', 0.0):.2f}s | "
            f"{rl_summary.get('average_generated_tokens', 0.0):.1f} | {boxed_rate:.1%} |"
        ),
        "",
        f"Delta correct: **{rl_correct - baseline_correct:+d}**",
        f"Fixed vs baseline: **{len(fixed)}**",
        f"Broken vs baseline: **{len(broken)}**",
        f"Stop before scaling: **{stop_before_scaling}**",
        f"Stop reasons: {', '.join(stop_reasons) if stop_reasons else 'none'}",
        "",
        "## Train Curve Tail",
        "",
    ]
    curve_tail = _read_curve_tail(train_curve_path)
    if curve_tail:
        keys = list(curve_tail[0])
        lines.extend(["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"])
        for row in curve_tail:
            lines.append("| " + " | ".join(row.get(key, "") for key in keys) + " |")
        lines.append("")
    else:
        lines.extend(["No train curve CSV found.", ""])
    lines.extend(_example_block("Fixed Examples", fixed, baseline_by_id, rl_by_id))
    lines.extend(_example_block("Broken Examples", broken, baseline_by_id, rl_by_id))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare first-50 RLVR validation against Instruct baseline.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--rl", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_rl_accuracy", type=float, default=None)
    parser.add_argument("--fail_if_stop", action="store_true")
    args = parser.parse_args()

    output_md = resolve_output_path(args.output_md)
    output_json = resolve_output_path(args.output_json)
    report = build_report(
        baseline_path=resolve_output_path(args.baseline),
        rl_path=resolve_output_path(args.rl),
        train_dir=Path(args.train_dir).expanduser().resolve(),
        output_json=output_json,
        min_rl_accuracy=args.min_rl_accuracy,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(report)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if args.fail_if_stop and payload["stop_before_scaling"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
