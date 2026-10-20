from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.pipeline import resolve_output_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _variant_row(report_path: Path) -> dict:
    report = _read_json(report_path)
    train_summary = report.get("train_summary") or {}
    history = train_summary.get("history_summary") if isinstance(train_summary.get("history_summary"), dict) else {}
    dataset_filter = (
        train_summary.get("dataset_filter_metadata")
        if isinstance(train_summary.get("dataset_filter_metadata"), dict)
        else {}
    )
    run_dir = report_path.parent
    mixed_summary = _read_json(run_dir / "mixed_success_manifest.summary.json")
    row = {
        "variant": run_dir.name,
        "report_path": str(report_path),
        "n": report.get("n"),
        "baseline_accuracy": report.get("baseline_accuracy"),
        "rl_accuracy": report.get("rl_accuracy"),
        "delta_correct": report.get("delta_correct"),
        "fixes": len(report.get("fixed_vs_baseline") or []),
        "breaks": len(report.get("broken_vs_baseline") or []),
        "stop_before_scaling": report.get("stop_before_scaling"),
        "stop_reasons": report.get("stop_reasons") or [],
        "reward_mean": history.get("reward_mean"),
        "reward_std": history.get("reward_std"),
        "mixed_reward_group_rate": mixed_summary.get("mixed_reward_group_rate")
        if mixed_summary
        else history.get("mixed_reward_group_rate"),
        "kl_mean": history.get("kl_mean"),
        "response_length_mean": history.get("response_length_mean"),
        "train_runtime": (train_summary.get("train_metrics") or {}).get("train_runtime"),
        "dataset_filtered_count": dataset_filter.get("filtered_count"),
        "dataset_selected_count": dataset_filter.get("selected_count"),
    }
    return row


def _fmt(value: object, percent: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1%}" if percent else f"{value:.3f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GRPO stabilization study reports.")
    parser.add_argument("reports", nargs="+", help="Variant first50_report.json files or directories containing one.")
    parser.add_argument("--output_md", default="outputs/grpo_stabilization/summary.md")
    parser.add_argument("--output_json", default="outputs/grpo_stabilization/summary.json")
    args = parser.parse_args()

    report_paths = []
    for item in args.reports:
        path = resolve_output_path(item)
        if path.is_dir():
            path = path / "first50_report.json"
        report_paths.append(path)

    rows = [_variant_row(path) for path in sorted(report_paths)]
    payload = {"schema_version": "grpo_stabilization_summary_v1", "runs": rows}
    output_json = resolve_output_path(args.output_json)
    output_md = resolve_output_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    columns = [
        ("variant", "variant", False),
        ("n", "n", False),
        ("baseline", "baseline_accuracy", True),
        ("rl", "rl_accuracy", True),
        ("delta", "delta_correct", False),
        ("fixes", "fixes", False),
        ("breaks", "breaks", False),
        ("reward_mean", "reward_mean", False),
        ("reward_std", "reward_std", False),
        ("mixed_groups", "mixed_reward_group_rate", True),
        ("kl_mean", "kl_mean", False),
        ("resp_len", "response_length_mean", False),
        ("stop", "stop_before_scaling", False),
        ("reasons", "stop_reasons", False),
    ]
    lines = ["# GRPO Stabilization Study", ""]
    lines.append("| " + " | ".join(label for label, _key, _percent in columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key), percent) for _label, key, percent in columns) + " |")
    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
