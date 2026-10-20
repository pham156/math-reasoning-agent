from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from src.core.config import output_dir
from src.evaluation.summarize import load_jsonl


def resolve_patterns(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        requested = Path(pattern).expanduser()
        candidates: list[str]
        if any(char in pattern for char in "*?[]"):
            if requested.is_absolute():
                candidates = glob.glob(str(requested))
            elif requested.parts and requested.parts[0] == "outputs":
                scratch_pattern = output_dir(Path(*requested.parts[1:]))
                candidates = glob.glob(str(scratch_pattern))
            else:
                candidates = glob.glob(pattern)
        else:
            if requested.is_absolute() or requested.exists():
                candidates = [str(requested)]
            elif requested.parts and requested.parts[0] == "outputs":
                candidates = [str(output_dir(Path(*requested.parts[1:])))]
            else:
                candidates = [str(requested)]
        paths.extend(Path(candidate) for candidate in candidates)
    return sorted(set(paths))


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def normalize_mode(row: dict, metadata: dict | None) -> str:
    retrieval_source = (metadata or {}).get("retrieval_source") or row.get("retrieval_metadata", {}).get("requested_source")
    use_verifier = (metadata or {}).get("use_verifier")
    if use_verifier is None:
        use_verifier = row.get("verifier_verdict") is not None
    if retrieval_source == "math" and use_verifier:
        return "retrieval_verifier"
    if retrieval_source == "math":
        return "retrieval"
    return "no_retrieval"


def summarize_file(path: Path) -> dict:
    rows, metadata = load_jsonl(path)
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    benchmark = rows[0].get("benchmark") or (metadata or {}).get("benchmark") or path.stem
    mode = normalize_mode(rows[0], metadata)
    token_counts = [row.get("token_counts", {}) for row in rows]
    runtimes = [float(row.get("runtime", {}).get("total_sec", row.get("runtime_sec", 0.0)) or 0.0) for row in rows]
    generated_tokens = [float(tokens.get("solver_generated_tokens") or 0.0) for tokens in token_counts]
    input_tokens = [float(tokens.get("solver_input_tokens") or 0.0) for tokens in token_counts]
    verifier_rows = [row for row in rows if row.get("verifier_verdict") is not None]
    verifier_valid_rate = None
    if verifier_rows:
        verifier_valid_rate = safe_mean([1.0 if row.get("verifier_verdict") == "valid" else 0.0 for row in verifier_rows])
    max_new_tokens = (metadata or {}).get("max_new_tokens")
    if max_new_tokens is None:
        truncated_rows = []
    else:
        threshold = int(max_new_tokens)
        truncated_rows = [
            row
            for row in rows
            if int(row.get("token_counts", {}).get("solver_generated_tokens") or 0) >= threshold
        ]
    non_truncated_rows = [row for row in rows if row not in truncated_rows]
    return {
        "path": str(path),
        "benchmark": benchmark,
        "mode": mode,
        "n": len(rows),
        "accuracy": safe_mean([1.0 if row.get("exact_match") else 0.0 for row in rows]),
        "num_truncated": len(truncated_rows),
        "truncation_rate": len(truncated_rows) / len(rows) if rows else 0.0,
        "accuracy_non_truncated": safe_mean(
            [1.0 if row.get("exact_match") else 0.0 for row in non_truncated_rows]
        ),
        "avg_runtime": safe_mean(runtimes),
        "avg_generated_tokens": safe_mean(generated_tokens),
        "avg_input_tokens": safe_mean(input_tokens),
        "verifier_valid_rate": verifier_valid_rate,
        "rows": rows,
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def print_table(rows: list[dict]) -> None:
    headers = [
        "benchmark",
        "mode",
        "n",
        "accuracy",
        "num_truncated",
        "truncation_rate",
        "accuracy_non_truncated",
        "avg_runtime",
        "avg_generated_tokens",
        "avg_input_tokens",
        "verifier_valid_rate",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    str(row["mode"]),
                    str(row["n"]),
                    format_float(row["accuracy"]),
                    str(row["num_truncated"]),
                    format_float(row["truncation_rate"]),
                    format_float(row["accuracy_non_truncated"]),
                    format_float(row["avg_runtime"]),
                    format_float(row["avg_generated_tokens"]),
                    format_float(row["avg_input_tokens"]),
                    format_float(row["verifier_valid_rate"]),
                ]
            )
            + " |"
        )


def grouped_accuracy(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return [
        {
            key: group,
            "n": len(group_rows),
            "accuracy": safe_mean([1.0 if row.get("exact_match") else 0.0 for row in group_rows]),
        }
        for group, group_rows in sorted(grouped.items())
    ]


def print_math500_breakdowns(summaries: list[dict]) -> None:
    math500 = [summary for summary in summaries if summary["benchmark"] == "math500"]
    if not math500:
        return
    for summary in math500:
        print(f"\nMATH-500 by type: {summary['mode']}")
        print("| type | n | accuracy |")
        print("| --- | --- | --- |")
        for row in grouped_accuracy(summary["rows"], "type"):
            print(f"| {row['type']} | {row['n']} | {format_float(row['accuracy'])} |")

        print(f"\nMATH-500 by level: {summary['mode']}")
        print("| level | n | accuracy |")
        print("| --- | --- | --- |")
        for row in grouped_accuracy(summary["rows"], "level"):
            print(f"| {row['level']} | {row['n']} | {format_float(row['accuracy'])} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple benchmark JSONL evaluation runs.")
    parser.add_argument("jsonl_paths", nargs="+", help="JSONL files or glob patterns, e.g. outputs/benchmarks/*.jsonl")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of tables.")
    args = parser.parse_args()

    paths = resolve_patterns(args.jsonl_paths)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {args.jsonl_paths}")

    summaries = [summarize_file(path) for path in paths]
    summaries.sort(key=lambda item: (item["benchmark"], item["mode"], item["path"]))

    if args.json:
        serializable = [{key: value for key, value in summary.items() if key != "rows"} for summary in summaries]
        print(json.dumps(serializable, indent=2, ensure_ascii=False))
        return

    print_table(summaries)
    print_math500_breakdowns(summaries)


if __name__ == "__main__":
    main()
