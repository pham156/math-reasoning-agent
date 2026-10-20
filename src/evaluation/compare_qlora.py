from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.summarize import load_jsonl, resolve_input_path, safe_mean, summarize_rows


def model_name(metadata: dict | None) -> str:
    adapter_path = (metadata or {}).get("adapter_path")
    if adapter_path:
        return "Qwen2.5-Math-7B+QLoRA"
    model_metadata = (metadata or {}).get("model_metadata") or {}
    path = str(model_metadata.get("effective_model_path") or model_metadata.get("model_path") or "")
    if "Instruct" in path:
        return "Qwen2.5-Math-7B-Instruct"
    if "Qwen2.5-Math-7B" in path:
        return "Qwen2.5-Math-7B"
    return Path(path).name or "unknown"


def mode_name(metadata: dict | None) -> str:
    num_candidates = int((metadata or {}).get("num_candidates") or 1)
    selection = (metadata or {}).get("selection", "first")
    if num_candidates == 1:
        return "deterministic"
    return f"best_of_{num_candidates}_{selection}"


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def row_for_path(path: Path) -> dict:
    rows, metadata = load_jsonl(path)
    summary = summarize_rows(rows, run_metadata=metadata)
    token_counts = [row.get("token_counts", {}) for row in rows]
    input_tokens = [float(item.get("solver_input_tokens") or 0.0) for item in token_counts]
    generated_tokens = [float(item.get("solver_generated_tokens") or 0.0) for item in token_counts]
    return {
        "path": str(path),
        "benchmark": (metadata or {}).get("benchmark"),
        "model": model_name(metadata),
        "mode": mode_name(metadata),
        "n": len(rows),
        "accuracy": summary["accuracy"],
        "avg_runtime": summary["average_runtime_sec"],
        "avg_input_tokens": safe_mean(input_tokens),
        "avg_generated_tokens": safe_mean(generated_tokens),
    }


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "benchmark",
        "model",
        "mode",
        "n",
        "accuracy",
        "avg_runtime",
        "avg_input_tokens",
        "avg_generated_tokens",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    str(row["model"]),
                    str(row["mode"]),
                    str(row["n"]),
                    format_float(row["accuracy"]),
                    format_float(row["avg_runtime"]),
                    format_float(row["avg_input_tokens"]),
                    format_float(row["avg_generated_tokens"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def resolve_output_path(path: str | None) -> Path | None:
    if not path:
        return None
    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested
    parts = requested.parts
    if parts and parts[0] == "outputs":
        from src.core.config import output_dir

        return output_dir(Path(*parts[1:]))
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare QLoRA and baseline evaluation JSONLs.")
    parser.add_argument("jsonl_paths", nargs="+")
    parser.add_argument("--output_markdown", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    rows = [row_for_path(resolve_input_path(path)) for path in args.jsonl_paths]
    rows.sort(key=lambda row: (str(row["benchmark"]), str(row["model"]), str(row["mode"])))
    markdown = markdown_table(rows)
    markdown_path = resolve_output_path(args.output_markdown)
    json_path = resolve_output_path(args.output_json)
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown + "\n", encoding="utf-8")
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"schema_version": "qlora_comparison_v1", "runs": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(markdown)


if __name__ == "__main__":
    main()
