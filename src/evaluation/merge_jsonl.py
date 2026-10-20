from __future__ import annotations

import argparse
import json

from src.agents.pipeline import resolve_output_path
from src.evaluation.summarize import load_jsonl, resolve_input_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge evaluation JSONL shards into one JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows = []
    shard_metadata = []
    base_metadata = None
    for input_arg in args.inputs:
        input_path = resolve_input_path(input_arg)
        rows, metadata = load_jsonl(input_path)
        if base_metadata is None:
            base_metadata = dict(metadata or {})
        shard_metadata.append(
            {
                "path": str(input_path),
                "num_rows": len(rows),
                "metadata": metadata,
            }
        )
        merged_rows.extend(rows)

    merged_rows.sort(key=lambda row: (str(row.get("benchmark", "")), int(row.get("problem_index") or 0)))
    header = {
        **(base_metadata or {}),
        "kind": "run_metadata",
        "merged": True,
        "num_shards": len(shard_metadata),
        "shards": shard_metadata,
        "limit": len(merged_rows),
    }
    if "benchmark_metadata" in header and isinstance(header["benchmark_metadata"], dict):
        header["benchmark_metadata"] = {**header["benchmark_metadata"], "limit": len(merged_rows), "merged": True}

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged {len(merged_rows)} rows from {len(shard_metadata)} shards into {output_path}")


if __name__ == "__main__":
    main()
