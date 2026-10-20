"""Compare completed development evaluations against the frozen baseline."""

import argparse
import json
from pathlib import Path

from src.training.grpo.train import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    names = ["baseline", "grpo", "dr_grpo", "dapo_style", "gspo", "sft"]
    runs = {}
    for name in names:
        rows = [json.loads(line) for line in (args.root / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 300 or len({r["source_index"] for r in rows}) != 300:
            raise ValueError(f"Incomplete/duplicate evaluation: {name}")
        runs[name] = {r["source_index"]: r for r in rows}
    baseline = runs["baseline"]
    summaries = []
    for name, rows in runs.items():
        if rows.keys() != baseline.keys():
            raise ValueError("Mismatched development examples")
        n = len(rows)
        summaries.append({"name": name, "n": n, "accuracy": sum(r["correct"] for r in rows.values()) / n, "fixed": [i for i in rows if rows[i]["correct"] and not baseline[i]["correct"]], "broken": [i for i in rows if baseline[i]["correct"] and not rows[i]["correct"]], "avg_tokens": sum(r["generated_tokens"] for r in rows.values()) / n, "avg_runtime": sum(r["runtime_seconds"] for r in rows.values()) / n, "truncated": sum(r["truncated"] for r in rows.values())})
    write_json(args.root / "comparison.json", {"runs": summaries, "limitation": "Single-seed reserved MATH-train development results, not test benchmark results"})
    lines = ["# Shared 300-problem development comparison", "", "Single seed; reserved MATH-train development data, not benchmark test results. SFT and RL share training IDs and prompts, but differ in objective and compute.", "", "| Method | Accuracy | Fixed | Broken | Avg tokens | Avg runtime (s) | Truncated |", "|---|---|---|---|---|---|---|"]
    for row in summaries:
        lines.append(f"| {row['name']} | {row['accuracy']:.1%} | {len(row['fixed'])} | {len(row['broken'])} | {row['avg_tokens']:.1f} | {row['avg_runtime']:.2f} | {row['truncated']} |")
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
