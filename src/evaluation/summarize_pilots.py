"""Summarize completed or paused four-method pilots and paired changes."""

import argparse
import json
from pathlib import Path

from src.training.rl.methods import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    for method in METHODS:
        run = args.root / "pilot" / method
        progress = run / "progress.json"
        row = {"algorithm": method, "status": "missing"}
        if progress.exists():
            row.update(json.loads(progress.read_text()))
        result = run / "smoke_result.json"
        if row["status"] == "completed" and result.exists():
            metadata = json.loads(result.read_text())
            before = json.loads((run / "baseline.json").read_text())["examples"]
            after = json.loads((run / "after.json").read_text())["examples"]
            if [r["source_index"] for r in before] != [r["source_index"] for r in after]:
                raise ValueError("Paired evaluation source IDs differ")
            row.update({"n": len(after), "baseline_accuracy": sum(r["correct"] for r in before) / len(before), "accuracy": sum(r["correct"] for r in after) / len(after), "fixed": [a["source_index"] for b, a in zip(before, after, strict=True) if not b["correct"] and a["correct"]], "broken": [a["source_index"] for b, a in zip(before, after, strict=True) if b["correct"] and not a["correct"]], "avg_generated_tokens": sum(r["generated_tokens"] for r in after) / len(after), "truncated": sum(r["truncated"] for r in after), "history_summary": metadata["history_summary"]})
        rows.append(row)
    payload = {"runs": rows, "all_completed": all(r["status"] == "completed" for r in rows), "limitation": "Seed-0 pilots; 20 reserved MATH-train development diagnostics, not final benchmark results"}
    (args.root / "pilot_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Algorithm Pilots", "", payload["limitation"], "", "| Method | Status | Steps | Baseline | After | Fixed | Broken | Avg tokens |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        baseline = f"{row['baseline_accuracy']:.1%}" if "baseline_accuracy" in row else ""
        after = f"{row['accuracy']:.1%}" if "accuracy" in row else ""
        lines.append(f"| {row['algorithm']} | {row['status']} | {row.get('global_step', '')} | {baseline} | {after} | {len(row.get('fixed', []))} | {len(row.get('broken', []))} | {row.get('avg_generated_tokens', '')} |")
    (args.root / "pilot_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
