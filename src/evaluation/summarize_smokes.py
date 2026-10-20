"""Summarize smoke artifacts, including missing or failed array tasks."""

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
        result_file = args.root / method / "smoke_result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {"algorithm": method, "status": "missing"}
        rows.append(result)
    payload = {"runs": rows, "all_passed": all(row.get("status") == "passed" for row in rows), "purpose": "Small runtime/gradient/checkpoint checks, not a benchmark comparison"}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Algorithm Smoke Results", "", "| Method | Status | Steps | Parameter update observed | Runtime (s) |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['algorithm']} | {row['status']} | {row.get('global_step', '')} | {row.get('policy_update_observed', '')} | {row.get('runtime_seconds', '')} |")
        if row.get("error"):
            lines.append(f"\n{row['algorithm']}: {row['error']}\n")
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"all_passed": payload["all_passed"], "statuses": {row['algorithm']: row['status'] for row in rows}}))


if __name__ == "__main__":
    main()
