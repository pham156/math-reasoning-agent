"""Re-grade saved evaluation JSONL with the historical lenient grader and the strict grader.

Reads pilot_dev rows (one response per line) and pilot_exploration shards (a `candidates`
list per line). Inputs are only read, so copies of the scratch outputs can be audited
off-cluster; no GPU is needed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.evaluation.answer import closed_boxed_answers
from src.training.rl.smoke import STRICT_GRADER, grade_solution, grade_solution_strict


COUNTS = ("samples", "stored_correct", "lenient_correct", "strict_correct", "lenient_only", "truncated", "truncated_lenient_correct", "truncated_strict_correct", "multiple_distinct_boxes", "stored_mismatch_hash_fix", "stored_mismatch_unexplained")


def responses(row):
    """(problem id, reference, response) for each response stored in one JSONL row."""
    reference = row["reference_answer"]
    for candidate in row.get("candidates", [row]):
        yield row["source_index"], reference, candidate


def grade_response(reference, response):
    text = response["raw_solution"]
    lenient = grade_solution(text, reference) == 1.0
    strict = grade_solution_strict(text, reference) == 1.0
    stored = bool(response["correct"])
    truncated = bool(response.get("truncated", False))
    mismatch = stored != lenient
    # Before the '####' fix, a Markdown heading anywhere could override a later \boxed answer.
    hash_fix = mismatch and "####" in text and bool(closed_boxed_answers(text))
    return {"samples": 1, "stored_correct": stored, "lenient_correct": lenient, "strict_correct": strict, "lenient_only": lenient and not strict, "truncated": truncated, "truncated_lenient_correct": truncated and lenient, "truncated_strict_correct": truncated and strict, "multiple_distinct_boxes": len(set(closed_boxed_answers(text))) > 1, "stored_mismatch_hash_fix": hash_fix, "stored_mismatch_unexplained": mismatch and not hash_fix}


def gradable(row):
    """True for pilot_dev / pilot_exploration rows; other JSONL in the tree is skipped."""
    if not isinstance(row, dict) or "reference_answer" not in row or "source_index" not in row:
        return False
    found = row["candidates"] if isinstance(row.get("candidates"), list) else [row]
    return bool(found) and all(isinstance(c, dict) and "raw_solution" in c and "correct" in c for c in found)


def group_key(root, path):
    """pilot_exploration shards group by their mode/name directory, other files by stem."""
    relative = path.relative_to(root)
    return str(relative.parent) if path.name.startswith("shard_") else str(relative.with_suffix(""))


def regrade(root):
    groups = defaultdict(lambda: {"counts": dict.fromkeys(COUNTS, 0), "per_problem": defaultdict(lambda: [0, 0, 0])})
    skipped = []
    for path in sorted(root.rglob("*.jsonl")):
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (json.JSONDecodeError, UnicodeDecodeError):
            rows = []
        if not rows or not all(gradable(r) for r in rows):
            skipped.append(str(path.relative_to(root)))
            continue
        group = groups[group_key(root, path)]
        for problem, reference, response in (item for row in rows for item in responses(row)):
            graded = grade_response(reference, response)
            for key in COUNTS:
                group["counts"][key] += int(graded[key])
            tally = group["per_problem"][problem]
            tally[0] += 1
            tally[1] += int(graded["lenient_correct"])
            tally[2] += int(graded["strict_correct"])
    results = {}
    for key, group in sorted(groups.items()):
        counts = group["counts"]
        n = counts["samples"]
        results[key] = {**counts, "lenient_accuracy": counts["lenient_correct"] / n, "strict_accuracy": counts["strict_correct"] / n, "stored_accuracy": counts["stored_correct"] / n}
    for key, result in results.items():
        baseline_key = str(Path(key).with_name("baseline"))
        if key == baseline_key or baseline_key not in groups:
            continue
        mine, base = groups[key]["per_problem"], groups[baseline_key]["per_problem"]
        if sorted(mine) != sorted(base) or any(mine[p][0] != base[p][0] for p in mine):
            result["vs_baseline"] = "skipped: problems or samples per problem differ from baseline"
            continue
        n = sum(t[0] for t in mine.values())
        result["vs_baseline"] = {"lenient_change": (sum(t[1] for t in mine.values()) - sum(t[1] for t in base.values())) / n, "strict_change": (sum(t[2] for t in mine.values()) - sum(t[2] for t in base.values())) / n}
    return {"strict_grader": STRICT_GRADER, "groups": results, "skipped_files": skipped}


def markdown(report):
    lines = [f"# Re-graded outputs (lenient vs {report['strict_grader']})", "", "Lenient is the historical reward/eval grader. Strict credits only a closed final \\boxed{} answer.", "", "| Run | Samples | Stored | Lenient | Strict | Lenient-only | Truncated (lenient / strict correct) | Multi-box | Stored≠lenient (#### fix / other) | Change vs base lenient → strict |", "|---|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for key, r in report["groups"].items():
        change = r.get("vs_baseline", "—")
        if isinstance(change, dict):
            change = f"{change['lenient_change']:+.1%} → {change['strict_change']:+.1%}"
        lines.append(f"| {key} | {r['samples']} | {r['stored_accuracy']:.1%} | {r['lenient_accuracy']:.1%} | {r['strict_accuracy']:.1%} | {r['lenient_only']} | {r['truncated']} ({r['truncated_lenient_correct']} / {r['truncated_strict_correct']}) | {r['multiple_distinct_boxes']} | {r['stored_mismatch_hash_fix']} / {r['stored_mismatch_unexplained']} | {change} |")
    if report["skipped_files"]:
        lines += ["", f"Skipped {len(report['skipped_files'])} JSONL files without graded responses."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory searched recursively for evaluation JSONL")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    root = Path(args.root)
    report = regrade(root)
    if not report["groups"]:
        parser.error(f"No graded evaluation JSONL under {root}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report), end="")


if __name__ == "__main__":
    main()
