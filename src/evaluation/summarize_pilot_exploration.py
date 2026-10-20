"""Compare common-stop controls and sampled pass@k without using test benchmarks."""

import argparse
import json
from pathlib import Path

from src.core.config import ensure_scratch_path
from src.training.grpo.train import write_json
from src.evaluation.pilot_exploration import NAMES


def read_rows(root, mode, name, n, candidates):
    rows = []
    for path in sorted((root / mode / name).glob("shard_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows) != n or len({r["source_index"] for r in rows}) != n or any(len(r["candidates"]) != candidates for r in rows):
        raise ValueError(f"Incomplete or duplicate {mode}/{name}: {len(rows)}/{n}")
    return {r["source_index"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--comparison_root", type=Path, required=True)
    args = parser.parse_args()
    ensure_scratch_path(args.output_dir)
    ensure_scratch_path(args.comparison_root)
    greedy = {name: read_rows(args.output_dir, "greedy", name, 300, 1) for name in NAMES}
    sampled = {name: read_rows(args.output_dir, "sampled", name, 100, 8) for name in NAMES[:-1]}
    baseline = greedy["baseline"]
    control_summary = []
    for name, rows in greedy.items():
        if rows.keys() != baseline.keys() or any(r["prompt_hash"] != baseline[i]["prompt_hash"] for i, r in rows.items()):
            raise ValueError("Greedy prompts or IDs differ across models")
        original = {r["source_index"]: r for r in (json.loads(line) for line in (args.comparison_root / "evaluation" / f"{name}.jsonl").read_text().splitlines() if line.strip())}
        if original.keys() != rows.keys():
            raise ValueError("Original evaluation IDs differ")
        def correct(row):
            return row["candidates"][0]["correct"]
        candidates = [r["candidates"][0] for r in rows.values()]
        control_summary.append({"name": name, "n": 300, "accuracy": sum(correct(r) for r in rows.values()) / 300, "legacy_accuracy": sum(r["correct"] for r in original.values()) / 300, "fixed_by_stop": [i for i in rows if correct(rows[i]) and not original[i]["correct"]], "broken_by_stop": [i for i in rows if not correct(rows[i]) and original[i]["correct"]], "fixed_vs_common_baseline": [i for i in rows if correct(rows[i]) and not correct(baseline[i])], "broken_vs_common_baseline": [i for i in rows if not correct(rows[i]) and correct(baseline[i])], "truncated": sum(c["truncated"] for c in candidates), "avg_tokens": sum(c["generated_tokens"] for c in candidates) / 300, "avg_runtime": sum(r["runtime_seconds"] for r in rows.values()) / 300})
    exploration = []
    sample_baseline = sampled["baseline"]
    for name, rows in sampled.items():
        if rows.keys() != sample_baseline.keys() or any(r["prompt_hash"] != sample_baseline[i]["prompt_hash"] for i, r in rows.items()):
            raise ValueError("Sampled prompts or IDs differ across models")
        candidates = [c for r in rows.values() for c in r["candidates"]]
        exploration.append({"name": name, "n": 100, "pass_at_k": {str(k): sum(r["pass_at_k"][str(k)] for r in rows.values()) / 100 for k in [1, 4, 8]}, "all_wrong_rate": sum(r["num_correct"] == 0 for r in rows.values()) / 100, "all_correct_rate": sum(r["num_correct"] == 8 for r in rows.values()) / 100, "mixed_rate": sum(0 < r["num_correct"] < 8 for r in rows.values()) / 100, "avg_tokens": sum(c["generated_tokens"] for c in candidates) / 800, "truncation_rate": sum(c["truncated"] for c in candidates) / 800, "avg_group_runtime": sum(r["runtime_seconds"] for r in rows.values()) / 100, "gained_sampled_success": [i for i in rows if rows[i]["num_correct"] > 0 and sample_baseline[i]["num_correct"] == 0], "lost_sampled_success": [i for i in rows if rows[i]["num_correct"] == 0 and sample_baseline[i]["num_correct"] > 0]})
    payload = {"greedy_controls": control_summary, "exploration": exploration, "limitations": "Single training seed, fixed development data. pass@1 is sampled, separate from greedy accuracy. Zero successes among eight samples does not prove absent capability. Both EOS and chat-end stop accepted uniformly. Original results preserved."}
    write_json(args.output_dir / "summary.json", payload)
    lines = ["# Evaluation controls and RL exploration", "", payload["limitations"], "", "## Common-stop greedy controls (300 problems)", "", "| Model | Original accuracy | Common-stop accuracy | Stop fixes/breaks | Truncated | Avg tokens |", "|---|---|---|---|---|---|"]
    for r in control_summary:
        lines.append(f"| {r['name']} | {r['legacy_accuracy']:.1%} | {r['accuracy']:.1%} | {len(r['fixed_by_stop'])}/{len(r['broken_by_stop'])} | {r['truncated']} | {r['avg_tokens']:.1f} |")
    lines.extend(["", "## Eight-rollout exploration (same 100 problems)", "", "| Model | Sampled pass@1 | pass@4 | pass@8 | All wrong | Mixed | All correct | Avg tokens |", "|---|---|---|---|---|---|---|---|"])
    for r in exploration:
        p = r["pass_at_k"]
        lines.append(f"| {r['name']} | {p['1']:.1%} | {p['4']:.1%} | {p['8']:.1%} | {r['all_wrong_rate']:.1%} | {r['mixed_rate']:.1%} | {r['all_correct_rate']:.1%} | {r['avg_tokens']:.1f} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
