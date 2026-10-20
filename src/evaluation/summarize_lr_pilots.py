"""Single-seed LR pilot comparison; development-only provisional choices."""

import argparse
import json
from pathlib import Path

from src.core.config import ensure_scratch_path
from src.evaluation.summarize_pilot_exploration import read_rows
from src.training.grpo.train import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    ensure_scratch_path(args.root)
    manifest = json.loads((args.root / "submission.json").read_text())
    controls = Path(manifest["reused_evaluation_root"])
    baseline = read_rows(controls, "greedy", "baseline", 300, 1)
    summaries = []
    for run in manifest["runs"]:
        name = run["algorithm"]
        rows = read_rows(Path(run["evaluation_root"]), "greedy", name, 300, 1)
        if rows.keys() != baseline.keys() or any(r["prompt_hash"] != baseline[i]["prompt_hash"] for i, r in rows.items()):
            raise ValueError("Evaluation split or prompts changed")
        run_root = Path(run["run_root"])
        result = json.loads((run_root / "smoke_result.json").read_text())
        saved = json.loads((run_root / "training_args.json").read_text())
        if result["global_step"] != 128 or saved["learning_rate"] != run["learning_rate"]:
            raise ValueError("Training budget or learning rate does not match")
        candidates = {i: r["candidates"][0] for i, r in rows.items()}
        summaries.append({"algorithm": name, "learning_rate": run["learning_rate"], "reused": run["reused"], "n": 300, "accuracy": sum(c["correct"] for c in candidates.values()) / 300, "truncated": sum(c["truncated"] for c in candidates.values()), "avg_generated_tokens": sum(c["generated_tokens"] for c in candidates.values()) / 300, "avg_runtime": sum(r["runtime_seconds"] for r in rows.values()) / 300, "fixed_vs_baseline": [i for i, c in candidates.items() if c["correct"] and not baseline[i]["candidates"][0]["correct"]], "broken_vs_baseline": [i for i, c in candidates.items() if not c["correct"] and baseline[i]["candidates"][0]["correct"]], "history_summary": result["history_summary"], "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"], "training_job_ids": run["training_jobs"], "evaluation_root": run["evaluation_root"]})
    choices = {name: sorted([r for r in summaries if r["algorithm"] == name], key=lambda r: (-r["accuracy"], r["avg_generated_tokens"], r["learning_rate"]))[0]["learning_rate"] for name in sorted({r["algorithm"] for r in summaries})}
    write_json(args.root / "summary.json", {"runs": summaries, "provisional_learning_rates": choices, "selection_rule": "Development accuracy, tie lower generated tokens, then lower LR. One seed; confirm promising settings with another seed before main study.", "cost_note": "Per-problem evaluation runtime and peak memory recorded. Training job IDs retained for full Slurm elapsed-cost accounting; final-segment train_runtime is not total training time."})
    lines = ["# Learning-rate pilots", "", "Same512 train IDs, seed0,128 updates, correctness-only reward; common-stop dev300. One-seed provisional selection only. Tie-breaks: lower generated tokens, then lower LR. Training costs require sum of Slurm segment elapsed times, not last-segment runtime.", "", "| Algorithm | LR | Reused | Accuracy | Truncated | Avg tokens | Reward mean | Mixed groups |", "|---|---|---|---|---|---|---|---|"]
    for r in summaries:
        h = r["history_summary"]
        lines.append(f"| {r['algorithm']} | {r['learning_rate']:.0e} | {r['reused']} | {r['accuracy']:.1%} | {r['truncated']} | {r['avg_generated_tokens']:.1f} | {h['reward_mean']:.3f} | {h['mixed_reward_group_rate']:.1%} |")
    lines.extend(["", "Provisional rates: " + json.dumps(choices)])
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
