"""Require complete paired development evaluations before comparing GRPO controls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from src.core.config import ensure_scratch_path
from src.evaluation.summarize_pilot_exploration import read_rows
from src.training.grpo.train import summarize_history, write_json
from src.training.rl.ablation_controls import check_saved, validate_gate


def metric_mean(history, key):
    values = [float(row[key]) for row in history if key in row]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Nonfinite {key}")
    return mean(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    manifest = json.loads((root / "submission.json").read_text())
    probe_hash = hashlib.sha256((root / "probes.json").read_bytes()).hexdigest()
    if manifest["probe_sha256"] != probe_hash:
        raise ValueError("Drift probes changed")
    rows_by_label = {run["label"]: read_rows(Path(run["evaluation_root"]), "greedy", "grpo", 300, 1) for run in manifest["runs"]}
    control = rows_by_label["seed0_lr1e-06"]
    summaries = []
    for run in manifest["runs"]:
        label, config = run["label"], run["settings"]
        rows = rows_by_label[label]
        if rows.keys() != control.keys() or any(row["prompt_hash"] != control[i]["prompt_hash"] for i, row in rows.items()):
            raise ValueError("Development inputs changed")
        path = Path(run["run_root"])
        result = json.loads((path / "smoke_result.json").read_text())
        training = json.loads((path / "training_args.json").read_text())
        check_saved(label, result["arguments"], training, config)
        if result["global_step"] != 128:
            raise ValueError("Training incomplete")
        split = json.loads((path / "dataset_manifest.json").read_text())
        control_split = json.loads((Path(manifest["runs"][0]["run_root"]) / "dataset_manifest.json").read_text())
        if split != control_split:
            raise ValueError("Frozen training/development manifest mismatch")
        if label.startswith(("kl_", "clip_")):
            validate_gate(root / "gates" / label, label)
        drift = json.loads((root / "diagnostics" / (label + ".json")).read_text())
        if drift["probe_sha256"] != probe_hash or len(drift["rows"]) != 8 or not math.isfinite(drift["mean_kl"]):
            raise ValueError("Invalid drift measurement")
        if Path(drift["model_path"]).resolve() != (path / "final_model").resolve():
            raise ValueError("Drift checkpoint mismatch")
        anchor = "seed1_lr1e-06" if label == "seed1_lr3e-06" else "seed0_lr1e-06"
        reference = rows_by_label[anchor]
        fixed = [i for i in rows if rows[i]["candidates"][0]["correct"] and not reference[i]["candidates"][0]["correct"]]
        broken = [i for i in rows if not rows[i]["candidates"][0]["correct"] and reference[i]["candidates"][0]["correct"]]
        history = json.loads((path / "train_history.json").read_text())["log_history"]
        history_summary = summarize_history(history)
        if config["kl_beta"] and history_summary["kl_mean"] is None:
            raise ValueError("KL penalty run has no training KL")
        summaries.append({"label": label, "settings": config, "reused": run["reused"], "accuracy": mean(int(r["candidates"][0]["correct"]) for r in rows.values()), "n": len(rows), "anchor": anchor, "fixed": fixed, "broken": broken,
                          "truncated": sum(r["candidates"][0]["truncated"] for r in rows.values()),
                          "avg_generated_tokens": mean(r["candidates"][0]["generated_tokens"] for r in rows.values()),
                          "avg_runtime": mean(r["runtime_seconds"] for r in rows.values()),
                          "history": history_summary, "clip_fraction_mean": metric_mean(history, "clip_ratio/region_mean"),
                          "clip_low_mean": metric_mean(history, "clip_ratio/low_mean"), "clip_high_mean": metric_mean(history, "clip_ratio/high_mean"),
                          "fixed_prefix_reference_kl": drift["mean_kl"], "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"], "training_jobs": run["training_jobs"]})
    pairs = []
    for seed in (0, 1):
        low = next(r for r in summaries if r["label"] == f"seed{seed}_lr1e-06")
        high = next(r for r in summaries if r["label"] == f"seed{seed}_lr3e-06")
        pairs.append({"seed": seed, "accuracy_lr1e6": low["accuracy"], "accuracy_lr3e6": high["accuracy"], "paired_delta": high["accuracy"] - low["accuracy"]})
    aggregates = {str(rate): {"mean_accuracy": mean(r["accuracy"] for r in summaries if r["settings"]["learning_rate"] == rate and r["label"].startswith("seed")), "sample_std_accuracy": stdev(r["accuracy"] for r in summaries if r["settings"]["learning_rate"] == rate and r["label"].startswith("seed"))} for rate in (1e-6, 3e-6)}
    write_json(root / "summary.json", {"runs": summaries, "lr_seed_pairs": pairs, "two_seed_lr_summary": aggregates, "note": "KL/clip are seed0 one-factor ablations against original LR1e-6; LR has paired seed0/1 comparisons. Training KL is TRL on-policy sampled estimator and missing for beta0, NOT zero. Diagnostic KL is exact vocabulary KL(policy||base) on shared fixed baseline prefixes, NOT on-policy or sequence KL. No external benchmarks for selection. Total GPU cost requires summing saved Slurm job elapsed times."})
    lines = ["# GRPO seed confirmation and one-factor controls", "", "All pilots: same512 train problems,128 updates,dev300. KL/clipping anchor is seed0 LR1e-6. Training KL is missing, not zero, for beta0. Fixed-prefix diagnostic KL is a different statistic.", "", "| Run | Accuracy | Fixes/breaks vs anchor | Truncated | Avg tokens | Reward | Train KL | Fixed-prefix KL | Clip fraction |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "N/A" if value is None else f"{value:.5g}"
    for row in summaries:
        h = row["history"]
        lines.append(f"| {row['label']} | {row['accuracy']:.1%} | {len(row['fixed'])}/{len(row['broken'])} | {row['truncated']} | {row['avg_generated_tokens']:.1f} | {h['reward_mean']:.3f} | {number(h['kl_mean'])} | {number(row['fixed_prefix_reference_kl'])} | {number(row['clip_fraction_mean'])} |")
    lines.extend(["", "## Matched-Seed Learning-Rate Comparison", "", "| Seed | LR1e-6 | LR3e-6 | Delta |", "|---|---:|---:|---:|"])
    for row in pairs:
        lines.append(f"| {row['seed']} | {row['accuracy_lr1e6']:.1%} | {row['accuracy_lr3e6']:.1%} | {100 * row['paired_delta']:+.1f} pp |")
    lines.extend(["", "Two-seed mean/sample std: " + json.dumps(aggregates), "", "Two seeds do not establish a definitive optimum. Near-zero clip fraction means a clipping-bound change may not materially affect the actual updates. Detailed entropy, reward/group variance, length, runtime, peak memory, and fixed/broken IDs are saved in summary.json; raw per-update curves remain in each run's train_history.json."])
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
