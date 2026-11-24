"""Validate fixed-development evaluations at the extended GRPO training checkpoints."""

import argparse
from collections import Counter
import json
from pathlib import Path

from src.core.config import ensure_scratch_path
from src.evaluation.summarize_grpo_studies import compare_rows, evaluation_metrics
from src.evaluation.summarize_pilot_exploration import read_rows
from src.training.rl.checkpoint_storage import weights, write_receipt
from src.training.rl.extend import read, validate_extension_gate


def summarize(root, step):
    root = ensure_scratch_path(root)
    manifest = read(root / "submission.json")
    validate_extension_gate(root / "gate")
    run = root / "run"
    if read(run / "dataset_manifest.json") != read(Path(manifest["source"]) / "dataset_manifest.json"):
        raise ValueError("Continuation changed the frozen data split")
    progress = read(run / "progress.json")
    if progress["global_step"] != step or progress["status"] != ("paused" if step == 256 else "completed"):
        raise ValueError("Training has not reached the requested checkpoint")
    baseline = manifest["control"]
    greedy = read_rows(Path(baseline["greedy_root"]), "greedy", "grpo", 300, 1)
    sampled = read_rows(Path(baseline["sampled_root"]), "sampled", "grpo", 100, 8)
    rows = [{"step": 128, **evaluation_metrics(greedy, sampled), "fixed": [], "broken": []}]
    for milestone in (256, 512):
        if milestone > step:
            break
        directory = root / "evaluations" / f"step{milestone}"
        current = read_rows(directory, "greedy", "grpo", 300, 1)
        samples = read_rows(directory, "sampled", "grpo", 100, 8)
        changes = compare_rows(current, greedy)
        compare_rows(samples, sampled)
        model = run / ("checkpoints/checkpoint-256" if milestone == 256 else "final_model")
        for mode, count in (("greedy", 3), ("sampled", 5)):
            for shard in range(count):
                config = read(directory / mode / "grpo" / f"shard_{shard}_config.json")
                if (Path(config["model_path"]).resolve() != model.resolve() or
                        config["max_new_tokens"] != 3000 or config["temperature"] != (0 if mode == "greedy" else 1)):
                    raise ValueError("Changed checkpoint or evaluation protocol")
        rows.append({"step": milestone, **evaluation_metrics(current, samples), **changes})
    checkpoint = run / "checkpoints" / f"checkpoint-{step}"
    if not weights(checkpoint) or any(not (checkpoint / name).is_file() for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth")):
        raise ValueError("Missing latest resumable checkpoint")
    records = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    counts = Counter(r["problem"] for r in records)
    if len(records) != step * 16 or len(counts) != 512 or set(counts.values()) != {step * 16 // 512}:
        raise ValueError("Wrong repeated training exposure")
    history = read(run / "train_history.json")["log_history"]
    for row in rows:
        limit = row["step"]
        prefix = records[:limit * 16]
        lengths = [r["completions/mean_length"] for r in history if "completions/mean_length" in r and r["step"] <= limit]
        if len(lengths) != limit // 4:
            raise ValueError("Missing rollout-length accounting")
        row["training_correctness"] = sum(r.get("correctness", r["reward"]) for r in prefix) / len(prefix)
        row["training_completion_tokens"] = round(sum(lengths) * 64)
    payload = {"stage": step, "rows": rows, "source": manifest["source"],
               "training_history": history, "note": "Seed0 development-only continuation on the same512 problems; not new data or external benchmark results. Sampled pass@1 differs from greedy accuracy. Constant LR, optimizer/RNG continued; original control unchanged."}
    write_receipt(root / f"summary_{step}.json", payload)
    lines = ["# Longer GRPO Training", "", payload["note"], "", "| Updates | Accuracy | Fixed/broken vs128 | pass@1/4/8 | Avg tokens | Truncated | Training tokens |", "|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        pk = "/".join(f"{r['pass_at_k'][str(k)]:.1%}" for k in (1, 4, 8))
        lines.append(f"| {r['step']} | {r['accuracy']:.1%} | {len(r['fixed'])}/{len(r['broken'])} | {pk} | {r['avg_generated_tokens']:.1f} | {r['truncated']} | {r['training_completion_tokens']} |")
    (root / f"summary_{step}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--step", type=int, choices=[256, 512], required=True)
    args = parser.parse_args()
    summarize(args.root, args.step)


if __name__ == "__main__":
    main()
