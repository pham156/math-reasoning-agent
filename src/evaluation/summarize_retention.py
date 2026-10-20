"""Paired retention deltas against the untouched base; not a tuning score."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from src.core.config import data_dir, ensure_scratch_path
from src.evaluation.retention import NAMES
from src.training.grpo.train import write_json


def paired_interval(deltas):
    rng = random.Random(42)
    estimates = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    return [estimates[49], estimates[1949]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    ensure_scratch_path(args.root)
    frozen_path = data_dir("rl_retention_v1/frozen.json")
    frozen = json.loads(frozen_path.read_text())
    expected = [r["id"] for r in frozen["items"]]
    runs = {}
    for name in NAMES:
        rows = [json.loads(line) for line in (args.root / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        if [r["id"] for r in rows] != expected:
            raise ValueError(f"Incomplete or changed retention set for {name}")
        runs[name] = {r["id"]: r for r in rows}
    baseline = runs["baseline"]
    results = []
    for name, rows in runs.items():
        wiki = [r for r in rows.values() if r["task"] == "wikitext"]
        nll = sum(r["nll_sum"] for r in wiki) / sum(r["scored_tokens"] for r in wiki)
        base_nll = sum(baseline[r["id"]]["nll_sum"] for r in wiki) / sum(r["scored_tokens"] for r in wiki)
        tasks = []
        for task in ("boolq", "arc_easy"):
            for style in ("raw", "chat"):
                selected = [r for r in rows.values() if r["task"] == task and r["format"] == style]
                deltas = [int(r["correct"]) - int(baseline[r["id"]]["correct"]) for r in selected]
                tasks.append({"task": task, "format": style, "n": len(selected), "accuracy": sum(r["correct"] for r in selected) / len(selected), "delta_accuracy": sum(deltas) / len(deltas), "paired_bootstrap_95ci": paired_interval(deltas), "fixed": [r["id"] for r in selected if r["correct"] and not baseline[r["id"]]["correct"]], "broken": [r["id"] for r in selected if not r["correct"] and baseline[r["id"]]["correct"]], "prediction_counts": dict(Counter(r["prediction"] for r in selected)), "gold_majority_accuracy": max(Counter(r["gold"] for r in selected).values()) / len(selected)})
        results.append({"name": name, "wikitext_nll": nll, "wikitext_subset_perplexity": math.exp(nll), "delta_nll": nll - base_nll, "nll_delta_paired_block_95ci": paired_interval([(r["nll_sum"] - baseline[r["id"]]["nll_sum"]) / r["scored_tokens"] for r in wiki]), "tasks": tasks})
    limits = "Small fixed retention subsets, single training seed. No LR/checkpoint selection. Perplexity is for sampled blocks, not full WikiText-2. Task scores are zero-shot label likelihoods, not generated instruction-following/coding quality. Bootstrap intervals reflect these sampled items/blocks, not training seeds; no multiple-comparison correction. Pretraining contamination unknown."
    write_json(args.root / "summary.json", {"runs": results, "sources": frozen["sources"], "limitations": limits})
    lines = ["# Non-math retention diagnostic", "", limits, "", "| Model | Wiki subset PPL (lower better) | Delta NLL | BoolQ raw | BoolQ chat | ARC-Easy raw | ARC-Easy chat |", "|---|---|---|---|---|---|---|"]
    for row in results:
        scores = " | ".join(f"{r['accuracy']:.1%} ({r['delta_accuracy'] * 100:+.1f} pp)" for r in row["tasks"])
        lines.append(f"| {row['name']} | {row['wikitext_subset_perplexity']:.2f} | {row['delta_nll']:+.4f} | {scores} |")
    lines.extend(["", "Detailed paired confidence intervals and fixed/broken IDs are in summary.json. Positive delta NLL means worse general-language loss; negative accuracy deltas mean worse task performance."])
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
