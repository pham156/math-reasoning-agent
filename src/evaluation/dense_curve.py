"""Assemble the every-16-update held-out curve for seed0 LR3e-6 GRPO (0..512).

Measured points only: new replay evaluations at 16..496 (dev300 + GSM8K-100 every 16,
MATH-500 every 64) plus the existing base/128/256/512 results. Each new point must come
from an export whose validation receipt proves rollout/metric equality with the original run.
"""

import argparse
import csv
import json
from pathlib import Path

from src.core.config import ensure_scratch_path, output_dir
from src.evaluation.aime_benchmarks import read_shards
from src.evaluation.pilot_benchmarks import summarize_rows
from src.evaluation.summarize_sft_study import dev300_rows
from src.training.grpo.train import write_json

STUDY = output_dir("rl_1p5b_study")


def anchors():
    out = {0: {}, 128: {}, 256: {}, 512: {}}
    controls = json.loads((STUDY / "controls_exploration_20261014/summary.json").read_text())["greedy_controls"]
    out[0]["dev300"] = next(r["accuracy"] for r in controls if r["name"] == "baseline")
    for row in json.loads((STUDY / "grpo_extended_20261105/summary_512.json").read_text())["rows"]:
        out[row["step"]]["dev300"] = row["accuracy"]
    labels = {"base_no_rl": 0, "step128": 128, "step256": 256, "step512": 512}
    for row in json.loads((STUDY / "checkpoint_benchmarks_20261106/summary.json").read_text())["runs"]:
        out[labels[row["model"]]][row["benchmark"]] = row["accuracy"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    m = json.loads((root / "submission.json").read_text())
    curve = {step: {**values, "source": "existing"} for step, values in anchors().items()}
    for step in m["eval_steps"]:
        export = root / "exports" / f"step-{step}"
        receipt = json.loads((export / "validation.json").read_text())
        if receipt["step"] != step or not receipt["rollouts_identical"] or not receipt["training_metrics_matched"]:
            raise ValueError(f"Unvalidated export {step}")
        evals = root / "evaluations" / f"step-{step}"
        point = {"source": "dense replay", "dev300": summarize_rows(dev300_rows(evals / "dev300/greedy/grpo", export))["accuracy"]}
        for bench, frozen in m["benchmarks"].items():
            if (evals / bench).exists():
                point[bench] = summarize_rows(read_shards(evals / bench / "greedy/grpo", Path(frozen), "greedy", export))["accuracy"]
        curve[step] = point
    replay_checks = {}
    for receipt in sorted((root / "exports").glob("step-*/validation.json")):
        data = json.loads(receipt.read_text())
        if "max_abs_weight_difference" in data:
            replay_checks[data["step"]] = data["max_abs_weight_difference"]
    steps = sorted(curve)
    write_json(root / "curve.json", {"curve": {str(s): curve[s] for s in steps}, "weight_checks": replay_checks, "note": "Greedy, common stop, 3000 cap; seed0 single run. Measured points only."})
    with (root / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "dev300", "gsm8k100", "math500", "source"])
        for s in steps:
            writer.writerow([s, curve[s].get("dev300", ""), curve[s].get("gsm8k100", ""), curve[s].get("math500", ""), curve[s]["source"]])
    print(f"{len(steps)} measured steps; weight checks: {replay_checks}")


if __name__ == "__main__":
    main()
