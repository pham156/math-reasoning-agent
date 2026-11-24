"""Plot measured greedy/sampling results and training dynamics for extended GRPO."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

from src.core.config import cache_dir, ensure_scratch_path, project_root
from src.evaluation.summarize_pilot_exploration import read_rows


def trailing_mean(values, window):
    if window < 1:
        raise ValueError("Positive smoothing window required")
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window) for i in range(len(values))]


def training_curve(history, metric):
    points = [(r["step"], r[metric]) for r in history if metric in r]
    expected = list(range(1, 513)) if metric in ("loss", "entropy") else list(range(1, 512, 4))
    if [x for x, _ in points] != expected or any(not math.isfinite(y) for _, y in points):
        raise ValueError(f"Incomplete or invalid {metric} curve")
    return points


def matched_greedy(root, manifest):
    baseline_ids = None
    values = []
    for step in (128, 256, 512):
        greedy_root = Path(manifest["control"]["greedy_root"]) if step == 128 else root / "evaluations" / f"step{step}"
        sampled_root = Path(manifest["control"]["sampled_root"]) if step == 128 else greedy_root
        greedy = read_rows(greedy_root, "greedy", "grpo", 300, 1)
        sampled = read_rows(sampled_root, "sampled", "grpo", 100, 8)
        if baseline_ids is None:
            baseline_ids = set(sampled)
        if set(sampled) != baseline_ids or not sampled.keys() <= greedy.keys():
            raise ValueError("Diagnostic subsets differ across checkpoints")
        if any(sampled[i]["prompt_hash"] != greedy[i]["prompt_hash"] for i in sampled):
            raise ValueError("Greedy/sampled prompts differ")
        values.append(sum(greedy[i]["candidates"][0]["correct"] for i in sampled) / 100)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    summary = json.loads((root / "summary_512.json").read_text())
    manifest = json.loads((root / "submission.json").read_text())
    rows = summary["rows"]
    if [r["step"] for r in rows] != [128, 256, 512] or any(r["n"] != 300 for r in rows):
        raise ValueError("Require complete validated checkpoint evaluations")
    matched = matched_greedy(root, manifest)
    destination = args.output_dir or project_root() / "figures" / root.name
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    exported = []
    steps = [r["step"] for r in rows]
    blue, orange, green, dark = "#0072B2", "#D55E00", "#23834F", "#333333"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    series = [("Greedy (300 problems)", "greedy_300", [r["accuracy"] for r in rows], blue, "-", "o"),
              ("Greedy (same 100 problems)", "greedy_matched_100", matched, blue, "--", "s"),
              ("Sampled pass@1 (100 problems)", "sampled_pass_at_1", [r["pass_at_k"]["1"] for r in rows], orange, "-", "o")]
    for label, metric, values, color, style, marker in series:
        axes[0].plot(steps, [v * 100 for v in values], label=label, color=color, linestyle=style, marker=marker, linewidth=2)
        exported.extend({"metric": metric, "step": x, "raw_value": y, "plotted_value": y} for x, y in zip(steps, values))
    for k, color, marker in ((4, green, "s"), (8, dark, "o")):
        values = [r["pass_at_k"][str(k)] for r in rows]
        axes[1].plot(steps, [v * 100 for v in values], label=f"Sampled pass@{k}", color=color, marker=marker, linewidth=2)
        exported.extend({"metric": f"sampled_pass_at_{k}", "step": x, "raw_value": y, "plotted_value": y} for x, y in zip(steps, values))
    axes[0].set(title="Single-answer accuracy", ylabel="Correct answers (%)", ylim=(60, 85))
    axes[1].set(title="Sampling coverage (same 100 problems)", ylabel="Estimated pass@k (%)", ylim=(80, 95))
    for axis in axes:
        axis.set(xlabel="Total optimizer updates", xticks=steps)
        axis.grid(axis="y", color="#E5E5E5")
        axis.legend(fontsize=8.5, loc="lower left", frameon=False)
    figure.suptitle("Longer GRPO training: deterministic vs sampled evaluation\nSeed 0 | same 512 training problems | measured checkpoints only", fontsize=12)
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"accuracy_and_sampling.{suffix}")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specifications = [("reward", "Training correctness reward", "Mean reward", 4, blue),
                      ("loss", "GRPO training policy loss (not cross-entropy)", "Policy surrogate loss", 16, orange),
                      ("entropy", "Training policy entropy", "Logged token entropy", 16, dark),
                      ("completions/mean_length", "Training response length", "Mean completion tokens", 4, green)]
    for axis, (metric, title, ylabel, window, color) in zip(axes.flat, specifications):
        points = training_curve(summary["training_history"], metric)
        xs, ys = zip(*points)
        smooth = trailing_mean(ys, window)
        if metric == "loss":
            axis.plot(xs, ys, color="#BBBBBB", linewidth=0.7, label="Raw per-update loss")
            axis.axhline(0, color="#777777", linewidth=0.7)
        axis.plot(xs, smooth, color=color, linewidth=1.8, label=f"Trailing {window}-observation mean")
        for checkpoint in (128, 256):
            axis.axvline(checkpoint, color="#999999", linestyle=":", linewidth=0.8)
        axis.set(title=title, ylabel=ylabel, xlabel="Optimizer update", xlim=(0, 512), xticks=[0, 128, 256, 384, 512])
        axis.grid(axis="y", color="#E5E5E5")
        axis.legend(fontsize=8, frameon=False)
        exported.extend({"metric": metric, "step": x, "raw_value": y, "plotted_value": z} for x, y, z in zip(xs, ys, smooth))
    figure.suptitle("GRPO training dynamics through 512 updates\nReward/length: 4 observations = up to 16 updates; loss/entropy: 16 updates", fontsize=12)
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"training_dynamics.{suffix}")
    plt.close(figure)
    with (destination / "plotted_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "step", "raw_value", "plotted_value"])
        writer.writeheader()
        writer.writerows(exported)
    (destination / "README.md").write_text(
        "# Extended GRPO Figures\n\n"
        "Source: " + str(root) + ". Single seed; repeated training on the same512 problems.\n\n"
        "Accuracy figure: three measured checkpoints only; connecting lines are not intermediate evaluations. "
        "Dashed greedy curve uses exactly the same100 problems as sampling; full greedy uses300. "
        "pass@1 is sampled accuracy, not greedy decoding. pass@4/8 measure oracle coverage, not answer selection.\n\n"
        "Training figure: logged GRPO policy loss, reward, entropy and completion length. "
        "Gray loss is raw; colored curves are the labeled trailing means. Dotted lines mark128/256 updates. "
        "The surrogate loss can be negative and is not a cross-entropy or correctness score; "
        "it need not decline monotonically. No verifier reward or SFT loss is shown. "
        "No synthetic uncertainty bands. All raw/plotted observations are in plotted_data.csv.\n"
    )
    print(f"Saved two PNG/PDF figures and {len(exported)} source points to {destination}")


if __name__ == "__main__":
    main()
