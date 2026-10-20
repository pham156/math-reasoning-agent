"""Export pilot training curves and complete development accuracies to PNG/PDF."""

import argparse
import csv
import json
import os
from pathlib import Path

from src.core.config import cache_dir, ensure_scratch_path, project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_root", required=True, type=Path)
    parser.add_argument("--comparison_root", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, help="Figure directory; defaults to PROJECT_ROOT/figures/<comparison-name>")
    args = parser.parse_args()
    ensure_scratch_path(args.pilot_root)
    ensure_scratch_path(args.comparison_root)
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output_dir or project_root() / "figures" / args.comparison_root.name
    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 180})
    names = {"grpo": "GRPO", "dr_grpo": "Dr. GRPO", "dapo_style": "DAPO-style", "gspo": "GSPO"}
    colors = {"grpo": "#167d9a", "dr_grpo": "#d1495b", "dapo_style": "#34864c", "gspo": "#7654a3", "sft": "#444444", "baseline": "#888888"}
    histories = {name: json.loads((args.pilot_root / "pilot" / name / "train_history.json").read_text())["log_history"] for name in names}
    curves = [("reward", "Training correctness reward", False), ("completions/mean_length", "Mean training response length (tokens)", False), ("frac_reward_zero_std", "Groups with mixed rewards", True), ("completions/clipped_ratio", "Training responses hitting token cap", False), ("entropy", "Training policy entropy", False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    exported = []
    for axis, (metric, title, invert) in zip(axes.flat, curves):
        for name, history in histories.items():
            points = [(row["step"], 1 - row[metric] if invert else row[metric]) for row in history if metric in row]
            if not points:
                continue
            xs, ys = zip(*points)
            axis.plot(xs, ys, color=colors[name], alpha=0.18, linewidth=1)
            smooth = [sum(ys[max(0, i - 3):i + 1]) / len(ys[max(0, i - 3):i + 1]) for i in range(len(ys))]
            axis.plot(xs, smooth, color=colors[name], label=names[name], linewidth=1.8)
            exported.extend({"method": name, "metric": "mixed_reward_group_rate" if invert else metric, "step": step, "value": value} for step, value in points)
        axis.set_title(title)
        axis.set_xlabel("Optimizer update")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=9)
    sft = json.loads((args.comparison_root / "sft/train_history.json").read_text())["log_history"]
    points = [(r["step"], r["loss"]) for r in sft if "loss" in r]
    axes.flat[5].plot(*zip(*points), color=colors["sft"], alpha=0.7)
    axes.flat[5].set_title("SFT assistant-token cross-entropy")
    axes.flat[5].set_xlabel("Optimizer update")
    axes.flat[5].grid(alpha=0.2)
    exported.extend({"method": "sft", "metric": "cross_entropy", "step": step, "value": value} for step, value in points)
    fig.suptitle("512-problem pilot training | seed 0\nFaint curves: raw values; RL bold curves: trailing 4-observation mean", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"training_dashboard.{suffix}")
    plt.close(fig)
    with (root / "training_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "metric", "step", "value"])
        writer.writeheader()
        writer.writerows(exported)

    complete = []
    for name in ["baseline", *names, "sft"]:
        path = args.comparison_root / "evaluation" / f"{name}_summary.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["n"] == 300:
                complete.append((name, row))
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar([names.get(name, name.upper()) for name, _ in complete], [100 * row["accuracy"] for _, row in complete], color=[colors[name] for name, _ in complete])
    for bar, (_, row) in zip(bars, complete):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{row['accuracy']:.1%}", ha="center")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Greedy answer accuracy (%)")
    axis.set_title("Shared 300-problem development set | seed 0\nCompleted evaluations only; reserved MATH-train data")
    axis.grid(axis="y", alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"development_accuracy.{suffix}")
    plt.close(fig)
    (root / "README.md").write_text("# Pilot figures\n\nTraining dashboard: all four RL methods and SFT on the same 512 training IDs. RL bold curves use a trailing four-observation mean, with raw values faintly shown. SFT loss and RL reward are different objectives, not comparable loss scales. Training metrics are not held-out accuracy. KL was not logged and is not plotted.\n\nDevelopment chart: only completed 300-problem evaluations, one seed, greedy decoding. No incomplete SFT or baseline results are substituted. Rerun after all evaluations complete to include them. No uncertainty estimate across training seeds is available.\n\nPNG files are previews; PDF files are exportable figures; training_metrics.csv contains raw plotted observations.\n")
    print(f"Saved plots to {root}")


if __name__ == "__main__":
    main()
