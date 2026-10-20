"""Export the midterm's line plots from completed pilot artifacts."""

import csv
import json
import math
import os

from src.core.config import cache_dir, output_dir, project_root


METHODS = {
    "baseline": ("No RL", "#333333", "--", "o"),
    "grpo": ("GRPO", "#0072B2", "-", "o"),
    "dr_grpo": ("Dr. GRPO", "#D55E00", "--", "s"),
}


def trailing_mean(values, window=4):
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window) for i in range(len(values))]


def pass_at_k(correct, k):
    return 1.0 if 8 - correct < k else 1 - math.comb(8 - correct, k) / math.comb(8, k)


def training_points(history, metric):
    points = [(r["step"], r[metric]) for r in history if metric in r]
    expected = list(range(1, 129)) if metric == "entropy" else list(range(1, 128, 4))
    if [x for x, _ in points] != expected or any(not math.isfinite(y) for _, y in points):
        raise ValueError(f"Incomplete or invalid training metric: {metric}")
    return points


def paired_lr_points(summary):
    points = {}
    for seed in (0, 1):
        values = []
        for rate in (1e-6, 3e-6):
            label = f"seed{seed}_lr{rate:.0e}"
            matches = [row for row in summary["runs"] if row["label"] == label]
            if len(matches) != 1 or matches[0]["n"] != 300:
                raise ValueError("Incomplete paired-seed learning-rate results")
            row = matches[0]
            if row["settings"]["seed"] != seed or row["settings"]["learning_rate"] != rate:
                raise ValueError("Learning-rate label and settings differ")
            values.append(row["accuracy"])
        points[seed] = values
    for index, rate in enumerate((1e-6, 3e-6)):
        if not math.isclose(sum(points[s][index] for s in (0, 1)) / 2, summary["two_seed_lr_summary"][str(rate)]["mean_accuracy"], abs_tol=1e-12):
            raise ValueError("Paired-seed mean differs from the validated summary")
    return points


def main():
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    study = output_dir("rl_1p5b_study")
    pilot = study / "pilot_20261013_000313/pilot"
    exploration = study / "controls_exploration_20261014"
    summary = json.loads((exploration / "summary.json").read_text())
    ablations = json.loads((study / "grpo_controls_20261017/summary.json").read_text())
    paired_lr_points(ablations)
    expected = {r["name"]: r for r in summary["exploration"]}
    destination = project_root() / "docs/midterm/figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.08))
    fig.subplots_adjust(left=0.069, right=0.988, bottom=0.22, top=0.72, wspace=0.38)
    exported = []
    for name, (label, color, style, marker) in METHODS.items():
        if name != "baseline":
            history = json.loads((pilot / name / "train_history.json").read_text())["log_history"]
            if max(r.get("step", 0) for r in history) != 128:
                raise ValueError(f"Incomplete training history: {name}")
            for axis, metric in ((axes[0], "reward"), (axes[2], "entropy")):
                points = training_points(history, metric)
                xs, ys = zip(*points)
                smoothed = trailing_mean(ys, window=8 if metric == "entropy" else 4)
                axis.plot(xs, smoothed, color=color, linestyle=style, linewidth=1.45,
                          marker=marker, markersize=2.5, markevery=16 if metric == "entropy" else 4)
                exported.extend({"method": name, "metric": metric, "x": x,
                                 "raw_value": y, "plotted_value": s} for x, y, s in zip(xs, ys, smoothed))
        rows = []
        for shard in range(5):
            rows.extend(json.loads(line) for line in (exploration / "sampled" / name / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip())
        if len(rows) != 100 or len({r["source_index"] for r in rows}) != 100 or any(len(r["candidates"]) != 8 for r in rows):
            raise ValueError(f"Incomplete sampled evaluation: {name}")
        ids = [r["source_index"] for r in rows]
        if name == "baseline":
            baseline_ids = ids
        if ids != baseline_ids:
            raise ValueError("Exploration problem order differs across models")
        scores = [sum(pass_at_k(sum(c["correct"] for c in r["candidates"]), k) for r in rows) / 100 for k in range(1, 9)]
        for k in (1, 4, 8):
            if not math.isclose(scores[k - 1], expected[name]["pass_at_k"][str(k)], abs_tol=1e-12):
                raise ValueError("pass@k does not match the published summary")
        axes[1].plot(range(1, 9), [100 * s for s in scores], color=color, linestyle=style,
                     marker=marker, markersize=3, linewidth=1.4)
        exported.extend({"method": name, "metric": "sampled_pass_at_k", "x": k,
                         "raw_value": s, "plotted_value": s} for k, s in enumerate(scores, 1))
    axes[0].set(title="(a) Training correctness reward", ylabel="Mean reward", xlabel="Optimizer update", xlim=(0, 128), ylim=(0, 0.7))
    axes[1].set(title="(b) Held-out sampling coverage", ylabel="Estimated pass@k (%)", xlabel="Sampling budget k", xlim=(0.8, 8.2), ylim=(20, 100))
    axes[1].set_xticks([1, 2, 4, 6, 8])
    axes[0].xaxis.set_major_locator(MultipleLocator(32))
    axes[2].set(title="(c) Policy entropy", ylabel="Logged token entropy", xlabel="Optimizer update", xlim=(0, 128))
    axes[2].set_ylim(bottom=0)
    axes[2].xaxis.set_major_locator(MultipleLocator(32))
    for axis in axes:
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.6)
    handles = [Line2D([], [], label=label, color=color, linestyle=style, marker=marker,
                      markersize=3, linewidth=1.5) for label, color, style, marker in METHODS.values()]
    fig.legend(handles=handles[1:], loc="upper center", bbox_to_anchor=(0.52, 0.99),
               ncol=2, frameon=False, fontsize=7.5, columnspacing=1.0, handlelength=2.3)
    handles[0].set_label("No RL (b only)")
    axes[1].legend(handles=handles[:1], loc="lower right", frameon=False, fontsize=7, handlelength=2.1)
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"rl_dynamics_and_exploration.{suffix}")
    plt.close(fig)
    with (destination / "plotted_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "metric", "x", "raw_value", "plotted_value"])
        writer.writeheader()
        writer.writerows(exported)
    print(f"Exported three line panels and {len(exported)} source points to {destination}")


if __name__ == "__main__":
    main()
