"""Plot completed common-stop controls and sampled exploration in home."""

import argparse
import json
import os
from pathlib import Path

from src.core.config import cache_dir, ensure_scratch_path, project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    ensure_scratch_path(args.root)
    data = json.loads((args.root / "summary.json").read_text())
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#888888", "#167d9a", "#d1495b", "#34864c", "#7654a3", "#444444"]
    root = project_root() / "figures" / args.root.name
    root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    controls = data["greedy_controls"]
    names = [r["name"] for r in controls]
    values = [100 * r["accuracy"] for r in controls]
    bars = axes[0, 0].bar(names, values, color=colors)
    axes[0, 0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("Common-stop greedy accuracy | 300 problems")
    axes[0, 0].set_ylabel("Accuracy (%)")
    for row, color in zip(data["exploration"], colors):
        axes[0, 1].plot([1, 4, 8], [100 * row["pass_at_k"][str(k)] for k in [1, 4, 8]], marker="o", label=row["name"], color=color)
    axes[0, 1].set_title("Sampled pass@k | same 100 problems, 8 samples each")
    axes[0, 1].set_xticks([1, 4, 8])
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Estimated pass@k (%)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(names, [r["avg_tokens"] for r in controls], color=colors)
    axes[1, 0].set_title("Common-stop greedy response length")
    axes[1, 0].set_ylabel("Mean generated tokens")
    axes[1, 1].bar(names, [100 * r["truncated"] / r["n"] for r in controls], color=colors)
    axes[1, 1].set_title("Common-stop greedy truncation rate")
    axes[1, 1].set_ylabel("Responses hitting the token cap (%)")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Pilot evaluation controls and exploration\nSingle seed, reserved development data; sampled pass@1 differs from greedy accuracy", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"controls_and_exploration.{suffix}", dpi=180)
    plt.close(fig)
    print(root / "controls_and_exploration.png")


if __name__ == "__main__":
    main()
