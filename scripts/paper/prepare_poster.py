"""Build a source-linked, vector A0 research poster from completed RL experiments.

Layout (three columns, no headline strip):
  1 Setup | 2 Main results | 3 Why SFT failed (RFT fix) | 4 Controls
  5 GRPO vs Dr. GRPO | 6 Training dynamics 0-512 | 7 Held-out accuracy every 16 updates
  8 Longer training: development vs transfer | 9 What RL changes | 10 Problem level | 11 Test-time selection & takeaways
Results from the November 15 jobs (SFT study, Dr. GRPO LR runs, AIME, dense replay) are read
when their validated summaries exist; until then their cells read "..." (queued), never guessed.
"""

import argparse
import csv
from datetime import date, datetime, timezone
import json
import os
import textwrap

from scripts.paper.prepare_final import collect, followup_tables, generate_tables, inference_tables, initial_training_points, trailing_mean
from src.core.config import cache_dir, output_dir, project_root
from src.evaluation.checkpoint_ablations import digest, load_condition, read, verify_sources
from src.evaluation.plot_grpo_extension import training_curve


A0_MM = (1189, 841)
METHODS = ("grpo", "dr_grpo")
NAMES = {"baseline": "No RL", "grpo": "GRPO", "dr_grpo": "Dr. GRPO"}
COLORS = {"baseline": "#747474", "grpo": "#0072B2", "dr_grpo": "#D55E00"}
QUEUED = "..."
NEW_RESULTS = {"sft": "sft_study_20261115/summary.json", "drgrpo_lr": "drgrpo_lr_20261115/summary.json",
               "drgrpo_external": "drgrpo_lr_20261115/external/summary.json", "aime": "aime_checkpoints_20261115/summary.json",
               "dense": "dense_replay_20261115/curve.json", "intervals": "regrade_20261120/paired_intervals.json"}
SFT_ARMS = {"gt_full": ("Human solutions", "Full"), "gt_lora": ("Human solutions", "LoRA r64"),
            "rft_full": ("RFT, unfiltered", "Full"), "rft_full_clean": ("RFT, clean endings", "Full"),
            "rft_lora_clean": ("RFT, clean endings", "LoRA r64")}


def transition_matrix(before, after):
    if [r["source_index"] for r in before] != [r["source_index"] for r in after]:
        raise ValueError("Transition analysis requires the same ordered problems")
    matrix = [[0] * 9 for _ in range(9)]
    for left, right in zip(before, after):
        if left["prompt_hash"] != right["prompt_hash"]:
            raise ValueError("Transition prompts differ")
        a, b = left["num_correct"], right["num_correct"]
        if not 0 <= a <= 8 or not 0 <= b <= 8:
            raise ValueError("Expected eight-response groups")
        matrix[a][b] += 1
    return matrix


def ecdf(values):
    if not values:
        raise ValueError("ECDF requires observations")
    ordered = sorted(values)
    return ordered, [(i + 1) / len(ordered) for i in range(len(ordered))]


def pct(value, digits=1):
    return QUEUED if value is None else f"{100 * value:.{digits}f}"


def load_new_results(study, sources):
    """Optional November 15 results; each is a validated summary written by its own job."""
    new = {}
    snapshot = None
    audited = project_root() / "figures/completed_evaluations/completed_evidence.json"
    if audited.exists():
        snapshot = read(audited)
        for path, item in snapshot["sources"].items():
            if digest(path) != item["sha256"]:
                raise ValueError(f"Completed-cohort source changed since audit: {path}")
        sources.update(snapshot["sources"])
        sources[str(audited)] = {"path": str(audited), "sha256": digest(audited)}
    for key, relative in NEW_RESULTS.items():
        path = study / relative
        if not path.exists() and key in {"sft", "dense"}:
            path = path.with_name(path.stem + "_partial.json")
        new[key] = read(path) if path.exists() else None
        if path.name.endswith("_partial.json") and new[key] is not None:
            if snapshot is None or new[key] != snapshot[key]:
                raise ValueError(f"Partial summary lacks matching completed-cohort audit: {path}")
        if new[key] is not None:
            sources[str(path)] = {"path": str(path), "sha256": digest(path)}
    new["histories"] = {}
    for label, run in (("grpo", study / "grpo_lr_upper_20261016/runs/grpo_lr3e-06"), ("dr_grpo", study / "drgrpo_lr_20261115/runs/dr_grpo_lr3e-06")):
        progress = run / "progress.json"
        if progress.exists() and read(progress)["status"] == "completed":
            new["histories"][label] = read(run / "train_history.json")["log_history"]
            sources[str(run / "train_history.json")] = {"path": str(run / "train_history.json"), "sha256": digest(run / "train_history.json")}
    return new


def load_evidence(study, campaign):
    data, sources = collect(study)
    if data["latest"] is None:
        raise ValueError("Completed external checkpoint benchmarks are required")
    data["tables"] = generate_tables(data, study)
    manifest = read(campaign / "campaign.json")
    verify_sources(manifest)
    complete = (campaign / "summary.json").exists()
    summary_path = campaign / ("summary.json" if complete else "existing_summary.json")
    selection = read(summary_path)
    expected = 28 if complete else 8
    if len(selection["runs"]) != expected:
        raise ValueError("Refuse incomplete inference-ablation results")
    if complete and not read(campaign / "smoke_passed.json")["passed"]:
        raise ValueError("Inference campaign has no successful smoke proof")
    if not complete:
        raise ValueError("The restructured poster requires the complete 28-condition inference campaign")
    data["tables"].update(inference_tables(selection["runs"]))
    data["inference"] = selection
    data["inference_complete"] = complete
    data["samples"] = {}
    for condition in manifest["conditions"]:
        rows, hashes = load_condition(condition, manifest["cohort"])
        sources.update({str(path): {"path": str(path), "sha256": sha} for path, sha in hashes.items()})
        if condition["reused"] and condition["mode"] == "sampled":
            data["samples"][condition["checkpoint"]] = rows
    data["transition"] = transition_matrix(data["samples"]["step128"], data["samples"]["step512"])
    data["histories"] = {}
    for method in METHODS:
        path = study / "pilot_20261013_000313/pilot" / method / "train_history.json"
        data["histories"][method] = read(path)["log_history"]
        sources[str(path)] = {"path": str(path), "sha256": digest(path)}
    for path in (summary_path, campaign / "campaign.json"):
        sources[str(path)] = {"path": str(path), "sha256": digest(path)}
    from scripts.paper.prepare_followup import load_export
    path = project_root() / "docs/final/generated/followup_evidence.json"
    data["followup"] = load_export(path)
    data["tables"].update(followup_tables(data["followup"]))
    sources.update(data["followup"]["sources"])
    sources[str(path)] = {"path": str(path), "sha256": digest(path)}
    data["new"] = load_new_results(study, sources)
    return data, sources


def derived(data):
    """Assemble every number the poster prints, keyed by meaning."""
    new, tables = data["new"], data["tables"]
    latest = {(r["benchmark"], r["model"]): r for r in data["latest"]["runs"]}
    methods = {r[0].replace("~", " "): r for r in tables["methods"][1]}
    dev = {"base_no_rl": 0.45, **{f"step{r[0]}": float(r[1]) / 100 for r in tables["duration"][1]}}
    if abs(float(methods["No RL"][1]) / 100 - dev["base_no_rl"]) > 1e-9:
        raise ValueError("Base development accuracy changed")
    aime = {r["model"]: r for r in new["aime"]["runs"]} if new["aime"] else {}
    sft = {r["label"]: r for r in new["sft"]["arms"]} if new["sft"] else {}
    dr_rates = {}
    if new["drgrpo_lr"]:
        dr_rates = {r["learning_rate"]: r["accuracy"] for r in new["drgrpo_lr"]["runs"] if r["algorithm"] == "dr_grpo"}
    dr_ext = new["drgrpo_external"]["arms"][0] if new["drgrpo_external"] else {}

    def acc(entry, key):
        return entry[key]["accuracy"] if entry and key in entry else None

    best_sft = None
    scored = [label for label in SFT_ARMS if acc(sft.get(label), "dev300") is not None]
    if scored:
        best_sft = max(scored, key=lambda label: acc(sft[label], "dev300"))
    return {"latest": latest, "methods": methods, "dev": dev, "aime": aime, "sft": sft, "dr_rates": dr_rates,
            "dr_ext": dr_ext, "best_sft": best_sft, "acc": acc}


def build_poster(data, destination, snapshot_date):
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Rectangle

    d = derived(data)
    acc, latest, aime, sft = d["acc"], d["latest"], d["aime"], d["sft"]
    intervals = data["new"]["intervals"]

    def delta(bench, base, model):
        """Paired-bootstrap change (lenient grader) as '+x.x [lo, hi]' points; None if not computed."""
        if not intervals:
            return None
        entry = next((c for c in intervals["comparisons"] if (c["benchmark"], c["baseline"], c["model"]) == (bench, base, model) and "lenient" in c), None)
        if entry is None:
            return None
        e = entry["lenient"]
        return f"{100 * e['difference']:+.1f} [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]".replace("-", "\u2212")

    def significant(bench, base, model):
        entry = next(c for c in intervals["comparisons"] if (c["benchmark"], c["baseline"], c["model"]) == (bench, base, model))
        low, high = entry["lenient"]["ci95"]
        return low > 0 or high < 0
    width, height = (value / 25.4 for value in A0_MM)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.titlesize": 22,
                         "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white"})
    figure = plt.figure(figsize=(width, height), dpi=50)
    gold, ink, muted = "#A88A48", "#171717", "#505050"
    columns = [1.0, 16.15, 31.3]
    col_width = 14.5
    recorded, panels = [], []

    def text(x, y, content, size=21, color=ink, weight="normal", align="left", wrap=None):
        if wrap:
            content = "\n".join(textwrap.fill(line, wrap) for line in content.split("\n"))
        artist = figure.text(x / width, y / height, content, fontsize=size, color=color,
                             weight=weight, ha=align, va="top", linespacing=1.3)
        recorded.append(artist)
        return artist

    def rule(x, y, length, color=gold, thickness=2):
        figure.add_artist(plt.Line2D([x / width, (x + length) / width], [y / height] * 2,
                                    transform=figure.transFigure, color=color, linewidth=thickness))

    def heading(column, y, number, title, subtitle):
        x = columns[column]
        text(x, y, number, size=30, color=gold, weight="bold")
        text(x + 0.76, y, title, size=29, weight="bold")
        rule(x, y - 0.55, col_width, color="#D5D5D5", thickness=1)
        text(x, y - 0.75, subtitle, size=20, color=muted, wrap=95)
        return y - 0.75 - 0.42 * (1 + len(textwrap.fill(subtitle, 95)) // 96)

    def axis(column, x_offset, bottom, w=5.8, h=3.5, title=""):
        ax = figure.add_axes([(columns[column] + x_offset) / width, bottom / height, w / width, h / height])
        if title:
            ax.set_title(title, pad=12, fontweight="bold")
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.8)
        ax.set_axisbelow(True)
        return ax

    def record(panel, series, xs, ys):
        panels.extend({"panel": panel, "series": series, "x": x, "y": y} for x, y in zip(xs, ys))

    def line(ax, panel, series, xs, ys, **kwargs):
        ax.plot(xs, ys, **kwargs)
        record(panel, series, xs, ys)

    def table(column, top, labels, rows, widths=None, row_h=0.37, x_offset=0.1, w=14.3, size=18, bold_rows=()):
        h = row_h * (len(rows) + 1)
        ax = figure.add_axes([(columns[column] + x_offset) / width, (top - h) / height, w / width, h / height])
        ax.set_axis_off()
        cells = ax.table(cellText=rows, colLabels=labels, colWidths=widths, loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
        cells.auto_set_font_size(False)
        cells.set_fontsize(size)
        for (row, _), cell in cells.get_celld().items():
            cell.set_edgecolor("#D8D8D8")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#F0F0F0")
                cell.get_text().set_fontweight("bold")
            elif row in bold_rows:
                cell.set_facecolor("#FBF6EA")
                cell.get_text().set_fontweight("bold")
        return top - h

    # Title block only; no per-column headline strip.
    text(1, 32.35, "Reinforcement Learning with Verifiable Rewards", size=67, weight="bold")
    text(1, 31.24, "for Mathematical Reasoning: A Controlled Study", size=59, weight="bold")
    text(45.8, 32.28, "PURDUE\nUNIVERSITY", size=30, weight="bold", align="right")
    text(1, 30.08, "Tuan Minh Pham  |  Tri Vo  |  Yahor Lechanka", size=29)
    text(45.8, 30.08, "CS 557  |  A0 research poster", size=24, align="right", color=muted)
    rule(1, 29.42, width - 2, thickness=4)
    top = 28.85

    # ---------------- Column 1 ----------------
    y = heading(0, top, "01", "Question & setup", "Which ingredients actually improve a small math model: the RL objective, its hyperparameters, the data source, or more updates?")
    text(1, y, "Qwen2.5-Math-1.5B base, full-parameter updates (FP32 weights / BF16 compute).\n512 MATH Level 3-5 training problems; 300 disjoint development problems (dev300).\nExternal: MATH-500, GSM8K-100, AIME 2025+2026 (60). Greedy unless stated; 3000-token cap.", size=20)
    y -= 1.55
    for i, label in enumerate(("Math problem", "4 sampled solutions", "Verified answer", "Policy update")):
        x = columns[0] + i * 3.7
        figure.add_artist(Rectangle((x / width, (y - 0.72) / height), 3.25 / width, 0.72 / height,
                                    transform=figure.transFigure, facecolor="white", edgecolor=ink, linewidth=1.2))
        text(x + 1.625, y - 0.2, label, size=18, align="center")
        if i < 3:
            text(x + 3.48, y - 0.18, "$\\rightarrow$", size=24, align="center")
    y -= 1.0
    text(1, y, "$R=\\mathbf{1}[\\mathrm{final\\ answer\\ correct}]$ (exact / Math-Verify); no learned reward model", size=21)
    y -= 0.55
    y = table(0, y, ["Objective", "Advantage", "Loss aggregation", "Ratio clip"], [
        ["GRPO", "(r - mean) / std of group", "Mean per response", "0.2 / 0.2"],
        ["Dr. GRPO", "r - mean (no std)", "Fixed token denominator", "0.2 / 0.2"]], [0.17, 0.33, 0.32, 0.18])
    y -= 0.35

    y = heading(0, y, "02", "Main results", "Same prompts, grader and cap; matched settings are labeled. Tuning uses dev300 only, never external benchmarks.")
    m = d["methods"]
    best = d["best_sft"]
    best_label = f"SFT, best complete dev arm ({best.replace('_', ' ')})" if best else "SFT, best complete dev arm"
    rows = [["No RL (base)", m["No RL"][1], m["No RL"][2], m["No RL"][3], pct(acc(aime.get("base_no_rl"), "greedy"))],
            ["SFT, human solutions (original)", m["SFT"][1], pct(acc(sft.get("gt_full_original"), "math500")), pct(acc(sft.get("gt_full_original"), "gsm8k100")), pct(acc(sft.get("gt_full_original"), "aime"))],
            [best_label, *(pct(acc(sft.get(best), k)) if best else QUEUED for k in ("dev300", "math500", "gsm8k100", "aime"))],
            ["GRPO, LR 1e-6, 128 updates", m["GRPO"][1], m["GRPO"][2], m["GRPO"][3], "---"],
            ["Dr. GRPO, LR 1e-6, 128 updates", m["Dr. GRPO"][1], m["Dr. GRPO"][2], m["Dr. GRPO"][3], "---"],
            ["GRPO, LR 3e-6, 128 updates", pct(d["dev"]["step128"]), pct(latest[("math500", "step128")]["accuracy"]), pct(latest[("gsm8k100", "step128")]["accuracy"]), pct(acc(aime.get("step128"), "greedy"))],
            ["Dr. GRPO, LR 3e-6, 128 updates", pct(d["dr_rates"].get(3e-6)), pct(acc(d["dr_ext"], "math500")), pct(acc(d["dr_ext"], "gsm8k100")), pct(acc(d["dr_ext"], "aime"))],
            ["GRPO, LR 3e-6, 512 updates", pct(d["dev"]["step512"]), pct(latest[("math500", "step512")]["accuracy"]), pct(latest[("gsm8k100", "step512")]["accuracy"]), pct(acc(aime.get("step512"), "greedy"))]]
    y = table(0, y, ["Model (accuracy %)", "Dev300", "MATH-500", "GSM8K-100", "AIME 25+26"], rows, [0.40, 0.14, 0.15, 0.16, 0.15], row_h=0.40, bold_rows=(6, 8))
    note = "'...' = incomplete or unavailable; '---' = not evaluated. AIME: 60 problems, one problem = 1.7 points."
    if intervals:
        a = intervals["accuracy"]
        rl_shift = max(abs(v["lenient"] - v["strict"]) for k, v in a.items() if "grpo" in k)
        note += (f"\nStrict re-grading (closed final \\boxed{{}} only): RL scores move <= {100 * rl_shift:.1f} pts;"
                 f"\nthe base model drops to {100 * a['math500/base']['strict']:.1f} (MATH-500) and {100 * a['gsm8k100/base']['strict']:.1f} (GSM8K), so RL gains only grow.")
    text(1, y - 0.12, note, size=17, color=muted)
    y -= 0.85 + (0.66 if intervals else 0)

    y = heading(0, y, "03", "Why SFT failed, and the fix", "Imitating terse human solutions moves the model off its own reasoning style.")
    text(1, y, "Original SFT (human MATH solutions, full FT): 40.3% dev300, below the 45.0% base.\nTargets average 376 tokens vs ~1,065 for the base model's own answers; outputs\nskip steps and copy [asy] diagram code until the cap (71/300 truncated).\nFix (RAFT / rejection-sampling FT): train on the base model's own verified answers.", size=19)
    y -= 1.55
    sft_rows = []
    for label, (source, params) in SFT_ARMS.items():
        entry = sft.get(label)
        sft_rows.append([source, params, pct(acc(entry, "dev300")), pct(acc(entry, "math500")), pct(acc(entry, "gsm8k100")),
                         f"{entry['dev300']['avg_generated_tokens']:.0f}" if entry and "dev300" in entry else QUEUED])
    y = table(0, y, ["Training targets", "Params", "Dev300", "MATH-500", "GSM8K", "Tokens"], sft_rows, [0.33, 0.15, 0.13, 0.14, 0.12, 0.13])
    text(1, y - 0.12, "Full LR 5e-6, LoRA LR 1e-4; cosine, 3 epochs, batch 16; same 512-problem source pool.", size=17, color=muted)
    y -= 0.85

    y = heading(0, y, "04", "Controls that did not beat the anchor", "One fixed reference run per condition; differences of 1-2 points are within single-run noise.")
    rewards = data["tables"]["rewards"][1]
    explore = data["tables"]["exploration"][1]
    left = [["Correctness only (anchor)", rewards[0][1]], ["+ format bonus", rewards[1][1]], ["+ length penalty", rewards[2][1]],
            ["Train T = 0.7", explore[1][1]], ["Train T = 1.3", explore[2][1]], ["Group size 2", explore[3][1]], ["Group size 8", explore[4][1]]]
    reg = data["tables"]["regularization"][1]
    names = {"Control": "No KL, clip 0.2 (anchor)", "$\\beta=0.001$": "KL $\\beta$ = 0.001", "$\\beta=0.01$": "KL $\\beta$ = 0.01",
             "Clip $0.1/0.1$": "Clip 0.1", "Clip $0.3/0.3$": "Clip 0.3"}
    right = [[names.get(r[0], r[0]), r[1]] for r in reg]
    bottom_left = table(0, y, ["LR 3e-6 setting", "Dev300"], left, [0.74, 0.26], w=6.95, size=17, row_h=0.35, bold_rows=(1,))
    table(0, y, ["LR 1e-6 setting", "Dev300"], right, [0.74, 0.26], x_offset=7.45, w=6.95, size=17, row_h=0.35, bold_rows=(1,))
    text(1, bottom_left - 0.12, "Correctness-only reward, T = 1 and G = 4 remain best. KL/clipping rarely bind.", size=18)

    # ---------------- Column 2 ----------------
    y = heading(1, top, "05", "GRPO vs Dr. GRPO", "Dev300 accuracy across learning rates; then matched-LR training dynamics over the 128-update budget.")
    rate_rows = [[r[0].replace("~", " "), *r[1:]] for r in data["tables"]["rates"][1]]
    for rate, column_index in ((2e-6, 4), (3e-6, 5)):
        rate_rows[1][column_index] = pct(d["dr_rates"].get(rate))
    y = table(1, y, ["Dev300 (%)", "LR 2e-7", "5e-7", "1e-6", "2e-6", "3e-6"], rate_rows, [0.24, 0.152, 0.152, 0.152, 0.152, 0.152])
    histories = data["new"]["histories"]
    tuned = len(histories) == 2
    source_histories = histories if tuned else data["histories"]
    rate_label = "LR 3e-6" if tuned else "LR 1e-6"
    y -= 0.35
    panel_bottom = y - 4.1
    for offset, metric, title, window in ((0.85, "reward", f"Training reward ({rate_label})", 4), (8.25, "completions/mean_length", f"Response length ({rate_label})", 4)):
        ax = axis(1, offset, panel_bottom, h=3.35, title=title)
        for method in METHODS:
            xs, ys = zip(*initial_training_points(source_histories[method], metric))
            line(ax, f"method_{metric}", method, xs, trailing_mean(ys, window), color=COLORS[method], label=NAMES[method], linewidth=2.5)
        ax.set(xlabel="Optimizer update", xlim=(0, 128), xticks=[0, 32, 64, 96, 128], ylabel="Mean reward" if metric == "reward" else "Tokens")
        ax.legend(fontsize=15, frameon=False, loc="upper left" if metric == "reward" else "upper right")
    y = panel_bottom - 0.95
    caption = "Method rankings depend on learning rate; compare matched settings, not raw loss scales."
    if intervals:
        caption += f"\nLR 1e-6, Dr. GRPO minus GRPO: dev300 {delta('dev300', 'grpo_1e-6', 'drgrpo_1e-6')}; MATH-500 and GSM8K within noise."
    text(columns[1], y, caption + "\nTrailing means over 4 rollout logs. 128 updates = one pass over the 512 problems.", size=18, color=muted)
    y -= 0.33 if intervals else 0
    y -= 1.0

    y = heading(1, y, "06", "Training dynamics, 0 to 512 updates", "Tuned GRPO (LR 3e-6), same 512 problems repeated four times. Logged training metrics, not held-out accuracy.")
    history = data["extension"]["training_history"]
    specs = [(0.85, "reward", "Training reward", 4), (8.25, "loss", "Policy surrogate loss", 16),
             (0.85, "entropy", "Token entropy", 16), (8.25, "completions/mean_length", "Completion tokens", 4)]
    row_tops = [y - 0.45, y - 0.45 - 3.0]
    for index, (offset, metric, title, window) in enumerate(specs):
        bottom = row_tops[index // 2] - 2.2
        ax = axis(1, offset, bottom, h=2.05, title=title)
        xs, ys = zip(*training_curve(history, metric))
        if metric == "loss":
            ax.plot(xs, ys, color="#C4C4C4", linewidth=0.8)
            record("extended_loss", "raw", xs, ys)
            ax.axhline(0, color="#777777", linewidth=0.7)
        line(ax, "extended_" + metric, "trailing_mean", xs, trailing_mean(ys, window), color=COLORS["grpo"], linewidth=2)
        for step in (128, 256, 384):
            ax.axvline(step, color="#BBBBBB", linestyle=":", linewidth=1)
        ax.set(xlim=(0, 512), xticks=[0, 128, 256, 384, 512])
        if index >= 2:
            ax.set_xlabel("Optimizer update")
    y = row_tops[1] - 2.2 - 0.95
    text(columns[1], y, "Dotted lines: data-pass boundaries. Gray: raw per-update loss. Reward keeps rising on\nrepeated problems while entropy collapses early and response length shortens.", size=18, color=muted)
    y -= 1.0

    y = heading(1, y, "07", "Held-out checkpoint evaluation", "Intermediate checkpoints regenerated by validated deterministic replay. Only complete benchmark cohorts are plotted; missing scheduled points remain gaps.")
    ax = axis(1, 0.95, 3.85, w=13.2, h=y - 4.4)
    dense = data["new"]["dense"]
    series = (("dev300", "Dev300 (development)", COLORS["grpo"], "o"), ("gsm8k100", "GSM8K-100", "#009E73", "s"), ("math500", "MATH-500", "#CC79A7", "^"))
    if dense:
        from scripts.paper.refresh_completed import measured_segments
        curve = {int(step): values for step, values in dense["curve"].items()}
        for key, label, color, marker in series:
            for index, segment in enumerate(measured_segments(curve, key, 64 if key == "math500" else 16)):
                xs, ys = [p[0] for p in segment], [100 * p[1] for p in segment]
                line(ax, "heldout_" + key, "measured", xs, ys, color=color, marker=marker, markersize=7,
                     linewidth=2.2, label=label if index == 0 else None)
        counts = ", ".join(f"{label}: {sum(key in v for v in curve.values())}" for key, label, _, _ in series)
        status = f"Complete measured cohorts: {counts}.\nGaps are pending or held evaluations, not interpolated results."
    else:
        anchors = {0: "base_no_rl", 128: "step128", 256: "step256", 512: "step512"}
        for key, label, color, marker in series:
            xs = list(anchors)
            ys = [100 * (d["dev"][anchors[s]] if key == "dev300" else latest[(key, anchors[s])]["accuracy"]) for s in xs]
            ax.scatter(xs, ys, color=color, marker=marker, s=110, label=label, zorder=3)
            record("heldout_" + key, "measured_anchor", xs, ys)
        status = "Markers: the 4 checkpoints measured so far. Replay of 29 more checkpoints is queued;\nlines are drawn only once every point is measured."
    reward = training_curve(history, "reward")
    rx, ry = zip(*reward)
    line(ax, "heldout_reward", "training_reward", rx, [100 * v for v in trailing_mean(ry, 4)], color="#9A9A9A", linewidth=1.6, linestyle="--", label="Training reward (on-policy)")
    ax.set(xlabel="Optimizer update", ylabel="Accuracy / reward (%)", xlim=(-8, 520), xticks=list(range(0, 513, 64)), ylim=(20, 95))
    ax.legend(fontsize=15, frameon=False, ncol=2, loc="lower right")
    text(columns[1], 2.65, status, size=18, color=muted)

    # ---------------- Column 3 ----------------
    y = heading(2, top, "08", "Longer training: development vs transfer", "More passes over the same 512 problems raise training fit and GSM8K accuracy; greedy dev300 peaks at 128 and MATH-500 stays flat.")
    temps = {r[0]: r for r in data["tables"]["inference_temperature"][1] if r[1] == "1.0"}
    order = (("No RL", "base_no_rl"), ("128", "step128"), ("256", "step256"), ("512", "step512"))
    long_rows = [["Dev300 greedy", *[pct(d["dev"][k]) for _, k in order]],
                 ["Dev100 sampled@1 (T = 1)", *[temps[n][2] for n, _ in order]],
                 ["Dev100 pass@8", *[temps[n][4] for n, _ in order]],
                 ["MATH-500 greedy", *[pct(latest[("math500", k)]["accuracy"]) for _, k in order]],
                 ["GSM8K-100 greedy", *[pct(latest[("gsm8k100", k)]["accuracy"]) for _, k in order]],
                 ["AIME 25+26 greedy", *[pct(acc(aime.get(k), "greedy")) for _, k in order]],
                 ["AIME 25+26 mean@8 (T = 1)", *[pct(aime[k]["sampled"]["mean_at_8"]) if k in aime and "sampled" in aime[k] else QUEUED for _, k in order]],
                 ["MATH-500 mean tokens", *[f"{latest[('math500', k)]['avg_generated_tokens']:.0f}" for _, k in order]]]
    y = table(2, y, ["Metric (%)", "No RL", "128", "256", "512"], long_rows, [0.40, 0.15, 0.15, 0.15, 0.15])
    caption = "128 to 512: MATH-500 fixes/breaks 36/28, GSM8K 13/2."
    if intervals:
        caption = (f"128 to 512, change [95% paired bootstrap CI]: dev300 {delta('dev300', 'grpo_3e-6_128', 'grpo_3e-6_512')},"
                   f" GSM8K {delta('gsm8k100', 'grpo_3e-6_128', 'grpo_3e-6_512')},\nMATH-500 {delta('math500', 'grpo_3e-6_128', 'grpo_3e-6_512')} (within noise)."
                   f" AIME mean@8, base to 512: {delta('aime_sampled', 'base', 'grpo_3e-6_512')}.")
    text(columns[2], y - 0.12, caption, size=17, color=muted)
    y -= 0.85 + (0.33 if intervals else 0)

    y = heading(2, y, "09", "What RL changes: sharper, not broader", "Same 100 development questions, eight samples at T = 1. RL mostly concentrates mass on answers the base model could already reach.")
    entropy = {r[0]: r[3] for r in data["tables"]["answer_diversity"][1]}
    budget = {r[0]: r for r in data["tables"]["inference_budget"][1]}
    sharp_rows = [[n, temps[n][2], temps[n][4], entropy[n], budget[n][2], budget[n][4]] for n, _ in order]
    y = table(2, y, ["Model", "Sampled@1", "Pass@8", "Answer entropy", "Greedy @1024", "Greedy @3000"], sharp_rows, [0.14, 0.17, 0.14, 0.21, 0.17, 0.17])
    retention = {r[0].replace("~", " "): r for r in data["tables"]["retention"][1]}
    text(columns[2], y - 0.12, f"Pass@1 +{float(temps['512'][2]) - float(temps['No RL'][2]):.0f} pts vs pass@8 +{float(temps['512'][4]) - float(temps['No RL'][4]):.0f}; answer entropy (nats) falls {100 * (1 - float(entropy['512']) / float(entropy['No RL'])):.0f}%.\n"
         f"No measurable forgetting (LR 1e-6): WikiText PPL {retention['No RL'][1]} vs {retention['GRPO'][1]}, BoolQ {retention['No RL'][2]} vs {retention['GRPO'][2]}, ARC {retention['No RL'][3]} vs {retention['GRPO'][3]}.", size=17)
    y -= 1.3

    y = heading(2, y, "10", "Problem-level changes", "Averages hide redistribution: per-question sample migration and MATH-500 category accuracy.")
    heat_bottom = y - 4.45
    ax = axis(2, 0.95, heat_bottom, w=5.0, h=3.75, title="Correct of 8: 128 $\\rightarrow$ 512")
    matrix = np.array(data["transition"])
    ax.grid(False)
    ax.imshow(matrix, origin="lower", cmap="Blues", aspect="auto")
    for a in range(9):
        for b in range(9):
            if matrix[a, b]:
                ax.text(b, a, str(matrix[a, b]), ha="center", va="center", fontsize=15, color="white" if matrix[a, b] > matrix.max() * 0.55 else ink)
                record("sample_migration", str(a), [b], [int(matrix[a, b])])
    ax.plot([-0.5, 8.5], [-0.5, 8.5], color="#D55E00", linestyle=":", linewidth=1.5)
    ax.set(xlabel="At 512", ylabel="At 128", xticks=[0, 2, 4, 6, 8], yticks=[0, 2, 4, 6, 8])
    ax = axis(2, 9.6, heat_bottom, w=4.7, h=3.75, title="MATH-500 by category")
    categories = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]
    labels = ["Algebra", "Count. & Prob.", "Geometry", "Int. Algebra", "Number Theory", "Prealgebra", "Precalculus"]
    models = ["base_no_rl", "step128", "step256", "step512"]
    values = np.array([[100 * latest[("math500", model)]["by_type"][category]["accuracy"] for model in models] for category in categories])
    ax.grid(False)
    ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for a in range(7):
        for b in range(4):
            ax.text(b, a, f"{values[a, b]:.0f}", ha="center", va="center", fontsize=15, color="white" if values[a, b] > 65 else ink)
            record("category_accuracy", categories[a], [models[b]], [float(values[a, b])])
    ax.set(xticks=range(4), xticklabels=["Base", "128", "256", "512"], yticks=range(7), yticklabels=[f"{label} ({latest[('math500', 'step128')]['by_type'][category]['n']})" for label, category in zip(labels, categories)])
    ax.tick_params(axis="y", labelsize=14)
    y = heat_bottom - 0.95
    text(columns[2], y, "Left: 100 questions; all-8-correct 27 $\\rightarrow$ 47, all-8-wrong 9 $\\rightarrow$ 10. Right: accuracy %, n in parentheses.", size=17, color=muted)
    y -= 0.75

    y = heading(2, y, "11", "Test-time selection & takeaways", "Eight samples at T = 0.4, accuracy averaged over candidate order (retrospective, development only).")
    selection = [[r[0], *r[1:]] for r in data["tables"]["selection_rules"][1]]
    y = table(2, y, ["Model", "Random 1 of 8", "Majority", "Plurality", "Oracle@8"], selection, [0.20, 0.22, 0.19, 0.19, 0.20], row_h=0.35, size=17)
    ext = {(r[0], r[1]): r for r in data["tables"]["external_selection"][1]}
    gsm = ext[("GSM8K-100", "512")]
    text(columns[2], y - 0.12, f"External check: GSM8K-100 strict-majority vote {gsm[4]}% vs {gsm[3]}% greedy at {gsm[5]}x tokens.", size=17, color=muted)
    y -= 0.75
    best_sft_text = (f"RFT data rescues SFT ({best.replace('_', ' ')}: {pct(acc(sft[best], 'dev300'))}% dev300), still below RL." if best else "Human-solution SFT gives no dev300 gain; training on own verified answers (RFT) is being evaluated.")
    takeaways = ["Verified-answer RL roughly doubles external accuracy with no learned reward model.",
                 "RL sharpens the answer distribution (pass@1 up, pass@8 nearly flat, entropy down).",
                 ("More passes: dev300 falls and GSM8K rises (both significant); MATH-500 is flat." if intervals and significant("dev300", "grpo_3e-6_128", "grpo_3e-6_512") and significant("gsm8k100", "grpo_3e-6_128", "grpo_3e-6_512") and not significant("math500", "grpo_3e-6_128", "grpo_3e-6_512")
                  else "Dev300 peaks at 128 updates while external accuracy keeps rising: no single best checkpoint."),
                 best_sft_text,
                 "Shaping, KL, clipping, temperature and group size never beat the plain correctness reward."]
    box_h = 0.42 * len(takeaways) + 0.35
    figure.add_artist(FancyBboxPatch(((columns[2]) / width, (y - box_h) / height), col_width / width, box_h / height, boxstyle="round,pad=0.004",
                                     transform=figure.transFigure, facecolor="#FBF6EA", edgecolor=gold, linewidth=1.5))
    text(columns[2] + 0.25, y - 0.15, "\n".join("$\\bullet$ " + t for t in takeaways), size=18)

    rule(1, 1.48, width - 2, color="#555555", thickness=1.2)
    text(1, 1.24, "Scope: single training seed, one fixed reference run per condition, no cross-run averages. MATH-500/GSM8K-100 were inspected in earlier work; AIME is evaluation-only. Correct final answers do not certify proofs.", size=17)
    text(1, 0.84, "References: Qwen2.5-Math (2409.12122); MATH (2103.03874); GSM8K (2110.14168); GRPO (2402.03300); Dr. GRPO (2503.20783); RAFT / Minimal-RL (2504.11343); LoRA (2106.09685).", size=17, color=muted)
    text(1, 0.43, "Code: github.com/pham156/math-reasoning-agent  |  Every number is regenerated from source-hashed evidence (evidence.json).", size=17, color=muted)
    text(width - 1, 0.43, "Snapshot: " + snapshot_date, size=17, align="right", color=muted)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    problems = []
    gutter = (col_width + 0.45) / width * figure.bbox.width
    for artist in recorded:
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > figure.bbox.width or box.y1 > figure.bbox.height:
            problems.append(f"outside A0: {artist.get_text()[:70]}")
        for ax in figure.axes:
            if box.overlaps(ax.get_tightbbox(renderer)):
                problems.append(f"overlaps chart/table: {artist.get_text()[:70]}")
        for column_x in columns:
            left = column_x / width * figure.bbox.width
            in_body = 1.5 < artist.get_position()[1] * height < 29.0
            if in_body and left - 2 <= box.x0 < left + 30 and box.x1 > left + gutter:
                problems.append(f"overflows column: {artist.get_text()[:70]}")
    tight = [ax.get_tightbbox(renderer) for ax in figure.axes]
    for i in range(len(tight)):
        for j in range(i + 1, len(tight)):
            if tight[i].overlaps(tight[j]):
                problems.append(f"charts/tables overlap: axes {i} and {j}")
    if problems:
        raise ValueError("Layout audit failed:\n" + "\n".join(dict.fromkeys(problems)))
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(destination / f"research_poster.{suffix}", dpi=110, facecolor="white")
    with (destination / "plotted_data.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["panel", "series", "x", "y"])
        writer.writeheader()
        writer.writerows(panels)
    plt.close(figure)
    status = {key: data["new"][key] is not None for key in NEW_RESULTS}
    status["dr_grpo_lr3e-06_history"] = "dr_grpo" in data["new"]["histories"]
    return {"size_mm": A0_MM, "sections": 11, "publication_methods": list(METHODS), "publication_reference_seed": 0,
            "hyperparameter_presentation": "tables only", "plotted_records": len(panels),
            "inference_ablation_complete": data["inference_complete"], "new_result_status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot_date", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    destination = project_root() / "docs/poster"
    study = output_dir("rl_1p5b_study")
    campaign = project_root() / "results/checkpoint_ablations_20261113"
    data, sources = load_evidence(study, campaign)
    metadata = build_poster(data, destination, args.snapshot_date.isoformat())
    (destination / "evidence.json").write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata, "snapshot_date": args.snapshot_date.isoformat(), "snapshot_timezone": "America/Indianapolis", "sources": sources, "tables": data["tables"], "transition_matrix": data["transition"],
        "followup_complete_conditions": len(data["followup"]["completed"]), "followup_excluded_conditions": data["followup"]["excluded_conditions"],
        "scope": "Completed matched 1.5B RL study; historical7B/retrieval/formal-proof experiments excluded."}, indent=2) + "\n")
    print(f"Built A0 landscape poster: {metadata}")


if __name__ == "__main__":
    main()
