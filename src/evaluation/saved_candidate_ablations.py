"""Exhaustive, order-averaged analyses of already generated candidate groups."""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import math
import os
from pathlib import Path

from src.evaluation.answer import normalize_answer
from src.evaluation.checkpoint_ablations import load_condition, metrics, read, selection, verify_sources, write


RULES = ("first", "majority_first", "plurality", "majority_abstain")


def selection_weights(candidates, rule):
    """Marginalize uniform candidate order without inspecting correctness labels."""
    if not candidates or rule not in RULES:
        raise ValueError("Unknown rule or empty candidates")
    n = len(candidates)
    groups = defaultdict(list)
    for i, candidate in enumerate(candidates):
        answer = normalize_answer(candidate["answer"])
        if answer:
            groups[answer].append(i)
    largest = max(map(len, groups.values()), default=0)
    winners = [indices for indices in groups.values() if len(indices) == largest]
    majority = largest > n / 2
    if rule == "first" or (rule == "majority_first" and not majority):
        return [1 / n] * n
    if rule == "majority_abstain" and not majority:
        return [0.0] * n
    if rule == "plurality" and not winners:
        return [1 / n] * n
    # Equal-size plurality ties are equally likely to appear first in a random order.
    weights = [0.0] * n
    for indices in winners:
        for i in indices:
            weights[i] = 1 / (len(winners) * len(indices))
    return weights


def subset_metrics(candidates, k, rule):
    if not 1 <= k <= len(candidates):
        raise ValueError("Invalid candidate budget")
    totals = Counter()
    count = math.comb(len(candidates), k)
    for indices in combinations(range(len(candidates)), k):
        subset = [candidates[i] for i in indices]
        weights = selection_weights(subset, rule)
        totals["accuracy"] += sum(w * c["correct"] for w, c in zip(weights, subset))
        totals["coverage"] += sum(weights)
        totals["oracle_accuracy"] += any(c["correct"] for c in subset)
        totals["completion_tokens"] += sum(c["generated_tokens"] for c in subset)
    return {key: value / count for key, value in totals.items()}


def diversity(candidates):
    answers = [normalize_answer(c["answer"]) for c in candidates]
    counts = Counter(answers)
    n = len(answers)
    return {"unique_nonempty_answers": len(set(answers) - {""}),
            "answer_entropy_nats": -sum((v / n) * math.log(v / n) for v in counts.values()),
            "modal_answer_mass": max(counts.values()) / n,
            "empty_answer_rate": counts.get("", 0) / n,
            "correct_candidate_mass": sum(c["correct"] for c in candidates) / n,
            "mixed_correctness": 0 < sum(c["correct"] for c in candidates) < n}


def analyze_rows(rows):
    budgets = []
    for k in (1, 2, 4, 8):
        for rule in RULES:
            per_problem = [subset_metrics(row["candidates"], k, rule) for row in rows]
            result = {key: sum(r[key] for r in per_problem) / len(rows) for key in per_problem[0]}
            result.update(k=k, rule=rule, n=len(rows), subsets_per_problem=math.comb(8, k))
            result["answered_accuracy"] = result["accuracy"] / result["coverage"] if result["coverage"] else None
            budgets.append(result)
    details = [{"source_index": r["source_index"], "type": r["type"], "level": r["level"],
                **diversity(r["candidates"])} for r in rows]
    groups = {}
    for field in ("type", "level"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field] or "unspecified"].append(row)
        groups["by_" + field] = {name: metrics(part) for name, part in grouped.items()}
    return {"candidate_budgets": budgets, "diversity": details,
            "mean_diversity": {key: sum(r[key] for r in details) / len(details)
                               for key in diversity(rows[0]["candidates"])},
            "ordered_eight_candidate_metrics": metrics(rows), **groups}


def paired_breakdown(rows, controls):
    if [r["source_index"] for r in rows] != [r["source_index"] for r in controls]:
        raise ValueError("Paired cohorts differ")
    records = []
    for row, control in zip(rows, controls):
        after = row["candidates"][selection(row["candidates"])["selected_index"]]["correct"]
        before = control["candidates"][selection(control["candidates"])["selected_index"]]["correct"]
        records.append({"source_index": row["source_index"], "type": row["type"], "level": row["level"],
                        "before": before, "after": after})
    groups = {}
    for field in ("type", "level"):
        grouped = defaultdict(list)
        for record in records:
            grouped[record[field] or "unspecified"].append(record)
        groups["by_" + field] = {label: {"n": len(part),
            "control_accuracy": sum(r["before"] for r in part) / len(part),
            "selected_accuracy": sum(r["after"] for r in part) / len(part),
            "fixed": [r["source_index"] for r in part if not r["before"] and r["after"]],
            "broken": [r["source_index"] for r in part if r["before"] and not r["after"]]}
            for label, part in grouped.items()}
    return groups


def analyze(source, destination, plots=False):
    if destination.exists():
        raise ValueError("Refuse to overwrite an existing analysis")
    manifest = read(source / "campaign.json")
    verify_sources(manifest)
    if not read(source / "smoke_passed.json")["passed"]:
        raise ValueError("Source smoke did not pass")
    results, evidence, saved = [], {}, {}
    for condition in manifest["conditions"]:
        if condition["mode"] != "sampled":
            continue
        rows, hashes = load_condition(condition, manifest["cohort"])
        saved[(condition["checkpoint"], condition["temperature"])] = rows
        evidence.update(hashes)
        results.append({"condition": condition["key"], "checkpoint": condition["checkpoint"],
                        "temperature": condition["temperature"], **analyze_rows(rows)})
    if len(results) != 12:
        raise ValueError("Expected four checkpoints x three temperatures")
    for result in results:
        result["paired_vs_base_same_temperature"] = paired_breakdown(
            saved[(result["checkpoint"], result["temperature"])], saved[("base", result["temperature"])])
    note = ("Saved eight-candidate groups on the same 100 development problems. All subsets at N=1/2/4/8, "
            "with exact expectation over uniform random candidate order, not cherry-picked prefixes. "
            "First means a random first member; majority requires >N/2 nonempty matching answers; "
            "plurality ties randomize tied groups; all-empty plurality falls back to a random member. "
            "Abstentions count as incorrect overall. Correctness labels are used only for scoring. "
            "Costs count completion tokens in the subset, not latency or an actual resampled run. "
            "Grouping is textual normalization, not mathematical equivalence. Entropy is final-answer "
            "entropy, not token-policy entropy. Subsets are correlated and are not extra independent questions.")
    write(destination / "analysis.json", {"note": note, "runs": results, "evidence_sha256": evidence})
    flat = [{"checkpoint": r["checkpoint"], "temperature": r["temperature"], **row}
            for r in results for row in r["candidate_budgets"]]
    with (destination / "candidate_budgets.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Saved-Candidate Ablations", "", note, "",
             "## Eight-Candidate Selection Rules (Order-Averaged)", "",
             "| Checkpoint | T | Random first | Majority + random-first fallback | Plurality | Majority-only accuracy / coverage | Unique answers | Answer entropy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        full = {b["rule"]: b for b in r["candidate_budgets"] if b["k"] == 8}
        d = r["mean_diversity"]
        lines.append(f"| {r['checkpoint']} | {r['temperature']} | {full['first']['accuracy']:.2%} | {full['majority_first']['accuracy']:.2%} | {full['plurality']['accuracy']:.2%} | {full['majority_abstain']['accuracy']:.2%} / {full['majority_abstain']['coverage']:.1%} | {d['unique_nonempty_answers']:.2f} | {d['answer_entropy_nats']:.3f} |")
    lines += ["", "## Category and Level Transfer: Step512 vs Base at T=0.7", "",
              "Fixed-order eight-candidate strict majority with first-candidate fallback (the original protocol). Small subgroup counts are descriptive, not independent confirmation.", "",
              "| Group | n | Base | Step512 | Fixed | Broken | Net |", "|---|---:|---:|---:|---:|---:|---:|"]
    paired = next(r for r in results if r["checkpoint"] == "step512" and r["temperature"] == 0.7)["paired_vs_base_same_temperature"]
    for field in ("by_type", "by_level"):
        for label, values in sorted(paired[field].items()):
            fixed, broken = len(values["fixed"]), len(values["broken"])
            lines.append(f"| {label} | {values['n']} | {values['control_accuracy']:.1%} | {values['selected_accuracy']:.1%} | {fixed} | {broken} | {fixed - broken:+d} |")
    lines += ["", "Category/level breakdowns for every condition, per-problem diversity, oracle coverage, and every candidate budget are in analysis.json. Original fixed-order selection metrics are retained separately for comparison with earlier reports."]
    (destination / "analysis.md").write_text("\n".join(lines) + "\n")
    if plots:
        plot(results, destination)
    print("\n".join(lines))


def plot(results, destination):
    from src.core.config import cache_dir
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for checkpoint in ("base", "step128", "step256", "step512"):
        runs = sorted((r for r in results if r["checkpoint"] == checkpoint), key=lambda r: r["temperature"])
        run = next(r for r in runs if r["temperature"] == 0.7)
        budgets = [b for b in run["candidate_budgets"] if b["rule"] == "majority_first"]
        axes[0, 0].plot([b["k"] for b in budgets], [100 * b["accuracy"] for b in budgets], "o-", label=checkpoint)
        axes[0, 1].plot([b["completion_tokens"] for b in budgets], [100 * b["accuracy"] for b in budgets], "o-", label=checkpoint)
        axes[1, 0].plot([r["temperature"] for r in runs], [r["mean_diversity"]["answer_entropy_nats"] for r in runs], "o-", label=checkpoint)
        axes[1, 1].plot([r["temperature"] for r in runs], [100 * r["mean_diversity"]["correct_candidate_mass"] for r in runs], "o-", label=checkpoint)
    settings = [("Candidate budget (T=0.7)", "Candidates", "Order-averaged accuracy (%)"),
                ("Completion-token tradeoff (T=0.7)", "Mean group completion tokens", "Order-averaged accuracy (%)"),
                ("Answer diversity", "Temperature", "Final-answer entropy (nats)"),
                ("Correct-answer probability mass", "Temperature", "Mean correct candidate fraction (%)")]
    for axis, (title, xlabel, ylabel) in zip(axes.flat, settings):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"saved_candidate_ablations.{suffix}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    analyze(args.source, args.destination, args.plots)


if __name__ == "__main__":
    main()
