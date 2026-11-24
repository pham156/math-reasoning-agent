"""Generate final-report tables, line figures and provenance from completed studies."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone

from src.core.config import cache_dir, output_dir, project_root
from scripts.paper.plot_midterm import trailing_mean, training_points
from src.evaluation.plot_grpo_extension import training_curve, matched_greedy


LABELS = {"baseline": "No RL", "sft": "SFT", "grpo": "GRPO", "dr_grpo": "Dr.~GRPO"}
PUBLICATION_METHODS = ("grpo", "dr_grpo")
PUBLICATION_REFERENCE_SEED = 0
REPORT_FIGURES = ("method_dynamics", "extended_training")


def initial_training_points(history, metric):
    if metric != "loss":
        return training_points(history, metric)
    points = [(row["step"], row[metric]) for row in history if metric in row]
    if [step for step, _ in points] != list(range(1, 129)) or any(not math.isfinite(value) for _, value in points):
        raise ValueError("Incomplete or invalid initial per-update loss")
    return points


def pct(value, digits=1):
    return "---" if value is None else f"{100 * value:.{digits}f}"


def table(headers, rows, caption, label, wide=False):
    environment = "table*" if wide else "table"
    return (f"\\begin{{{environment}}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            + "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\toprule\n"
            + " & ".join(headers) + "\\\\\n\\midrule\n"
            + "".join(" & ".join(map(str, row)) + "\\\\\n" for row in rows)
            + f"\\bottomrule\n\\end{{tabular}}\n\\end{{{environment}}}\n")


def validate_latest(latest, proofs):
    expected = {(benchmark, model): n for benchmark, n in (("math500", 500), ("gsm8k100", 100))
                for model in ("base_no_rl", "step128", "step256", "step512")}
    runs = latest["runs"]
    if len(runs) != len(expected) or {(r["benchmark"], r["model"]) for r in runs} != set(expected):
        raise ValueError("Latest benchmark summary is incomplete or duplicated")
    if any(r["n"] != expected[(r["benchmark"], r["model"])] for r in runs):
        raise ValueError("Latest benchmark sample count changed")
    for step in (192, 256):
        proof = proofs[step]
        if proof["step"] != step or proof["rollouts_identical"] is not True or proof["training_metrics_matched"] is not True:
            raise ValueError("Reconstructed checkpoint failed replay validation")
    if proofs[192]["development"]["n"] != 100:
        raise ValueError("The extra development point must use 100 problems")


def inference_tables(runs):
    models = ("base", "step128", "step256", "step512")
    expected = {(model, "greedy", 0.0, cap) for model in models for cap in (512, 1024, 2048, 3000)}
    expected.update((model, "sampled", temperature, 3000) for model in models for temperature in (0.4, 0.7, 1.0))
    index = {(r["checkpoint"], r["mode"], r["temperature"], r["cap"]): r for r in runs}
    if len(runs) != 28 or set(index) != expected:
        raise ValueError("Incomplete or duplicate inference conditions")
    if any(r["n"] != 100 or r["samples_per_problem"] != (8 if r["mode"] == "sampled" else 1) for r in runs):
        raise ValueError("Inference cohort or candidate count changed")
    labels = {"base": "No RL", "step128": "128", "step256": "256", "step512": "512"}
    budgets = [[labels[model], *[pct(index[(model, "greedy", 0.0, cap)]["first_accuracy"]) for cap in (512, 1024, 2048, 3000)]] for model in models]
    sampled = [[labels[model], str(temperature), pct(r["pass_at_1"], 2), pct(r["selected_accuracy"]), pct(r["pass_at_8"]), pct(r["majority_rate"]), f"{r['avg_problem_tokens']:.0f}"]
               for model in models for temperature in (0.4, 0.7, 1.0)
               for r in [index[(model, "sampled", temperature, 3000)]]]
    return {"inference_budget": (["Checkpoint", "512", "1024", "2048", "3000"], budgets),
            "inference_temperature": (["Checkpoint", "$T$", "Sampled@1", "Consensus", "Pass@8", "Majority", "Group tokens"], sampled)}


def load_inference_evidence(campaign, sources):
    from src.evaluation.checkpoint_ablations import digest, load_condition, metrics, read, verify_sources
    if not (campaign / "summary.json").exists():
        return None
    manifest, summary = read(campaign / "campaign.json"), read(campaign / "summary.json")
    verify_sources(manifest)
    if not read(campaign / "smoke_passed.json")["passed"]:
        raise ValueError("Inference smoke has not passed")
    inference_tables(summary["runs"])
    indexed = {r["key"]: r for r in summary["runs"]}
    for condition in manifest["conditions"]:
        rows, hashes = load_condition(condition, manifest["cohort"])
        measured = metrics(rows)
        if any(indexed[condition["key"]][key] != measured[key] for key in ("first_accuracy", "selected_accuracy", "pass_at_1", "pass_at_8", "total_generated_tokens")):
            raise ValueError("Inference summary differs from raw candidate evidence")
        sources.update({path: {"path": path, "sha256": value} for path, value in hashes.items()})
    for path in (campaign / "summary.json", campaign / "campaign.json", campaign / "smoke_passed.json"):
        sources[str(path)] = {"path": str(path), "sha256": digest(path)}
    return summary


def followup_tables(data):
    from scripts.paper.prepare_followup import validate_payload
    validate_payload(data)
    labels = {"base": "No RL", "step128": "128", "step256": "256", "step512": "512"}
    index = {(r["checkpoint"], r["dataset"], r["prompt_variant"]): r for r in data["completed"]}
    prompts = [[labels[model], pct(index[(model, "development", "reworded")]["control_accuracy"]),
                *[pct(index[(model, "development", variant)]["selected_accuracy"]) for variant in ("reworded", "concise")]]
               for model in labels]
    selectors = []
    for model in labels:
        run = next(r for r in data["saved"] if r["checkpoint"] == model and r["temperature"] == .4)
        rules = {r["rule"]: r for r in run["candidate_budgets"] if r["k"] == 8}
        selectors.append([labels[model], *[pct(rules[rule]["accuracy"], 2) for rule in ("first", "majority_first", "plurality")],
                          pct(rules["first"]["oracle_accuracy"])])
    external = [["MATH-500" if r["dataset"] == "math500" else "GSM8K-100", labels[r["checkpoint"]],
                 str(r["n"]), pct(r["control_accuracy"]), pct(r["selected_accuracy"]),
                 f"{r['avg_problem_tokens'] / r['control_tokens']:.2f}"]
                for r in data["completed"] if r["dataset"] != "development"]
    run = next(r for r in data["saved"] if r["checkpoint"] == "step256" and r["temperature"] == .4)
    candidate_rows = [[label, *[pct(next(b["accuracy"] for b in run["candidate_budgets"] if b["k"] == k and b["rule"] == rule), 2)
                               for k in (1, 2, 4, 8)]]
                      for rule, label in (("first", "Random first"), ("majority_first", "Majority"), ("plurality", "Plurality"))]
    diversity = [[labels[model], *[f"{next(r['mean_diversity']['answer_entropy_nats'] for r in data['saved'] if r['checkpoint'] == model and r['temperature'] == t):.3f}"
                                  for t in (.4, .7, 1.0)]] for model in labels]
    return {"prompt_sensitivity": (["Checkpoint", "Original", "Reworded", "Concise"], prompts),
            "selection_rules": (["Checkpoint", "Random first", "Majority", "Plurality", "Oracle@8"], selectors),
            "external_selection": (["Dataset", "Policy", "n", "Greedy", "Vote", "Tokens $\\times$"], external),
            "candidate_budget": (["Rule", "$N=1$", "$N=2$", "$N=4$", "$N=8$"], candidate_rows),
            "answer_diversity": (["Checkpoint", "$T=0.4$", "$T=0.7$", "$T=1.0$"], diversity)}


def plot_followup(data, destination):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
    colors = ("#747474", "#0072B2", "#D55E00", "#218558")
    for model, color in zip(("base", "step128", "step256", "step512"), colors):
        label = "No RL" if model == "base" else model.removeprefix("step")
        run = next(r for r in data["saved"] if r["checkpoint"] == model and r["temperature"] == .7)
        budget = [r for r in run["candidate_budgets"] if r["rule"] == "majority_first"]
        axes[0, 0].plot([r["k"] for r in budget], [100 * r["accuracy"] for r in budget], "o-", color=color, label=label, markersize=3)
        runs = sorted((r for r in data["saved"] if r["checkpoint"] == model), key=lambda r: r["temperature"])
        axes[0, 1].plot([r["temperature"] for r in runs], [r["mean_diversity"]["answer_entropy_nats"] for r in runs], "o-", color=color, label=label, markersize=3)
        prompts = {r["prompt_variant"]: r for r in data["completed"] if r["checkpoint"] == model and r["dataset"] == "development"}
        axes[1, 0].plot([0, 1, 2], [100 * prompts["reworded"]["control_accuracy"],
                       100 * prompts["reworded"]["selected_accuracy"], 100 * prompts["concise"]["selected_accuracy"]],
                       "o-", color=color, label=label, markersize=3)
    run = next(r for r in data["saved"] if r["checkpoint"] == "step256" and r["temperature"] == .4)
    for rule, label, color in (("first", "Random first", "#747474"), ("majority_first", "Majority", "#0072B2"),
                               ("plurality", "Plurality", "#D55E00")):
        budget = [r for r in run["candidate_budgets"] if r["rule"] == rule]
        axes[1, 1].plot([r["k"] for r in budget], [100 * r["accuracy"] for r in budget], "o-", color=color, label=label, markersize=3)
    specifications = [("(a) Candidate budget, T=0.7", "Candidates", "Order-averaged accuracy (%)"),
                      ("(b) Final-answer diversity", "Temperature", "Answer entropy (nats)"),
                      ("(c) Prompt sensitivity, greedy", "System instruction", "Accuracy (%)"),
                      ("(d) Selection: checkpoint 256, T=0.4", "Candidates", "Order-averaged accuracy (%)")]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, specifications):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=6, frameon=False)
    axes[0, 0].set_xticks([1, 2, 4, 8])
    axes[1, 1].set_xticks([1, 2, 4, 8])
    axes[1, 0].set_xticks([0, 1, 2], ["Original", "Reworded", "Concise"])
    for suffix in ("pdf", "png"):
        figure.savefig(destination / f"followup_ablations.{suffix}")
    plt.close(figure)


def collect(study):
    sources = {}

    def read(relative):
        file = study / relative
        sources[relative] = {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        return json.loads(file.read_text())

    controls = read("controls_exploration_20261014/summary.json")
    external = read("external_benchmarks_20261015/summary.json")
    low = read("lr_tuning_20261014/summary.json")
    high = read("grpo_lr_upper_20261016/summary.json")
    tuning = read("grpo_controls_20261017/summary.json")
    reward = read("reward_exploration_20261024/reward/summary.json")
    exploration = read("reward_exploration_20261024/exploration/summary.json")
    extension = read("grpo_extended_20261105/summary_512.json")
    retention = read("retention_20261014/summary.json")
    latest, proofs = None, {}
    latest_root = "checkpoint_benchmarks_20261106"
    if (study / latest_root / "summary.json").exists():
        latest = read(f"{latest_root}/summary.json")
        proofs = {step: read(f"{latest_root}/replay_{step}_check.json") for step in (192, 256)}
        validate_latest(latest, proofs)
    if any(r["n"] != 300 for r in controls["greedy_controls"]):
        raise ValueError("Incomplete initial development evaluation")
    if [r["step"] for r in extension["rows"]] != [128, 256, 512]:
        raise ValueError("Incomplete training-duration study")
    return dict(controls=controls, external=external, low=low, high=high, tuning=tuning,
                reward=reward, exploration=exploration, extension=extension, retention=retention,
                latest=latest, replay_proofs=proofs), sources


def reference_reward_rows(runs):
    rows = []
    for design, label in (("correctness", "Correctness"), ("format_bonus", "Format bonus"), ("length_penalty", "Length penalty")):
        values = [r for r in runs if r["settings"]["reward_design"] == design and r["settings"]["seed"] == PUBLICATION_REFERENCE_SEED]
        if len(values) != 1:
            raise ValueError("Missing or duplicate fixed-reference reward condition")
        value = values[0]
        rows.append([label, pct(value["accuracy"]), f"{value['avg_generated_tokens']:.0f}"])
    return rows


def generate_tables(data, study):
    result = {}
    dev = {r["name"]: r for r in data["controls"]["greedy_controls"]}
    sampled = {r["name"]: r for r in data["controls"]["exploration"]}
    external = {(r["name"], r["benchmark"]): r for r in data["external"]["runs"]}
    rows = []
    for name in ("baseline", "sft", *PUBLICATION_METHODS):
        scores = [external.get((name, benchmark), {}) for benchmark in ("math500", "gsm8k100")]
        p = sampled.get(name, {}).get("pass_at_k", {})
        rows.append([LABELS[name], pct(dev[name]["accuracy"]), *[pct(r.get("accuracy")) for r in scores],
                     pct(p.get("1"), 2), pct(p.get("8")), f"{scores[0]['avg_generated_tokens']:.0f}" if scores[0] else "---"])
    result["methods"] = (["Method", "Dev300", "MATH-500", "GSM8K-100", "Sampled@1", "Pass@8", "MATH tokens"], rows)
    rates = [2e-7, 5e-7, 1e-6, 2e-6, 3e-6]
    lr = {(r["algorithm"], r["learning_rate"]): r["accuracy"] for r in data["low"]["runs"]}
    lr.update({(r["algorithm"], r["learning_rate"]): r["accuracy"] for r in data["high"]["runs"]})
    result["rates"] = (["Method", "$2\\cdot10^{-7}$", "$5\\cdot10^{-7}$", "$10^{-6}$", "$2\\cdot10^{-6}$", "$3\\cdot10^{-6}$"],
                       [[LABELS[name], *[pct(lr.get((name, rate))) for rate in rates]] for name in PUBLICATION_METHODS])
    runs = {r["label"]: r for r in data["tuning"]["runs"]}
    labels = {"seed0_lr1e-06": "Control", "kl_1e-03": "$\\beta=0.001$", "kl_1e-02": "$\\beta=0.01$",
              "clip_0p1": "Clip $0.1/0.1$", "clip_0p3": "Clip $0.3/0.3$"}
    result["regularization"] = (["Setting", "Dev300", "Prefix KL", "Clip fraction"],
                                [[label, pct(runs[name]["accuracy"]), f"{runs[name]['fixed_prefix_reference_kl']:.4f}", f"{runs[name]['clip_fraction_mean']:.6f}"] for name, label in labels.items()])
    result["rewards"] = (["Reward", "Dev300", "Tokens"], reference_reward_rows(data["reward"]["runs"]))
    labels = {"correctness_seed0": "Control ($T=1,G=4$)", "temperature_0p7": "$T=0.7$", "temperature_1p3": "$T=1.3$", "rollouts_2": "$G=2$", "rollouts_8": "$G=8$"}
    result["exploration"] = (["Training setting", "Greedy", "Sampled@1", "Pass@8", "Train M tokens"],
                             [[labels[r["label"]], pct(r["accuracy"]), pct(r["pass_at_k"]["1"], 2), pct(r["pass_at_k"]["8"]), f"{r['training']['completion_tokens']/1e6:.2f}"] for r in data["exploration"]["runs"]])
    manifest = json.loads((study / "grpo_extended_20261105/submission.json").read_text())
    matched = matched_greedy(study / "grpo_extended_20261105", manifest)
    result["duration"] = (["Updates", "Greedy300", "Greedy100", "Sampled@1", "Pass@8", "Train M tokens"],
                          [[str(r["step"]), pct(r["accuracy"]), pct(g), pct(r["pass_at_k"]["1"], 2), pct(r["pass_at_k"]["8"]), f"{r['training_completion_tokens']/1e6:.2f}"] for r, g in zip(data["extension"]["rows"], matched)])
    retention = []
    for r in data["retention"]["runs"]:
        if r["name"] not in ("baseline", *PUBLICATION_METHODS):
            continue
        tasks = {(t["task"], t["format"]): t for t in r["tasks"]}
        retention.append([LABELS[r["name"]], f"{r['wikitext_subset_perplexity']:.2f}", pct(tasks[("boolq", "raw")]["accuracy"]), pct(tasks[("arc_easy", "raw")]["accuracy"])])
    result["retention"] = (["Model", "Subset PPL", "BoolQ raw", "ARC raw"], retention)
    return result


CAPTIONS = {
    "methods": "Matched GRPO/Dr.~GRPO comparison with no-RL and SFT controls. Greedy accuracy (\\%); sampled metrics use 100 fixed problems and eight responses. RL: LR $10^{-6}$,128 updates. SFT external/sampling entries were not evaluated. MATH tokens are mean greedy completion length.",
    "rates": "One-factor learning-rate sweep: dev 300 accuracy (\\%),128 updates, one fixed reference run per condition. Dashes denote unrun settings, not zero accuracy. Other settings remain fixed within each method.",
    "regularization": "GRPO regularization at LR $10^{-6}$, one fixed reference run per condition. Prefix KL is an offline shared-context diagnostic, not training KL. Clip fraction is a fraction, not a percentage.",
    "rewards": "Reward design at LR $3\\cdot 10^{-6}$: dev 300 accuracy (\\%) and mean completion tokens, using the fixed reference run per condition rather than cross-run averages. Reward-score scales differ.",
    "exploration": "Training exploration at LR $3\\cdot 10^{-6}$, one reference run per condition. $G$ is training group size and $T$ training temperature. Evaluations use common settings; group-size arms match unique problems, not updates or token cost.",
    "duration": "Longer GRPO training on the same 512 problems at LR $3\\cdot 10^{-6}$. Greedy 100 and sampled metrics use the same problems; train tokens are cumulative. Only measured checkpoints are tabulated; all accuracy metrics are percentages.",
    "retention": "Auxiliary retention diagnostic for original LR $10^{-6}$ models. WikiText subset perplexity uses 32 fixed 512-token blocks; raw BoolQ/ARC label-likelihood evaluations use 100 questions each. Not a claim of broad capability preservation.",
}


def plot(data, study, destination):
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.dpi": 220, "xtick.labelsize": 7, "ytick.labelsize": 7})
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.3), constrained_layout=True)
    styles = {"grpo": ("#0072B2", "-"), "dr_grpo": ("#D55E00", "--")}
    specifications = (("reward", "Training correctness", 4), ("loss", "Policy surrogate loss", 8),
                      ("entropy", "Token entropy", 8), ("completions/mean_length", "Response length", 4))
    for name, (color, style) in styles.items():
        history = json.loads((study / "pilot_20261013_000313/pilot" / name / "train_history.json").read_text())["log_history"]
        for axis, (metric, title, window) in zip(axes.flat, specifications):
            points = initial_training_points(history, metric)
            xs, ys = zip(*points)
            axis.plot(xs, trailing_mean(ys, window), color=color, linestyle=style, linewidth=1.4, label=LABELS[name].replace("~", " "))
            axis.set(title=title, xlabel="Optimizer update", xlim=(0, 128))
    for axis in axes.flat:
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        axis.legend(frameon=False, fontsize=6.5)
    for suffix in ("pdf", "png"):
        figure.savefig(destination / f"method_dynamics.{suffix}")
    plt.close(figure)
    history = data["extension"]["training_history"]
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.3), constrained_layout=True)
    for axis, metric, title, window in zip(axes.flat, ("reward", "loss", "entropy", "completions/mean_length"),
                                         ("Training correctness", "GRPO policy loss", "Training token entropy", "Training response length"), (4, 16, 16, 4)):
        points = training_curve(history, metric)
        xs, ys = zip(*points)
        if metric == "loss":
            axis.plot(xs, ys, color="#BBBBBB", linewidth=0.4, label="Raw loss")
            axis.axhline(0, color="#888888", linewidth=0.5)
        axis.plot(xs, trailing_mean(ys, window), color="#0072B2", linewidth=1.1, label=f"Mean ({window} observations)")
        axis.set(title=title, xlabel="Optimizer update", xticks=[0, 128, 256, 384, 512], xlim=(0, 512))
        axis.legend(frameon=False, fontsize=6)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    for suffix in ("pdf", "png"):
        figure.savefig(destination / f"extended_training.{suffix}")
    plt.close(figure)


def plot_inference(summary, destination):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
    specs = (("sampled", "pass_at_1", "(a) Sampled pass@1"), ("greedy", "first_accuracy", "(b) Greedy token budget"), ("sampled", "selected_accuracy", "(c) Consensus selection"))
    for axis, (mode, metric, title) in zip(axes, specs):
        key = "temperature" if mode == "sampled" else "cap"
        for model, label, color in (("base", "No RL", "#747474"), ("step128", "128", "#0072B2"), ("step256", "256", "#D55E00"), ("step512", "512", "#218558")):
            rows = sorted((r for r in summary["runs"] if r["checkpoint"] == model and r["mode"] == mode), key=lambda r: r[key])
            axis.plot([r[key] for r in rows], [100 * r[metric] for r in rows], marker="o", markersize=2.5, color=color, label=label)
        axis.set(title=title, xlabel="Temperature" if mode == "sampled" else "Completion cap", ylabel="Accuracy (%)", xticks=[0.4, 0.7, 1.0] if mode == "sampled" else [512, 1024, 2048, 3000])
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        if mode == "greedy":
            axis.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    for suffix in ("pdf", "png"):
        figure.savefig(destination / f"inference_ablations.{suffix}")
    plt.close(figure)


def main():
    study = output_dir("rl_1p5b_study")
    destination = project_root() / "docs/final"
    generated, figures = destination / "generated", destination / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    data, sources = collect(study)
    tables = generate_tables(data, study)
    for name, (headers, rows) in tables.items():
        (generated / f"{name}.tex").write_text(table(headers, rows, CAPTIONS[name], name, name in ("methods", "duration", "rates", "exploration")))
    if data["latest"]:
        rows = [[r["benchmark"], r["model"].replace("_", "\\_"), str(r["n"]), pct(r["accuracy"]), f"{r['avg_generated_tokens']:.0f}", f"{r['avg_runtime']:.1f}", str(r["truncated"])] for r in data["latest"]["runs"]]
        (generated / "latest_benchmarks.tex").write_text(table(["Benchmark", "Checkpoint", "n", "Accuracy", "Tokens", "Seconds", "Truncated"], rows, "Completed greedy external evaluation at LR $3\\cdot10^{-6}$, 3000-token cap; step 256 reconstructed with matching rollout and training-metric prefixes. Runtime is observed mean per problem, not a hardware-controlled speed benchmark.", "latest", True))
    else:
        (generated / "latest_benchmarks.tex").write_text("\\paragraph{Pending checkpoint benchmarks.} External evaluations of the LR $3\\cdot10^{-6}$ checkpoints at128,256 and512 updates are queued. No results from those jobs are reported in this draft. The rotated256 checkpoint is being reconstructed with rollout-prefix and training-metric checks before evaluation.\n")
    inference = load_inference_evidence(project_root() / "results/checkpoint_ablations_20261113", sources)
    if inference:
        extra = inference_tables(inference["runs"])
        captions = {"inference_budget": "Greedy answer accuracy (\\%) on the same 100 development problems, varying only the completion cap. These are diagnostic100 scores, not dev300 or external benchmark accuracy.",
                    "inference_temperature": "Completed inference-only ablations on 100 fixed development problems: eight responses per problem, cap3000. Accuracy, majority rate and pass@8 are percentages. Group tokens count all eight completions, including fallback cases; no verifier or oracle selection."}
        for name, (headers, rows) in extra.items():
            (generated / f"{name}.tex").write_text(table(headers, rows, captions[name], name, name == "inference_temperature"))
        tables.update(extra)
    from scripts.paper.prepare_followup import load_export
    followup = load_export(generated / "followup_evidence.json")
    sources.update(followup["sources"])
    followup_path = generated / "followup_evidence.json"
    sources[str(followup_path)] = {"path": str(followup_path), "sha256": hashlib.sha256(followup_path.read_bytes()).hexdigest()}
    captions = {"prompt_sensitivity": "System-instruction sensitivity: greedy accuracy (\\%) on the same 100 development questions, cap3000. Only wording changes; original controls are reused.",
                "selection_rules": "Retrospective development selection at $T=0.4$, eight candidates. Accuracy (\\%) averages uniformly over candidate order, including winning-group representatives and fallback/ties; it is not the original fixed-order consensus result. Oracle@8 is a scoring upper bound, never a selector.",
                "external_selection": "Complete external consensus conditions only: frozen $T=0.7$, eight responses, cap3000, original prompts, fixed-order strict majority with first-candidate fallback. Tokens $\\times$ is total completion-token cost relative to each policy's greedy control. The incomplete step512 MATH-500 consensus condition is excluded.",
                "candidate_budget": "Candidate-budget control: checkpoint256, $T=0.4$, same 100 development questions. Accuracy (\\%) averages all subsets and uniform candidate order. No extra model generation or independent replications are implied.",
                "answer_diversity": "Mean normalized final-answer entropy in nats on the fixed 100 development questions. This is answer diversity, not token-policy entropy; temperatures are discrete measured conditions."}
    extra = followup_tables(followup)
    for name, (headers, rows) in extra.items():
        (generated / f"{name}.tex").write_text(table(headers, rows, captions[name], name))
    tables.update(extra)
    (generated / "evidence.json").write_text(json.dumps({"audit_date": datetime.now(timezone.utc).date().isoformat(), "audit_timezone": "UTC", "sources": sources, "tables": tables,
        "latest_benchmarks_complete": data["latest"] is not None, "inference_ablations_complete": inference is not None,
        "followup_complete_conditions": len(followup["completed"]), "followup_excluded_conditions": followup["excluded_conditions"],
        "publication_methods": list(PUBLICATION_METHODS), "publication_reference_seed": PUBLICATION_REFERENCE_SEED,
        "publication_figures": list(REPORT_FIGURES), "hyperparameter_presentation": "tables only",
        "scope": "Focused GRPO/Dr.GRPO publication; baseline/SFT controls. Historical studies preserved outside this presentation; one fixed reference run per ablation, no cross-run averaging."}, indent=2) + "\n")
    plot(data, study, figures)
    print(f"Prepared {len(tables) + bool(data['latest'])} verified tables, 2 dense training figures and provenance in {destination}")


if __name__ == "__main__":
    main()
