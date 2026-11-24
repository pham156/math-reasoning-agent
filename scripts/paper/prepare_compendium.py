"""Create an organized LaTeX ledger of available, completed mathematical-RL results."""

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import fmean, pstdev
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.paper.prepare_final import collect, load_inference_evidence
from scripts.paper.prepare_followup import load_export
from src.core.config import data_dir, output_dir, project_root


def escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_\allowbreak{}", "/": r"/\allowbreak{}", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def number(value):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite result")
    return f"{value:.6g}"


def pct(value):
    return "--" if value is None else f"{100 * value:.2f}"


def longtable(headers, rows, widths=None):
    widths = widths or [1 / len(headers)] * len(headers)
    # Account for intercolumn padding; paragraph columns wrap rather than overflowing.
    available = min(.90, .99 - .013 * len(headers))
    columns = "".join(r">{\raggedright\arraybackslash}p{" + f"{available * w:.4f}" + r"\linewidth}" for w in widths)
    head = " & ".join(escape(h) for h in headers) + r" \\"
    return (r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{longtable}{" + columns + "}\n"
            + r"\toprule " + head + "\n" + r"\midrule\endfirsthead" + "\n"
            + r"\toprule " + head + "\n" + r"\midrule\endhead" + "\n"
            + r"\midrule\multicolumn{" + str(len(headers)) + r"}{r}{Continued on next page}\endfoot" + "\n"
            + r"\bottomrule\endlastfoot" + "\n"
            + "\n".join(" & ".join(escape(c) for c in row) + r" \\" for row in rows)
            + "\n" + r"\end{longtable}\endgroup" + "\n")


SKIP_DETAILS = {"training_history", "curve_metrics", "diversity", "arguments", "model_path", "checkpoint",
                "final_model", "evaluation_root", "training_job_ids", "training_jobs", "slurm_job_id",
                "slurm_array_task_id", "versions", "gpu", "sources", "prompt", "raw_solution"}


def numeric_details(value, prefix=""):
    """Scalar aggregates only; raw histories and per-problem IDs stay in source evidence."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in SKIP_DETAILS:
                yield from numeric_details(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            if "ci" in prefix.lower() and len(value) == 2:
                yield prefix, "[" + ", ".join(number(v) for v in value) + "]"
            else:
                yield prefix + ".count", str(len(value))
        elif value and all(isinstance(v, dict) for v in value):
            for index, item in enumerate(value):
                yield from numeric_details(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)):
        yield prefix, number(value)
    elif isinstance(value, str) and prefix.rsplit(".", 1)[-1] in {
            "task", "format", "algorithm", "mode", "rule", "reward_design", "loss_type",
            "scale_rewards", "importance_sampling_level", "data", "lr_scheduler"}:
        yield prefix, value


class Ledger:
    def __init__(self):
        self.records = []
        self.sources = {}
        self.setup = {}

    def read(self, path):
        path = Path(path)
        content = path.read_bytes()
        self.sources[str(path)] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
        return json.loads(content)

    def add(self, section, label, cohort, metrics, source, expected=None):
        if expected is not None and metrics.get("n") != expected:
            raise ValueError(f"Incomplete cohort: {label}/{cohort}")
        if metrics.get("status") in {"failed", "incomplete", "paused"}:
            raise ValueError(f"Refuse unsuccessful condition: {label}")
        identity = (section, label, cohort)
        if any(tuple(r[k] for k in ("section", "label", "cohort")) == identity for r in self.records):
            raise ValueError(f"Duplicate ledger row: {identity}")
        self.records.append({"id": f"R{len(self.records) + 1:04d}", "section": section,
                             "label": label, "cohort": cohort, "metrics": metrics, "source": str(source)})


def reference_only(runs):
    return [r for r in runs if r.get("settings", {}).get("seed", 0) == 0]


def audit_sample_count(path):
    if "smoke" in path or "rft_data" in path or "seed1" in path:
        return None
    sampled = "/sampled/" in path
    if "/aime/" in path:
        return 480 if sampled else 60
    if "math500" in path:
        return 4000 if sampled else 500
    if "gsm8k100" in path:
        return 800 if sampled else 100
    if "checkpoint_ablations" in path or "checkpoint_followup" in path:
        return 800 if sampled else 100
    if "/development/step192/" in path:
        return 100
    return 800 if sampled else 300


def summarize_training(history, through):
    """Summarize actual logged updates, without zero-filling absent metrics."""
    result = {}
    for key in ("loss", "reward", "entropy", "grad_norm", "completions/mean_length", "kl"):
        points = {}
        for row in history:
            step, value = row.get("step", 0), row.get(key)
            if 0 < step <= through and isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {key} at update {step}")
                if step in points and points[step] != value:
                    raise ValueError(f"Conflicting {key} at update {step}")
                points[step] = value
        if points:
            values = [value for _, value in sorted(points.items())]
            window = max(1, len(values) // 4)
            result[key] = {"mean": fmean(values), "std": pstdev(values), "first_quarter": fmean(values[:window]),
                           "last_quarter": fmean(values[-window:]), "observations": len(values)}
    return result


def gather_setup(root, ledger):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    run = ledger.read(root / "pilot_20261013_000313/pilot/grpo/run_config.json")
    split = ledger.read(Path(run["arguments"]["split_dir"]) / "manifest.json")
    model = ledger.read(Path(run["arguments"]["model_path"]) / "config.json")
    math_info = ledger.read(Path(split["dataset"]) / "train/dataset_info.json")
    state = ledger.read(data_dir("MATH-500/test/state.json"))
    problems = []
    for item in state["_data_files"]:
        path = data_dir("MATH-500/test") / item["filename"]
        ledger.sources[str(path)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with pa.memory_map(str(path), "r") as stream:
            problems.extend(ipc.open_stream(stream).read_all().column("problem").to_pylist())
    benchmark = {hashlib.sha256(" ".join(p.split()).encode()).hexdigest() for p in problems}
    train = {row["problem_hash"] for row in split["training"]}
    dev = {row["problem_hash"] for row in split["development"]}
    overlaps = {"train_dev": len(train & dev), "train_math500": len(train & benchmark), "dev_math500": len(dev & benchmark)}
    if (len(train), len(dev), len(problems), len(benchmark)) != (512, 300, 500, 500) or any(overlaps.values()):
        raise ValueError("Setup split-integrity check failed")
    if split["source_split"] != "train" or run["precision"]["peft"] or run["precision"]["quantization"]:
        raise ValueError("Reported RL architecture does not match saved run")
    original_sft = ledger.read(root / "dev300_sft_20261013/sft/sft_config.json")
    sft = {"gt_full_original": {k: original_sft[k] for k in ("train_n", "epochs", "learning_rate", "gradient_accumulation_steps", "max_length")}}
    for row in ledger.records:
        if row["section"] == "Supervised and rejection-sampling fine-tuning":
            config = row["metrics"].get("training_configuration", {})
            if config:
                sft[row["label"]] = {k: config[k] for k in ("train_n", "epochs", "learning_rate", "lr_scheduler", "gradient_accumulation_steps", "max_length", "lora") if k in config}
    restart_checks = []
    for method in ("grpo", "dr_grpo", "dapo_style", "gspo"):
        folder = root / "pilot_20261013_000313/resume_gate" / method
        boundary, resumed = [ledger.read(folder / name) for name in ("boundary_check.json", "resume_check.json")]
        for proof in (boundary, resumed):
            if not (proof["passed"] and proof["same_rollouts"] and proof["same_parameter_keys"]
                    and proof["deterministic_training"] and proof["max_parameter_difference"] <= proof["tolerance"]):
                raise ValueError(f"Unpassed restart proof for {method}")
        restart_checks.append({"method": method, "boundary": 4, "target": 8, **resumed})
    extended = ledger.read(root / "grpo_extended_20261105/gate/resume_check.json")
    if not (extended["passed"] and extended["same_rollouts"] and extended["same_parameter_keys"]
            and extended["max_parameter_difference"] <= extended["tolerance"]):
        raise ValueError("Unpassed extended restart proof")
    restart_checks.append({"method": "extended_grpo", "boundary": extended["boundary_step"], "target": extended["target_step"], **extended})
    replay_checks = [ledger.read(root / f"checkpoint_benchmarks_20261106/replay_{step}_check.json") for step in (192, 256)]
    if not all(p["rollouts_identical"] and p["training_metrics_matched"] for p in replay_checks):
        raise ValueError("Replay prefix proof failed")
    return {"model_name": Path(run["arguments"]["model_path"]).name,
            "architecture": {k: model[k] for k in ("architectures", "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "max_position_embeddings", "vocab_size")},
            "precision": run["precision"], "versions": run["versions"],
            "optimizer_batch_responses": run["optimizer_batch_responses"], "generation_batch_responses": run["generation_batch_responses"],
            "prompt_cap": run["arguments"]["max_prompt_length"], "completion_cap": run["arguments"]["max_completion_length"],
            "split_seed": split["split_seed"], "training_seed": run["arguments"]["seed"],
            "eligible": split["audited_eligible"], "math_original_counts": {k: v["num_examples"] for k, v in math_info["splits"].items()},
            "overlaps": overlaps, "sft": sft, "restart_checks": restart_checks, "replay_checks": replay_checks,
            "overlap_method": "SHA256 of whitespace-normalized problem text; not a semantic or pretraining-contamination audit"}


def gather(root):
    ledger = Ledger()
    data, sources = collect(root)
    ledger.sources.update({v["path"]: v for v in sources.values()})
    def source(relative):
        return root / relative
    for row in data["controls"]["greedy_controls"]:
        if row["name"] in ("grpo", "dr_grpo", "dapo_style", "gspo"):
            history = ledger.read(source(f"pilot_20261013_000313/pilot/{row['name']}/train_history.json"))["log_history"]
            row = {**row, "training_diagnostics": summarize_training(history, 128)}
        ledger.add("Initial controls", row["name"], "dev300 greedy, common stop", row, source("controls_exploration_20261014/summary.json"), 300)
    for row in data["controls"]["exploration"]:
        ledger.add("Initial exploration", row["name"], "dev100, eight samples, T=1", row, source("controls_exploration_20261014/summary.json"), 100)
    for row in data["external"]["runs"]:
        ledger.add("Initial external transfer", row["name"], row["benchmark"], row, source("external_benchmarks_20261015/summary.json"), 500 if row["benchmark"] == "math500" else 100)
    for key, relative in (("low", "lr_tuning_20261014/summary.json"), ("high", "grpo_lr_upper_20261016/summary.json")):
        for row in data[key]["runs"]:
            if key == "high" and row["learning_rate"] <= 1e-6:
                continue
            ledger.add("Learning-rate controls", f"{row['algorithm']} LR={row['learning_rate']:g}", "dev300", row, source(relative), 300)
    dr_path = source("drgrpo_lr_20261115/summary.json")
    for row in ledger.read(dr_path)["runs"]:
        if row["learning_rate"] != 1e-6:
            ledger.add("Learning-rate controls", f"dr_grpo LR={row['learning_rate']:g}", "dev300", row, dr_path, 300)
    for row in reference_only(data["tuning"]["runs"]):
        ledger.add("KL and clipping controls", row["label"], "dev300", row, source("grpo_controls_20261017/summary.json"), 300)
    for key, relative, title in (("reward", "reward_exploration_20261024/reward/summary.json", "Reward design"),
                                  ("exploration", "reward_exploration_20261024/exploration/summary.json", "Training exploration")):
        for row in reference_only(data[key]["runs"]):
            label = row["label"].removesuffix("_seed0")
            ledger.add(title, label, "dev300 + sampled dev100", row, source(relative), 300)
    for row in data["extension"]["rows"]:
        row = {**row, "training_diagnostics": summarize_training(data["extension"]["training_history"], row["step"])}
        ledger.add("Training duration", f"GRPO LR=3e-6, update {row['step']}", "dev300 + sampled dev100", row, source("grpo_extended_20261105/summary_512.json"), 300)
    for row in data["latest"]["runs"]:
        ledger.add("Checkpoint transfer", row["model"], row["benchmark"], row, source("checkpoint_benchmarks_20261106/summary.json"), 500 if row["benchmark"] == "math500" else 100)
    audit_path = project_root() / "figures/completed_evaluations/completed_evidence.json"
    audit = ledger.read(audit_path)
    for item in audit["sources"].values():
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Completed-cohort audit source changed: {item['path']}")
    ledger.sources.update(audit["sources"])
    for row in audit["sft"]["arms"]:
        for benchmark in ("dev300", "math500", "gsm8k100", "aime"):
            if benchmark in row:
                metrics = {**row[benchmark], "training_configuration": row.get("sft_config", {})}
                ledger.add("Supervised and rejection-sampling fine-tuning", row["label"], benchmark, metrics, audit_path,
                           {"dev300": 300, "math500": 500, "gsm8k100": 100, "aime": 60}[benchmark])
    external_path = source("drgrpo_lr_20261115/external/summary.json")
    for arm in ledger.read(external_path)["arms"]:
        for benchmark in ("math500", "gsm8k100", "aime"):
            ledger.add("Matched-LR Dr. GRPO transfer", arm["label"], benchmark, arm[benchmark], external_path,
                       {"math500": 500, "gsm8k100": 100, "aime": 60}[benchmark])
    aime_path = source("aime_checkpoints_20261115/summary.json")
    for row in ledger.read(aime_path)["runs"]:
        ledger.add("AIME 2025 and 2026", row["model"], "pooled greedy", row["greedy"], aime_path, 60)
        for year, metrics in row["greedy_by_year"].items():
            ledger.add("AIME 2025 and 2026", row["model"], year + " greedy", metrics, aime_path, 30)
        for cohort, metrics in [("pooled sampled", row["sampled"]), *row["sampled_by_year"].items()]:
            ledger.add("AIME 2025 and 2026", row["model"], cohort + " mean@8/pass@8", metrics, aime_path, 60 if cohort == "pooled sampled" else 30)
    for step, values in sorted(audit["dense"]["curve"].items(), key=lambda x: int(x[0])):
        for benchmark in ("dev300", "math500", "gsm8k100"):
            if benchmark in values:
                ledger.add("Measured held-out trajectory", f"update {step}", benchmark,
                           {"n": {"dev300": 300, "math500": 500, "gsm8k100": 100}[benchmark], "accuracy": values[benchmark]}, audit_path)
    campaign = project_root() / "results/checkpoint_ablations_20261113"
    inference = load_inference_evidence(campaign, ledger.sources)
    if inference is None:
        raise ValueError("Completed inference campaign unavailable")
    for row in inference["runs"]:
        ledger.add("Inference temperature and token budget", f"{row['checkpoint']} {row['mode']} T={row['temperature']:g} cap={row['cap']}", "dev100", row, campaign / "summary.json", 100)
    followup_path = project_root() / "docs/final/generated/followup_evidence.json"
    followup = load_export(followup_path)
    ledger.read(followup_path)
    ledger.sources.update(followup["sources"])
    for row in followup["completed"]:
        ledger.add("Prompt and external consensus controls", row["key"], row["dataset"], row, followup_path, row["n"])
    # A replacement completed after the publication export froze its eleven-condition scope.
    from src.evaluation.checkpoint_followup import control_rows, validate_condition, verify_campaign, verify_smoke
    from src.evaluation.checkpoint_ablations import metrics as inference_metrics
    followup_root = project_root() / "results/checkpoint_followup_20261115"
    manifest_path = followup_root / "campaign.json"
    manifest = ledger.read(manifest_path)
    verify_campaign(manifest)
    verify_smoke(followup_root, manifest)
    condition = next(c for c in manifest["conditions"] if c["key"] == "step512_math500_original_sampled")
    rows, hashes = validate_condition(manifest, condition)
    controls = control_rows(manifest, condition)
    if len(rows) != 500 or any((a["source_index"], a["problem"], a["reference_answer"], a["prompt_hash"]) !=
                               (b["source_index"], b["problem"], b["reference_answer"], b["prompt_hash"])
                               for a, b in zip(rows, controls, strict=True)):
        raise ValueError("Recovered benchmark differs from its full control cohort")
    recovered = inference_metrics(rows)
    recovered["control_accuracy"] = sum(r["candidates"][0]["correct"] for r in controls) / len(controls)
    for path, sha in hashes.items():
        ledger.sources[str(path)] = {"path": str(path), "sha256": sha}
    completed_path = followup_root / "summary.json"
    saved = next(r for r in ledger.read(completed_path)["runs"] if r["key"] == condition["key"])
    if any(saved[key] != recovered[key] for key in ("n", "selected_accuracy", "first_accuracy", "total_generated_tokens")):
        raise ValueError("Recovered summary does not match raw responses")
    ledger.add("Prompt and external consensus controls", condition["key"], "math500, recovered complete cohort", recovered, completed_path, 500)
    for row in followup["saved"]:
        prefix = f"{row['checkpoint']} T={row['temperature']:g}"
        for budget in row["candidate_budgets"]:
            ledger.add("Saved-candidate selection budgets", f"{prefix} N={budget['k']} {budget['rule']}", "dev100 order-averaged", budget, followup_path, 100)
        ledger.add("Answer-distribution diagnostics", prefix, "dev100, eight samples", row["mean_diversity"], followup_path)
        for dimension in ("by_type", "by_level"):
            for group, metrics in row[dimension].items():
                ledger.add("Inference subgroup diagnostics", prefix, dimension + ": " + group, metrics, followup_path)
    for row in data["retention"]["runs"]:
        ledger.add("Non-math retention", row["name"], "WikiText subset; BoolQ/ARC raw+chat", row, source("retention_20261014/summary.json"))
    intervals_path = source("regrade_20261120/paired_intervals.json")
    intervals = ledger.read(intervals_path)
    for label, metrics in intervals["accuracy"].items():
        ledger.add("Historical strict-grading audit", label, "stored vs closed-box grading", metrics, intervals_path)
    for row in intervals["comparisons"]:
        ledger.add("Paired uncertainty estimates", f"{row['model']} vs {row['baseline']}", row["benchmark"], row, intervals_path)
    regrade_path = source("regrade_20261120/regrade_all.json")
    for label, metrics in ledger.read(regrade_path)["groups"].items():
        expected = audit_sample_count(label)
        if expected is not None and metrics["samples"] == expected:
            ledger.add("Response-level grading diagnostics", label, "historical audit; n counts responses",
                       {**metrics, "n": expected, "accuracy": metrics["lenient_accuracy"]}, regrade_path, expected)
    smoke_path = source("smoke/11933114/summary.json")
    smoke = ledger.read(smoke_path)
    if not smoke["all_passed"] or any(r["status"] != "passed" for r in smoke["runs"]):
        raise ValueError("Incomplete smoke study")
    for row in smoke["runs"]:
        ledger.add("Successful setup diagnostics", row["algorithm"], "five-update setup check; not benchmark", row, smoke_path)
    pilot_path = source("pilot_20261013_000313/pilot_summary.json")
    pilot = ledger.read(pilot_path)
    if not pilot["all_completed"]:
        raise ValueError("Incomplete pilot study")
    for row in pilot["runs"]:
        ledger.add("Successful setup diagnostics", row["algorithm"], "initial 20-problem diagnostic", row, pilot_path, 20)
    diagnostic = source("dev300_sft_20261013/termination_diagnostic/summary.json")
    ledger.add("Successful setup diagnostics", "SFT chat-end audit", "five examples; not benchmark", ledger.read(diagnostic), diagnostic, 5)
    ledger.setup = gather_setup(root, ledger)
    return ledger


def generate(ledger, destination, snapshot):
    from scripts.paper.compendium_layout import render

    generated, figures = destination / "generated", destination / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    figure_specs = [(project_root() / "docs/final/figures/method_dynamics.pdf", "initial_training.pdf"),
                    (project_root() / "docs/final/figures/extended_training.pdf", "extended_training.pdf"),
                    (project_root() / "figures/completed_evaluations/heldout_accuracy.pdf", "heldout_accuracy.pdf")]
    for source, filename in figure_specs:
        shutil.copy2(source, figures / filename)
        ledger.sources[str(source)] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    content = render(ledger.records, ledger.setup)
    (generated / "results.tex").write_text(content)
    main = (destination / "main.tex").read_text()
    (destination / "all_results.tex").write_text(main.replace(r"\input{generated/results.tex}", content))
    counts = Counter(r["section"] for r in ledger.records)
    payload = {"snapshot_date": snapshot, "records": ledger.records, "sources": ledger.sources, "setup": ledger.setup,
               "counts": dict(counts), "scope": "Available completed cohorts; fixed-reference runs; no failed/partial cohort scores."}
    (destination / "results_ledger.json").write_text(json.dumps(payload, indent=2) + "\n")
    with ZipFile(destination / "results_overleaf.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in [destination / "main.tex", destination / "all_results.tex", generated / "results.tex",
                     destination / "results_ledger.json", *[figures / name for _, name in figure_specs]]:
            archive.write(path, str(path.relative_to(destination)))
    print(json.dumps({"archived_records": len(ledger.records), "comparison_tables": content.count(r"\caption{"),
                      "figures": len(figure_specs), "destination": str(destination)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot_date", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    generate(gather(output_dir("rl_1p5b_study")), project_root() / "docs/results_compendium", args.snapshot_date.isoformat())


if __name__ == "__main__":
    main()
