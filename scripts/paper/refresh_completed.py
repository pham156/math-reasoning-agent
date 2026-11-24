"""Audit completed evaluation cohorts without resubmitting jobs or scoring partial shards."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from src.core.config import cache_dir, ensure_scratch_path, output_dir, project_root
from src.evaluation.pilot_benchmarks import summarize_rows


def read(path, sources):
    path = Path(path)
    content = path.read_bytes()
    sources[str(path)] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    return json.loads(content)


def lines(path, sources):
    path = Path(path)
    content = path.read_bytes()
    sources[str(path)] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def assess(directory, model, reference, size, sources, benchmark_hash=None):
    """Validate every available prefix; publish a score only for the entire expected cohort."""
    directory = Path(directory)
    if len(reference) % size or not reference:
        raise ValueError("Reference cohort does not fit its shard size")
    rows, missing = [], []
    for shard, start in enumerate(range(0, len(reference), size)):
        config_path = directory / f"shard_{shard}_config.json"
        path = directory / f"shard_{shard}.jsonl"
        if not config_path.exists() or not path.exists():
            missing.append(shard)
            continue
        config = read(config_path, sources)
        expected = reference[start:start + size]
        ids = [r["source_index"] for r in expected]
        if (Path(config["model_path"]).resolve() != Path(model).resolve() or
                config["source_indices"] != ids or config["mode"] != "greedy" or
                config["num_samples"] != 1 or config["temperature"] != 0 or
                config["top_p"] != 1 or config["top_k"] != 0 or
                config["max_new_tokens"] != 3000 or set(config["stop_ids"]) != {151643, 151645}):
            raise ValueError(f"Changed evaluation protocol: {config_path}")
        if benchmark_hash and config.get("benchmark_sha256") != benchmark_hash:
            raise ValueError(f"Changed frozen benchmark: {config_path}")
        if len(config["prompt_hashes"]) != size:
            raise ValueError(f"Invalid prompt hashes: {config_path}")
        part = lines(path, sources)
        if len(part) > size or [r["source_index"] for r in part] != ids[:len(part)]:
            raise ValueError(f"Duplicate or changed cohort: {path}")
        for index, row in enumerate(part):
            source = expected[index]
            if (row["problem"] != source["problem"] or
                    str(row["reference_answer"]) != str(source.get("answer", source.get("reference_answer"))) or
                    row["prompt_hash"] != config["prompt_hashes"][index] or
                    ("prompt_hash" in source and row["prompt_hash"] != source["prompt_hash"]) or
                    len(row["candidates"]) != 1 or
                    type(row["candidates"][0]["correct"]) is not bool):
                raise ValueError(f"Changed problem, prompt or candidate: {path}/{index}")
        if len(part) < size:
            missing.append(shard)
        rows.extend(part)
    status = {"status": "incomplete" if missing else "complete", "observed": len(rows),
              "expected": len(reference), "missing_or_partial_shards": missing}
    return status, summarize_rows(rows) if not missing else None


def sole_model(directory):
    paths = list(Path(directory).glob("greedy/*"))
    if not paths:
        return Path(directory) / "greedy/missing"
    if len(paths) != 1:
        raise ValueError(f"Ambiguous model directories: {directory}")
    return paths[0]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def refresh(study, destination):
    sources = {}
    control = study / "controls_exploration_20261014/greedy/baseline"
    dev = [r for shard in range(3) for r in lines(control / f"shard_{shard}.jsonl", sources)]
    if len(dev) != 300 or len({r["source_index"] for r in dev}) != 300:
        raise ValueError("Invalid development reference")
    sft_root = study / "sft_study_20261115"
    manifest = read(sft_root / "submission.json", sources)
    sft, statuses = [], {}
    for arm in manifest["arms"]:
        entry = {key: arm[key] for key in ("label", "description", "model_path")}
        config = sft_root / "arms" / arm["label"] / "sft_config.json"
        if config.exists():
            entry["sft_config"] = read(config, sources)
        for benchmark in ("dev300", *manifest["benchmarks"]):
            frozen = None if benchmark == "dev300" else Path(manifest["benchmarks"][benchmark])
            reference = dev if frozen is None else read(frozen, sources)["rows"]
            size = 100 if frozen is None else read(frozen, sources)["metadata"].get("greedy_shard_size", 100)
            folder = sole_model(sft_root / "evaluations" / arm["label"] / benchmark)
            status, metrics = assess(folder, arm["model_path"], reference, size, sources,
                                     sources[str(frozen)]["sha256"] if frozen else None)
            statuses[f"sft/{arm['label']}/{benchmark}"] = status
            if metrics is not None:
                entry[benchmark] = metrics
        sft.append(entry)
    sft_summary = {"arms": sft, "status": "partial", "cohorts": statuses,
                   "note": "Complete cohorts only; missing benchmark cells are not zero accuracy. Historical stored grader, not strict regrading."}
    write(sft_root / "summary_partial.json", sft_summary)

    from src.evaluation.dense_curve import anchors
    for relative in ("controls_exploration_20261014/summary.json", "grpo_extended_20261105/summary_512.json",
                     "checkpoint_benchmarks_20261106/summary.json"):
        read(study / relative, sources)
    curve = {str(step): {**values, "source": "existing"} for step, values in anchors().items()}
    root = study / "dense_replay_20261115"
    manifest = read(root / "submission.json", sources)
    dense_status = {}
    for step in manifest["eval_steps"]:
        export = root / "exports" / f"step-{step}"
        receipt_path = export / "validation.json"
        if not receipt_path.exists():
            dense_status[str(step)] = {"status": "awaiting replay validation"}
            continue
        receipt = read(receipt_path, sources)
        if receipt["step"] != step or not receipt["rollouts_identical"] or not receipt["training_metrics_matched"]:
            raise ValueError(f"Invalid replay proof: {receipt_path}")
        point = {"source": "validated dense replay"}
        for benchmark in ("dev300", "gsm8k100", "math500"):
            if benchmark == "math500" and step not in manifest["math500_steps"]:
                continue
            frozen = None if benchmark == "dev300" else Path(manifest["benchmarks"][benchmark])
            reference = dev if frozen is None else read(frozen, sources)["rows"]
            folder = root / "evaluations" / f"step-{step}" / benchmark / "greedy/grpo"
            status, metrics = assess(folder, export, reference, 100, sources,
                                     sources[str(frozen)]["sha256"] if frozen else None)
            dense_status[f"{step}/{benchmark}"] = status
            if metrics is not None:
                point[benchmark] = metrics["accuracy"]
        if len(point) > 1:
            curve[str(step)] = point
    dense_summary = {"curve": curve, "status": "partial", "cohorts": dense_status,
                     "note": "Only complete validated cohorts; no interpolation across missing scheduled checkpoints."}
    write(root / "curve_partial.json", dense_summary)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "sft": sft_summary, "dense": dense_summary, "sources": sources}
    write(destination / "completed_evidence.json", payload)
    return payload


def measured_segments(curve, key, spacing):
    """Break lines at unmeasured scheduled checkpoints rather than filling the gap."""
    points = sorted((int(step), values[key]) for step, values in curve.items() if key in values)
    segments = []
    for point in points:
        if not segments or point[0] - segments[-1][-1][0] != spacing:
            segments.append([])
        segments[-1].append(point)
    return segments


def plot(payload, destination):
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for key, label, spacing, color, ax in (
            ("dev300", "Dev300", 16, "#0072B2", axes[0]),
            ("math500", "MATH-500", 64, "#CC79A7", axes[1]),
            ("gsm8k100", "GSM8K-100", 16, "#009E73", axes[1])):
        for index, segment in enumerate(measured_segments(payload["dense"]["curve"], key, spacing)):
            xs, ys = zip(*segment)
            ax.plot(xs, [100 * y for y in ys], "o-", color=color, markersize=4,
                    label=label if index == 0 else None)
    for ax, title in zip(axes, ("Development", "External benchmarks")):
        ax.set(title=title, xlabel="Optimizer update", ylabel="Greedy accuracy (%)", xlim=(-8, 520), ylim=(20, 95))
        ax.set_xticks(range(0, 513, 64))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    figure.suptitle("GRPO held-out evaluation: completed checkpoints")
    figure.supxlabel("Gaps are unevaluated checkpoints. Same prompt, common stop, 3000-token cap; one reference run.", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"heldout_accuracy.{suffix}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()
    destination = project_root() / "figures/completed_evaluations"
    destination.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        payload = json.loads((destination / "completed_evidence.json").read_text())
        for item in payload["sources"].values():
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Source changed since audit: {item['path']}")
        plot(payload, destination)
    else:
        payload = refresh(ensure_scratch_path(output_dir("rl_1p5b_study")), destination)
        print(json.dumps({"measured_checkpoints": len(payload["dense"]["curve"]),
                          "complete_sft_cohorts": sum(s["status"] == "complete" for s in payload["sft"]["cohorts"].values()),
                          "destination": str(destination)}))


if __name__ == "__main__":
    main()
