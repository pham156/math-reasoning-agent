"""Inference-only checkpoint ablations on the frozen 100-problem development cohort."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from src.evaluation.answer import normalize_answer
from src.evaluation.pilot_exploration import pass_at_k


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def selection(candidates):
    """Vote without correctness labels; empty answers cannot win a vote."""
    if not candidates:
        raise ValueError("Cannot select from an empty candidate group")
    groups = {}
    for index, candidate in enumerate(candidates):
        answer = normalize_answer(candidate["answer"])
        if answer:
            groups.setdefault(answer, []).append(index)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[1][0]))
    majority = bool(ranked and len(ranked[0][1]) > len(candidates) / 2)
    return {"selected_index": ranked[0][1][0] if majority else 0,
            "majority": majority, "fallback": not majority,
            "answer_groups": [{"answer": answer, "indices": indices} for answer, indices in ranked],
            "rule": "strict majority of all candidates; first candidate otherwise; no verifier"}


def load_condition(condition, cohort=None, n=100):
    rows, hashes = [], {}
    samples = 8 if condition["mode"] == "sampled" else 1
    for shard in condition["shards"]:
        folder = Path(condition["root"]) / condition["mode"] / condition["name"]
        config_path = folder / f"shard_{shard}_config.json"
        path = folder / f"shard_{shard}.jsonl"
        config = read(config_path)
        part = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        expected = {"mode": condition["mode"], "name": condition["name"], "shard": shard,
                    "max_new_tokens": condition["cap"], "temperature": condition["temperature"],
                    "num_samples": samples, "sampling_batch": samples, "top_p": 1.0, "top_k": 0,
                    "stop_ids": [151643, 151645], "precision": "FP32 parameters/BF16 compute",
                    "grader": "pilot exact/Math-Verify", "seed_rule": "100000 + source_index for each sampled group"}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Changed decoding/grading protocol: {condition['key']}")
        if Path(config["model_path"]).resolve() != Path(condition["model_path"]).resolve():
            raise ValueError("Wrong checkpoint in evaluation config")
        if ([r["source_index"] for r in part] != config["source_indices"] or
                [r["prompt_hash"] for r in part] != config["prompt_hashes"]):
            raise ValueError("Incomplete shard or changed prompt hashes")
        for row in part:
            candidates = row["candidates"]
            if len(candidates) != samples or [c["sample_index"] for c in candidates] != list(range(samples)):
                raise ValueError("Incomplete or reordered candidate group")
            for candidate in candidates:
                if (type(candidate["correct"]) is not bool or type(candidate["truncated"]) is not bool or
                        not 0 < candidate["generated_tokens"] <= condition["cap"] or
                        candidate["truncated"] != (candidate["generated_tokens"] == condition["cap"] and candidate["ending_token_id"] not in config["stop_ids"])):
                    raise ValueError("Invalid candidate grading/token/truncation record")
            if row["num_correct"] != sum(c["correct"] for c in candidates):
                raise ValueError("Recorded correctness count changed")
        hashes[str(path)] = digest(path)
        hashes[str(config_path)] = digest(config_path)
        rows.extend(part)
    if len(rows) != n or len({r["source_index"] for r in rows}) != n:
        raise ValueError(f"Incomplete/duplicate cohort: {condition['key']}")
    if cohort is not None:
        keys = ("source_index", "prompt_hash", "problem", "reference_answer", "type", "level")
        if [{k: r[k] for k in keys} for r in rows] != cohort:
            raise ValueError("Different development problems, references, or prompts")
    return rows, hashes


def metrics(rows):
    count = len(rows)
    candidates = [c for row in rows for c in row["candidates"]]
    votes = [selection(row["candidates"]) for row in rows]
    selected = [row["candidates"][vote["selected_index"]]["correct"] for row, vote in zip(rows, votes)]
    majority_correct = [correct for correct, vote in zip(selected, votes) if vote["majority"]]
    samples = len(rows[0]["candidates"])
    first = [r["candidates"][0]["correct"] for r in rows]
    passes = {str(k): sum(pass_at_k(samples, r["num_correct"], k) for r in rows) / count for k in (1, 2, 4, 8) if k <= samples}
    return {"n": count, "samples_per_problem": samples,
            "first_accuracy": sum(first) / count, "selected_accuracy": sum(selected) / count,
            "pass_at_k": passes, "pass_at_1": passes["1"], "pass_at_4": passes.get("4"), "pass_at_8": passes.get("8"),
            "majority_rate": len(majority_correct) / count,
            "majority_accuracy": sum(majority_correct) / len(majority_correct) if majority_correct else None,
            "fixed_by_consensus": sum(b and not a for a, b in zip(first, selected)),
            "broken_by_consensus": sum(a and not b for a, b in zip(first, selected)),
            "missed_correct": sum(r["num_correct"] > 0 and not s for r, s in zip(rows, selected)),
            "avg_candidate_tokens": sum(c["generated_tokens"] for c in candidates) / len(candidates),
            "avg_problem_tokens": sum(c["generated_tokens"] for c in candidates) / count,
            "total_generated_tokens": sum(c["generated_tokens"] for c in candidates),
            "truncation_rate": sum(c["truncated"] for c in candidates) / len(candidates),
            "avg_problem_runtime": sum(r["runtime_seconds"] for r in rows) / count}


def verify_sources(manifest):
    for path, expected in manifest["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Campaign code/evidence changed: {path}")
    for model, expected in manifest["model_files"].items():
        for path, fingerprint in expected.items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != fingerprint:
                raise ValueError(f"Checkpoint file changed: {model}: {path}")


def summarize(campaign, existing=False, plots=False):
    manifest = read(campaign / "campaign.json")
    verify_sources(manifest)
    if not existing and not read(campaign / "smoke_passed.json")["passed"]:
        raise ValueError("GPU smoke validation has not passed")
    results, evidence = [], {}
    details = Path(manifest["raw_root"]) / ("existing_selection.jsonl" if existing else "selection.jsonl")
    details.parent.mkdir(parents=True, exist_ok=True)
    with details.open("w") as stream:
        for condition in manifest["conditions"]:
            if existing and not condition["reused"]:
                continue
            rows, hashes = load_condition(condition, manifest["cohort"])
            evidence.update(hashes)
            results.append({**{k: condition[k] for k in ("key", "checkpoint", "mode", "temperature", "cap", "reused")}, **metrics(rows)})
            if condition["mode"] == "sampled":
                for row in rows:
                    vote = selection(row["candidates"])
                    stream.write(json.dumps({"condition": condition["key"], "source_index": row["source_index"],
                        **vote, "selected_answer": row["candidates"][vote["selected_index"]]["answer"],
                        "selected_correct": row["candidates"][vote["selected_index"]]["correct"],
                        "oracle_available": row["num_correct"] > 0,
                        "candidate_answers": [c["answer"] for c in row["candidates"]],
                        "candidate_correct": [c["correct"] for c in row["candidates"]]}) + "\n")
    prefix = "existing_summary" if existing else "summary"
    payload = {"runs": results, "evidence_sha256": evidence,
               "note": "Frozen 100-problem development set; no training or external-test selection. Sampled pass@1 is not greedy. Strict majority (>4/8), first-candidate fallback, first member of winning group; no verifier or ground-truth-based selection. Answer grouping uses the existing textual normalizer, not full mathematical equivalence. Recorded runtime reflects generating all candidates, not standalone one-sample latency. Reused rotated256 outputs are labeled and backed by replay proof, not original weight hashes."}
    write(campaign / f"{prefix}.json", payload)
    fields = ["checkpoint", "mode", "temperature", "cap", "first_accuracy", "pass_at_1", "pass_at_4", "pass_at_8", "selected_accuracy", "majority_rate", "avg_candidate_tokens", "avg_problem_tokens", "truncation_rate", "avg_problem_runtime", "total_generated_tokens"]
    with (campaign / f"{prefix}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Checkpoint Inference Ablations", "", payload["note"], "",
             "| Checkpoint | Mode | T | Cap | First | Pass@1 | Consensus | Pass@8 | Majority | Mean candidate tokens | Truncated |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        p8 = f"{r['pass_at_k']['8']:.1%}" if "8" in r["pass_at_k"] else "--"
        lines.append(f"| {r['checkpoint']} | {r['mode']} | {r['temperature']} | {r['cap']} | {r['first_accuracy']:.1%} | {r['pass_at_1']:.1%} | {r['selected_accuracy']:.1%} | {p8} | {r['majority_rate']:.1%} | {r['avg_candidate_tokens']:.1f} | {r['truncation_rate']:.1%} |")
    (campaign / f"{prefix}.md").write_text("\n".join(lines) + "\n")
    if plots:
        plot_results(results, campaign)
    print("\n".join(lines))


def plot_results(results, campaign):
    from src.core.config import cache_dir
    os.environ["MPLCONFIGDIR"] = str(cache_dir() / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for checkpoint in ("base", "step128", "step256", "step512"):
        sampled = sorted((r for r in results if r["checkpoint"] == checkpoint and r["mode"] == "sampled"), key=lambda r: r["temperature"])
        greedy = sorted((r for r in results if r["checkpoint"] == checkpoint and r["mode"] == "greedy"), key=lambda r: r["cap"])
        axes[0].plot([r["temperature"] for r in sampled], [100 * r["pass_at_k"]["1"] for r in sampled], marker="o", label=checkpoint)
        axes[1].plot([r["cap"] for r in greedy], [100 * r["first_accuracy"] for r in greedy], marker="o", label=checkpoint)
        axes[2].plot([r["temperature"] for r in sampled], [100 * r["selected_accuracy"] for r in sampled], marker="o", label=checkpoint)
    for axis, title, xlabel in zip(axes, ("Sampled pass@1", "Greedy budget sensitivity", "Consensus + first fallback"), ("Temperature", "Completion cap (tokens)", "Temperature")):
        axis.set(title=title, xlabel=xlabel, ylabel="Accuracy (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    destination = campaign / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"inference_ablations.{suffix}", dpi=180)
    plt.close(figure)


def validate_smoke(campaign):
    manifest = read(campaign / "campaign.json")
    verify_sources(manifest)
    evidence = {}
    for condition in manifest["smoke_conditions"]:
        _, hashes = load_condition(condition, manifest["cohort"][:2], n=2)
        evidence.update(hashes)
    # Exercise final aggregation on GPU-generated artifacts before releasing the grid.
    smoke_rows, _ = load_condition(manifest["smoke_conditions"][0], manifest["cohort"][:2], n=2)
    summary = metrics(smoke_rows)
    if summary["n"] != 2 or summary["samples_per_problem"] != 8:
        raise ValueError("Smoke final evaluation failed")
    write(campaign / "smoke_passed.json", {"passed": True, "evidence_sha256": evidence, "sampled_metrics": summary})


def run_task(campaign, task):
    import subprocess
    manifest = read(campaign / "campaign.json")
    verify_sources(manifest)
    smoke = task == "smoke"
    if not smoke and not read(campaign / "smoke_passed.json")["passed"]:
        raise ValueError("Smoke gate missing")
    entries = manifest["smoke_conditions"] if smoke else [manifest["conditions"][manifest["tasks"][int(task)]["condition"]]]
    for condition in entries:
        shards = [0] if smoke else [manifest["tasks"][int(task)]["shard"]]
        for shard in shards:
            command = ["bash", "scripts/rl/smoke_python.sh", "-m", "src.evaluation.pilot_exploration",
                       "--pilot_root", manifest["pilot_root"], "--comparison_root", manifest["comparison_root"],
                       "--output_dir", condition["root"], "--model_path", condition["model_path"],
                       "--mode", condition["mode"], "--name", condition["name"], "--shard", str(shard),
                       "--temperature", str(condition["temperature"]), "--max_new_tokens", str(condition["cap"]),
                       "--require_complete"]
            if smoke:
                command.extend(["--limit", "2"])
            subprocess.run(command, check=True)
    if smoke:
        validate_smoke(campaign)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--phase", choices=("check", "smoke", "task", "existing", "summary"), required=True)
    parser.add_argument("--task", type=int)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.phase == "check":
        verify_sources(read(args.campaign / "campaign.json"))
    elif args.phase in ("smoke", "task"):
        run_task(args.campaign, "smoke" if args.phase == "smoke" else args.task)
    else:
        summarize(args.campaign, existing=args.phase == "existing", plots=args.plots)


if __name__ == "__main__":
    main()
