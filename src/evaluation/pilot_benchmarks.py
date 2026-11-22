"""Freeze external pilot benchmarks and summarize complete common-stop runs."""

import argparse
from collections import defaultdict
import hashlib
import json
import random
from statistics import mean

from src.core.config import data_dir, ensure_scratch_path
from src.training.grpo.train import write_json


NAMES = ("baseline", "grpo", "dr_grpo", "dapo_style", "gspo")
BENCHMARKS = {"math500": 500, "gsm8k100": 100}
AIME_BENCHMARKS = ("aime2025", "aime2026")
AIME_SHARD_SIZE = 30


def problem_hash(problem):
    return hashlib.sha256(" ".join(problem.split()).encode()).hexdigest()


def prepare(root):
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi
    from src.evaluation.benchmarks import load_math500, load_gsm8k100
    from src.evaluation.answer import extract_final_answer
    from src.training.rl.smoke import conversation

    train = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
    train_hashes = {problem_hash(r["problem"]) for r in train}
    gsm_path = data_dir("GSM8K")
    if not (gsm_path / "dataset_dict.json").exists():
        revision = HfApi().dataset_info("openai/gsm8k").sha
        dataset = load_dataset("openai/gsm8k", "main", revision=revision)
        dataset.save_to_disk(str(gsm_path))
        write_json(root / "gsm8k_download.json", {"dataset_id": "openai/gsm8k", "revision": revision})
    for name, loader in (("math500", load_math500), ("gsm8k100", load_gsm8k100)):
        source, metadata = loader(limit=BENCHMARKS[name], sample_seed=42, sampling="random")
        rows = []
        for row in source:
            answer = row["ground_truth_answer_override"] or extract_final_answer(row["ground_truth_solution"])
            if not answer or problem_hash(row["problem"]) in train_hashes:
                raise ValueError(f"Empty answer or training overlap: {row['problem_id']}")
            rows.append({"source_index": row["problem_index"], "problem_id": row["problem_id"], "problem": row["problem"], "answer": answer, "solution": row["ground_truth_solution"], "type": row["type"], "level": row["level"], "prompt": conversation(row["problem"])})
        if len(rows) != BENCHMARKS[name] or len({problem_hash(r["problem"]) for r in rows}) != len(rows):
            raise ValueError("Benchmark count or uniqueness mismatch")
        metadata.update(training_overlap=0, training_count=len(train), purpose="External transfer evaluation; do not use for tuning", previously_inspected_in_7b_project=True)
        if name == "gsm8k100" and (root / "gsm8k_download.json").exists():
            metadata["dataset_revision"] = json.loads((root / "gsm8k_download.json").read_text())["revision"]
        payload = {"metadata": metadata, "rows": rows}
        path = root / f"{name}_frozen.json"
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError("Refusing to change frozen benchmark")
        write_json(path, payload)
        print(f"Frozen {name}: {len(rows)} problems; no normalized-text training overlap", flush=True)


def summarize_rows(rows):
    n = len(rows)
    return {"n": n, "accuracy": sum(r["candidates"][0]["correct"] for r in rows) / n,
            "truncated": sum(r["candidates"][0]["truncated"] for r in rows),
            "avg_generated_tokens": sum(r["candidates"][0]["generated_tokens"] for r in rows) / n,
            "avg_input_tokens": sum(r["input_tokens"] for r in rows) / n,
            "avg_runtime": sum(r["runtime_seconds"] for r in rows) / n}


def summarize(root):
    results = []
    for benchmark, total in BENCHMARKS.items():
        frozen_path = root / f"{benchmark}_frozen.json"
        frozen = json.loads(frozen_path.read_text())
        expected = [r["source_index"] for r in frozen["rows"]]
        baseline = None
        baseline_hashes = None
        for name in NAMES:
            rows, hashes = [], []
            for shard in range(total // 100):
                directory = root / benchmark / "greedy" / name
                config = json.loads((directory / f"shard_{shard}_config.json").read_text())
                if config["benchmark_sha256"] != hashlib.sha256(frozen_path.read_bytes()).hexdigest():
                    raise ValueError("Benchmark inputs changed")
                part = [json.loads(line) for line in (directory / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip()]
                if [r["source_index"] for r in part] != expected[shard * 100:(shard + 1) * 100]:
                    raise ValueError(f"Incomplete or changed shard: {benchmark}/{name}/{shard}")
                if [r["prompt_hash"] for r in part] != config["prompt_hashes"] or any(len(r["candidates"]) != 1 for r in part):
                    raise ValueError("Prompt or candidate mismatch")
                rows.extend(part)
                hashes.extend(config["prompt_hashes"])
            correct = [r["candidates"][0]["correct"] for r in rows]
            if baseline is None:
                baseline, baseline_hashes = correct, hashes
            if hashes != baseline_hashes:
                raise ValueError("Model prompt mismatch")
            result = {"benchmark": benchmark, "name": name, **summarize_rows(rows),
                      "fixed": [i for i, a, b in zip(expected, baseline, correct) if not a and b],
                      "broken": [i for i, a, b in zip(expected, baseline, correct) if a and not b]}
            for key in ("type", "level"):
                groups = defaultdict(list)
                for row in rows:
                    groups[row[key] or "unspecified"].append(row)
                result[f"by_{key}"] = {k: summarize_rows(v) for k, v in groups.items()}
            results.append(result)
    write_json(root / "summary.json", {"runs": results, "note": "One training seed; previously inspected benchmarks, not untouched confirmation data. Greedy common-stop, max3000, exact/Math-Verify grader."})
    lines = ["# External math benchmark evaluation", "", "One training seed; no retrieval or verifier. These benchmarks were inspected during older 7B experiments and are not untouched confirmation sets.", "", "| Benchmark | Model | n | Accuracy | Truncated | Avg tokens | Avg seconds | Fixes / breaks vs base |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['benchmark']} | {r['name']} | {r['n']} | {r['accuracy']:.1%} | {r['truncated']} | {r['avg_generated_tokens']:.1f} | {r['avg_runtime']:.1f} | {len(r['fixed'])} / {len(r['broken'])} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def freeze_aime_rows(source, benchmark, excluded_hashes):
    """Convert validated AIME rows to the frozen pilot format; reject pilot-split overlap."""
    from src.training.rl.smoke import conversation

    rows = []
    for row in source:
        if problem_hash(row["problem"]) in excluded_hashes:
            raise ValueError(f"{benchmark}: problem {row['problem_index']} overlaps the pilot train/development split")
        rows.append({"source_index": row["problem_index"], "problem_id": row["problem_id"], "problem": row["problem"], "answer": row["ground_truth_answer_override"], "solution": "", "type": row["type"], "level": row["level"], "topics": row["benchmark_example_metadata"].get("problem_type", []), "prompt": conversation(row["problem"])})
    if len({problem_hash(r["problem"]) for r in rows}) != len(rows):
        raise ValueError(f"{benchmark}: duplicate problem statements")
    return rows


def prepare_aime(root):
    from datasets import load_from_disk
    from src.evaluation.benchmarks import MATHARENA_AIME, load_matharena_aime

    pilot = {split: load_from_disk(str(data_dir(f"rl_1p5b_pilot_v1/{split}"))) for split in ("train", "development")}
    excluded = {problem_hash(r["problem"]) for split in pilot.values() for r in split}
    seen = set()
    for benchmark in AIME_BENCHMARKS:
        source, metadata = load_matharena_aime(benchmark)
        rows = freeze_aime_rows(source, benchmark, excluded | seen)
        seen |= {problem_hash(r["problem"]) for r in rows}
        # seed_offset gives problem k of each year its own sampling seed (2025 -> 100000 + 2025000 + k).
        metadata.update(shard_size=AIME_SHARD_SIZE, seed_offset=int(MATHARENA_AIME[benchmark]["year"]) * 1000, training_overlap=0, development_overlap=0, training_count=len(pilot["train"]), purpose="Post-release contamination check; report only, never use for tuning or checkpoint selection")
        payload = {"metadata": metadata, "rows": rows}
        path = root / f"{benchmark}_frozen.json"
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError("Refusing to change frozen benchmark")
        write_json(path, payload)
        print(f"Frozen {benchmark}: {len(rows)} problems at revision {metadata['revision']}; no pilot-split overlap", flush=True)


def read_competition_run(root, benchmark, mode, name):
    """Load one complete model/mode evaluation of a frozen competition benchmark."""
    frozen_path = root / f"{benchmark}_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    shard_size = frozen["metadata"]["shard_size"]
    expected = [r["source_index"] for r in frozen["rows"]]
    rows, hashes, samples = [], [], set()
    for shard in range(len(expected) // shard_size):
        directory = root / benchmark / mode / name
        config = json.loads((directory / f"shard_{shard}_config.json").read_text())
        if config.get("benchmark_sha256") != digest or config["mode"] != mode:
            raise ValueError(f"Benchmark inputs or mode changed: {benchmark}/{mode}/{name}/{shard}")
        path = directory / f"shard_{shard}.jsonl"
        part = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        if [r["source_index"] for r in part] != expected[shard * shard_size:(shard + 1) * shard_size]:
            raise ValueError(f"Incomplete or changed shard: {benchmark}/{mode}/{name}/{shard}")
        if [r["prompt_hash"] for r in part] != config["prompt_hashes"] or any(len(r["candidates"]) != config["num_samples"] for r in part):
            raise ValueError(f"Prompt or candidate mismatch: {benchmark}/{mode}/{name}/{shard}")
        rows.extend(part)
        hashes.extend(config["prompt_hashes"])
        samples.add(config["num_samples"])
    if len(samples) != 1:
        raise ValueError(f"Shards disagree on samples per problem: {benchmark}/{mode}/{name}")
    return rows, hashes


def competition_metrics(rows):
    from src.evaluation.pilot_exploration import pass_at_k, pass_k_values

    n = len(rows[0]["candidates"])
    counts = [sum(c["correct"] for c in r["candidates"]) for r in rows]
    candidates = [c for r in rows for c in r["candidates"]]
    return {"problems": len(rows), "samples_per_problem": n, "mean_accuracy": mean(c / n for c in counts),
            "strict_mean_accuracy": mean(per_problem_accuracy(rows, strict=True)),
            "pass_at_k": {str(k): mean(pass_at_k(n, c, k) for c in counts) for k in pass_k_values(n)},
            "solved_any": sum(c > 0 for c in counts),
            "truncated_fraction": sum(c["truncated"] for c in candidates) / len(candidates),
            "correct_without_boxed": sum(c["correct"] and "\\boxed" not in c.get("raw_solution", "") for c in candidates),
            "avg_generated_tokens": mean(c["generated_tokens"] for c in candidates)}


def per_problem_accuracy(rows, strict=False):
    """Stored (lenient, historical) correctness, or a re-grade with the closed-\\boxed strict grader."""
    if not strict:
        return [sum(c["correct"] for c in r["candidates"]) / len(r["candidates"]) for r in rows]
    from src.training.rl.smoke import grade_solution_strict

    return [sum(grade_solution_strict(c["raw_solution"], r["reference_answer"]) for c in r["candidates"]) / len(r["candidates"]) for r in rows]


def paired_difference(ids, baseline, other, resamples=10000, seed=0):
    """Mean per-problem accuracy change with a seeded paired bootstrap over problems."""
    diffs = [b - a for a, b in zip(baseline, other, strict=True)]
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return {"mean_difference": mean(diffs), "bootstrap_95ci": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
            "improved": [i for i, d in zip(ids, diffs) if d > 0], "worsened": [i for i, d in zip(ids, diffs) if d < 0]}


def format_change(d):
    return f"{d['mean_difference']:+.1%} [{d['bootstrap_95ci'][0]:+.1%}, {d['bootstrap_95ci'][1]:+.1%}]"


def summarize_aime(root, models, modes):
    results = []
    for mode in modes:
        loaded = {name: {b: read_competition_run(root, b, mode, name) for b in AIME_BENCHMARKS} for name in models}
        for b in AIME_BENCHMARKS:
            if len({len(loaded[name][b][0][0]["candidates"]) for name in models}) != 1:
                raise ValueError(f"Models use different samples per problem: {b}/{mode}")
        for benchmark in (*AIME_BENCHMARKS, "aime_pooled"):
            parts = AIME_BENCHMARKS if benchmark == "aime_pooled" else (benchmark,)
            ids = [f"{b}-{r['source_index']}" for b in parts for r in loaded["baseline"][b][0]]
            baseline_hashes = [h for b in parts for h in loaded["baseline"][b][1]]
            baseline = [x for b in parts for x in per_problem_accuracy(loaded["baseline"][b][0])]
            baseline_strict = [x for b in parts for x in per_problem_accuracy(loaded["baseline"][b][0], strict=True)]
            for name in models:
                rows = [r for b in parts for r in loaded[name][b][0]]
                if [h for b in parts for h in loaded[name][b][1]] != baseline_hashes:
                    raise ValueError(f"Model prompt mismatch: {benchmark}/{mode}/{name}")
                result = {"benchmark": benchmark, "mode": mode, "name": name, **competition_metrics(rows)}
                if name != "baseline":
                    result["vs_baseline"] = paired_difference(ids, baseline, per_problem_accuracy(rows))
                    result["vs_baseline_strict"] = paired_difference(ids, baseline_strict, per_problem_accuracy(rows, strict=True))
                results.append(result)
    note = "Single training seed. AIME 2025/2026 postdate the Qwen2.5-Math release; report only, never tune on them. Common-stop, max3000 tokens, exact/Math-Verify grader; strict columns re-grade crediting only a closed final \\boxed{} answer. Bootstrap intervals resample problems, not training seeds."
    write_json(root / "aime_summary.json", {"runs": results, "note": note})
    lines = ["# AIME 2025/2026 evaluation", "", note, "", "| Benchmark | Mode | Model | Problems | Samples | Mean accuracy | Strict accuracy | Top pass@k | Solved any | Truncated | Avg tokens | Change vs base [95% CI] | Strict change vs base [95% CI] |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        k, value = list(r["pass_at_k"].items())[-1]
        change, strict_change = (format_change(r[key]) if key in r else "—" for key in ("vs_baseline", "vs_baseline_strict"))
        lines.append(f"| {r['benchmark']} | {r['mode']} | {r['name']} | {r['problems']} | {r['samples_per_problem']} | {r['mean_accuracy']:.1%} | {r['strict_mean_accuracy']:.1%} | pass@{k} {value:.1%} | {r['solved_any']} | {r['truncated_fraction']:.1%} | {r['avg_generated_tokens']:.0f} | {change} | {strict_change} |")
    (root / "aime_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--suite", choices=["external", "aime"], default="external", help="external: MATH-500/GSM8K-100; aime: AIME 2025/2026")
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES), help="AIME summary models; baseline is required")
    parser.add_argument("--modes", nargs="+", choices=["greedy", "sampled"], default=["greedy", "sampled"], help="AIME summary modes")
    args = parser.parse_args()
    if args.suite == "aime" and "baseline" not in args.models:
        parser.error("AIME summaries compare against baseline; include it in --models")
    root = ensure_scratch_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.suite == "aime":
        prepare_aime(root) if args.prepare else summarize_aime(root, args.models, args.modes)
    else:
        prepare(root) if args.prepare else summarize(root)


if __name__ == "__main__":
    main()
