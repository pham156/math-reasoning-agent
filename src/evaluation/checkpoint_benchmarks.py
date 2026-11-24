"""Fixed external benchmarks and evidence checks for reconstructed GRPO checkpoints."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from src.evaluation.pilot_benchmarks import BENCHMARKS, summarize_rows
from src.training.rl.checkpoint_storage import weights, write_receipt
from src.training.rl.extend import read, validate_extension_gate


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_replay(original, replay, step):
    if len(replay) != step * 16 or replay != original[:step * 16]:
        raise ValueError("Reconstructed rollout prefix differs from the completed original run")


def validate_replay(root, step):
    m = read(root / "submission.json")
    original = Path(m["extension_root"])
    validate_extension_gate(original / "gate")
    run = root / "replay"
    progress = read(run / "progress.json")
    if progress["status"] != "paused" or progress["global_step"] != step:
        raise ValueError("Replay milestone incomplete")
    if read(run / "dataset_manifest.json") != read(original / "run/dataset_manifest.json"):
        raise ValueError("Replay data changed")
    reference = read_lines(original / "run/rollouts.jsonl")
    current = read_lines(run / "rollouts.jsonl")
    compare_replay(reference, current, step)
    def metrics(path):
        return {r["step"]: {k: r[k] for k in ("loss", "grad_norm", "entropy") if k in r}
                for r in read(path)["log_history"] if "loss" in r and r["step"] <= step}
    left, right = metrics(original / "run/train_history.json"), metrics(run / "train_history.json")
    if left.keys() != right.keys() or len(left) != step:
        raise ValueError("Replay training history incomplete")
    for index, values in left.items():
        if values.keys() != right[index].keys() or any(not math.isclose(value, right[index][key], rel_tol=0, abs_tol=1e-7) for key, value in values.items()):
            raise ValueError("Replay training metrics diverged")
    checkpoint = run / "checkpoints" / f"checkpoint-{step}"
    if not weights(checkpoint) or any(not (checkpoint / f).is_file() for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth")):
        raise ValueError("Reconstructed checkpoint lacks resume state")
    payload = {"step": step, "rollouts_identical": True, "training_metrics_matched": True,
               "original_rollouts_sha256": hashlib.sha256((original / "run/rollouts.jsonl").read_bytes()).hexdigest(),
               "note": "Original intermediate weights were rotated; equality of their weight hash cannot be checked. Replay uses the validated backend and matches all rollout records and logged training metrics."}
    if step == 192:
        folder = root / "development/step192/greedy/grpo"
        rows = read_lines(folder / "shard_0.jsonl")
        config = read(folder / "shard_0_config.json")
        expected = read(Path(m["control"]["greedy_root"]) / "greedy/grpo/shard_0_config.json")
        if ([r["source_index"] for r in rows] != expected["source_indices"] or
                config["prompt_hashes"] != expected["prompt_hashes"] or
                [r["prompt_hash"] for r in rows] != config["prompt_hashes"] or
                Path(config["model_path"]).resolve() != checkpoint.resolve() or
                config["max_new_tokens"] != 3000 or config["temperature"] != 0):
            raise ValueError("Dense development evaluation incomplete or changed")
        payload["development"] = summarize_rows(rows)
    write_receipt(root / f"replay_{step}_check.json", payload)
    print(json.dumps(payload))


def benchmark_rows(folder, name, model, frozen_path):
    frozen = read(frozen_path)
    expected = [r["source_index"] for r in frozen["rows"]]
    rows = []
    for shard in range(len(expected) // 100):
        directory = folder / "greedy" / name
        config = read(directory / f"shard_{shard}_config.json")
        part = read_lines(directory / f"shard_{shard}.jsonl")
        if (config["benchmark_sha256"] != hashlib.sha256(frozen_path.read_bytes()).hexdigest() or
                Path(config["model_path"]).resolve() != Path(model).resolve() or
                config["max_new_tokens"] != 3000 or config["temperature"] != 0 or
                [r["source_index"] for r in part] != expected[shard * 100:(shard + 1) * 100] or
                [r["prompt_hash"] for r in part] != config["prompt_hashes"] or
                any(len(r["candidates"]) != 1 for r in part)):
            raise ValueError("Incomplete or mismatched external benchmark protocol")
        rows.extend(part)
    return rows


def summarize(root):
    m = read(root / "submission.json")
    source = Path(m["benchmark_source"])
    results = []
    for benchmark in BENCHMARKS:
        frozen = source / f"{benchmark}_frozen.json"
        baseline = benchmark_rows(source / benchmark, "baseline", m["baseline_model"], frozen)
        baseline_correct = [r["candidates"][0]["correct"] for r in baseline]
        for label, model in [("base_no_rl", m["baseline_model"]), *m["models"].items()]:
            rows = baseline if label == "base_no_rl" else benchmark_rows(root / "evaluations" / label / benchmark, "grpo", model, frozen)
            if [r["prompt_hash"] for r in rows] != [r["prompt_hash"] for r in baseline]:
                raise ValueError("Benchmark prompts differ across models")
            correct = [r["candidates"][0]["correct"] for r in rows]
            result = {"benchmark": benchmark, "model": label, **summarize_rows(rows),
                      "fixed_vs_base": [r["source_index"] for r, a, b in zip(rows, baseline_correct, correct) if not a and b],
                      "broken_vs_base": [r["source_index"] for r, a, b in zip(rows, baseline_correct, correct) if a and not b]}
            for key in ("type", "level"):
                groups = defaultdict(list)
                for row in rows:
                    groups[row[key] or "unspecified"].append(row)
                result["by_" + key] = {k: summarize_rows(v) for k, v in groups.items()}
            results.append(result)
    note = "Greedy, max3000, seed0; identical previously frozen external benchmarks, not untouched confirmation data. Step256 is reconstructed and validated against original rollout/history prefixes, not original intermediate weight hashes. No retrieval/verifier or new optimizer settings."
    write_receipt(root / "summary.json", {"runs": results, "note": note})
    lines = ["# GRPO Checkpoint Benchmarks", "", note, "", "| Benchmark | Model | n | Accuracy | Avg tokens | Avg seconds | Truncated |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['benchmark']} | {r['model']} | {r['n']} | {r['accuracy']:.1%} | {r['avg_generated_tokens']:.1f} | {r['avg_runtime']:.1f} | {r['truncated']} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--validate_replay", type=int, choices=[192, 256])
    args = parser.parse_args()
    from src.core.config import ensure_scratch_path
    root = ensure_scratch_path(args.root)
    validate_replay(root, args.validate_replay) if args.validate_replay else summarize(root)


if __name__ == "__main__":
    main()
