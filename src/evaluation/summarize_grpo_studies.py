"""Compare independently controlled reward/exploration pilots on fixed development data."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from src.core.config import ensure_scratch_path
from src.evaluation.pilot_exploration import pass_at_k
from src.evaluation.summarize_pilot_exploration import read_rows
from src.training.grpo.train import write_json
from src.training.rl.reward_design import batch_metrics
from src.training.rl.study_controls import profiles, validate_gate, validate_run


def compare_rows(rows, anchor):
    if rows.keys() != anchor.keys() or any(r["prompt_hash"] != anchor[i]["prompt_hash"] for i, r in rows.items()):
        raise ValueError("Evaluation IDs or prompts differ")
    def correct(row):
        return bool(row["candidates"][0]["correct"])
    return {"fixed": [i for i in rows if correct(rows[i]) and not correct(anchor[i])],
            "broken": [i for i in rows if not correct(rows[i]) and correct(anchor[i])]}


def evaluation_metrics(greedy, sampled):
    if not sampled.keys() <= greedy.keys() or any(r["prompt_hash"] != greedy[i]["prompt_hash"] for i, r in sampled.items()):
        raise ValueError("Sampling diagnostics use different development inputs")
    candidates = [r["candidates"][0] for r in greedy.values()]
    return {"accuracy": mean(c["correct"] for c in candidates), "n": len(greedy),
            "truncated": sum(c["truncated"] for c in candidates),
            "avg_generated_tokens": mean(c["generated_tokens"] for c in candidates),
            "avg_input_tokens": mean(r["input_tokens"] for r in greedy.values()),
            "avg_runtime": mean(r["runtime_seconds"] for r in greedy.values()),
            "pass_at_k": {str(k): mean(pass_at_k(8, sum(c["correct"] for c in r["candidates"]), k) for r in sampled.values()) for k in (1, 4, 8)},
            "sampled_mixed_rate": mean(0 < sum(c["correct"] for c in r["candidates"]) < 8 for r in sampled.values())}


def training_metrics(path, config):
    records = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != 512 * config["num_generations"] or len({r["problem"] for r in records}) != 512:
        raise ValueError("Training did not expose exactly512 unique prompts once")
    has_components = all("correctness" in r for r in records)
    if not has_components and config["reward_design"] != "correctness":
        raise ValueError("Shaped rewards require separately logged correctness")
    compatible = [{**r, "correctness": r.get("correctness", r["reward"]), "format_score": r.get("format_score", 0.0),
                   "format_bonus": r.get("format_bonus", 0.0), "length_penalty": r.get("length_penalty", 0.0)} for r in records]
    metrics = batch_metrics(compatible, config["num_generations"])
    if not has_components:
        metrics["format_mean"] = None
    history = json.loads((path / "train_history.json").read_text())["log_history"]
    lengths = [r["completions/mean_length"] for r in history if "completions/mean_length" in r]
    if len(lengths) != 32:
        raise ValueError("Unexpected number of prompt-exposure rollout batches")
    completion_tokens = sum(lengths) * 16 * config["num_generations"]
    if has_components and abs(completion_tokens - sum(r["completion_tokens"] for r in records)) > 1:
        raise ValueError("Completion token accounting differs from trainer logs")
    return {**metrics, "reward_mean": mean(r["reward"] for r in records),
            "response_length": mean(lengths), "completion_tokens": round(completion_tokens),
            "responses": len(records), "unique_problems": 512,
            "curve_metrics": [{k: v for k, v in r.items() if k in ("step", "reward", "reward_std", "entropy", "completions/mean_length", "frac_reward_zero_std", "clip_ratio/region_mean", "kl") or k.startswith("reward_components/")} for r in history if "step" in r]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    manifest = json.loads((root / "submission.json").read_text())
    if {r["label"] for r in manifest["runs"]} != set(profiles(manifest["study"])):
        raise ValueError("Incomplete condition manifest")
    greedy = {r["label"]: read_rows(Path(r["greedy_root"]), "greedy", "grpo", 300, 1) for r in manifest["runs"]}
    sampled = {r["label"]: read_rows(Path(r["sampled_root"]), "sampled", "grpo", 100, 8) for r in manifest["runs"]}
    rows, split = [], None
    for run in manifest["runs"]:
        label, config, path = run["label"], run["settings"], Path(run["run_root"])
        if config != profiles(manifest["study"])[label]:
            raise ValueError("Condition settings changed")
        validate_run(path, config)
        current_split = json.loads((path / "dataset_manifest.json").read_text())
        if split is not None and current_split != split:
            raise ValueError("Training/development manifests differ")
        split = current_split
        if not run["reused"]:
            validate_gate(root / "gates" / label, config)
        for mode, folder, n in (("greedy", run["greedy_root"], 3), ("sampled", run["sampled_root"], 5)):
            for shard in range(n):
                c = json.loads((Path(folder) / mode / "grpo" / f"shard_{shard}_config.json").read_text())
                if Path(c["model_path"]).resolve() != (path / "final_model").resolve() or c["max_new_tokens"] != 3000 or c["temperature"] != (0 if mode == "greedy" else 1.0):
                    raise ValueError("Evaluation checkpoint/protocol differs")
        anchor = f"correctness_seed{config['seed']}"
        changes = compare_rows(greedy[label], greedy[anchor])
        compare_rows(sampled[label], sampled[anchor])
        rows.append({"label": label, "settings": config, "anchor": anchor, "reused": run["reused"],
                     **evaluation_metrics(greedy[label], sampled[label]), **changes,
                     "training": training_metrics(path, config), "training_jobs": run["training_jobs"]})
    aggregates = {}
    if manifest["study"] == "reward":
        for design in ("correctness", "format_bonus", "length_penalty"):
            subset = [r for r in rows if r["settings"]["reward_design"] == design]
            aggregates[design] = {"n_seeds": len(subset), "accuracy_mean": mean(r["accuracy"] for r in subset),
                                  "accuracy_sample_std": stdev(r["accuracy"] for r in subset)}
    note = "Development-only comparison; no final benchmark selection. Reward and exploration studies remain separate. Rollout-count conditions match unique-problem exposure, NOT optimizer steps or token cost. Evaluation uses fixed temperature1/eight samples for every checkpoint. Training curves and cost accounting are included, but full GPU cost requires saved Slurm job elapsed times (including gates/evaluation)."
    write_json(root / "summary.json", {"study": manifest["study"], "runs": rows, "seed_aggregates": aggregates, "note": note})
    lines = [f"# GRPO {manifest['study']} study", "", note, "", "| Condition | Accuracy | Fix/break | pass@1/4/8 | Avg tokens | Train reward | Correctness | Mixed groups | Training completion tokens |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        t = r["training"]
        pk = "/".join(f"{r['pass_at_k'][str(k)]:.1%}" for k in (1, 4, 8))
        lines.append(f"| {r['label']} | {r['accuracy']:.1%} | {len(r['fixed'])}/{len(r['broken'])} | {pk} | {r['avg_generated_tokens']:.1f} | {t['reward_mean']:.3f} | {t['correctness_mean']:.3f} | {t['mixed_group_fraction']:.1%} | {t['completion_tokens']} |")
    lines.extend(["", "Reward seed aggregates: " + json.dumps(aggregates), "", "Raw per-update entropy/reward/variance/response-length curves are in summary.json; rollout records retain separate correctness, formatting and length components. Fixed/broken IDs are relative to the same-seed best-setting correctness control."])
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    from src.training.rl.checkpoint_storage import cleanup_completed_studies
    cleanup_completed_studies(root.parent)


if __name__ == "__main__":
    main()
