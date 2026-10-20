"""Dry-run first; submit independent per-experiment checkpoint chains with --submit."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess


ALGORITHMS = ("grpo", "dr_grpo", "dapo_style", "gspo")


def grid_rates(values):
    rates = sorted(set([*values, 1e-6]))
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be finite and positive")
    labels = [f"{rate:.0e}" for rate in rates]
    if len(labels) != len(set(labels)):
        raise ValueError("Learning rates collide in run labels; choose distinct rounded rates")
    return rates


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[2e-7, 5e-7, 1e-6])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_relative_to(Path("/scratch/gilbreth")):
        raise ValueError("Tuning root must be in scratch")
    prior = Path("/scratch/gilbreth/pham156/math-reasoning-agent/outputs/rl_1p5b_study/pilot_20261013_000313")
    controls = prior.parent / "controls_exploration_20261014"
    names = list(dict.fromkeys(args.algorithms))
    rates = grid_rates(args.learning_rates)
    if (root / "submission.json").exists():
        raise ValueError("Submission manifest already exists; do not duplicate jobs")
    runs = []
    for name in names:
        index = ALGORITHMS.index(name)
        gate = json.loads((prior / "resume_gate" / name / "resume_check.json").read_text())
        if not gate["passed"]:
            raise ValueError("Original resume gate did not pass")
        old = prior / "pilot" / name
        config = json.loads((old / "run_config.json").read_text())["arguments"]
        scratch = Path("/scratch/gilbreth/pham156/math-reasoning-agent")
        for key, value in {"model_path": str(scratch / "models/Qwen2.5-Math-1.5B"), "dataset_dir": str(scratch / "data/MATH"), "split_dir": str(scratch / "data/rl_1p5b_pilot_v1")}.items():
            if config.get(key) != value:
                raise ValueError(f"Reused data/model mismatch: {name}/{key}")
        for key, value in {"seed": 0, "max_steps": 128, "train_samples": 512, "eval_samples": 20, "gradient_accumulation_steps": 16, "max_completion_length": 3000, "max_prompt_length": 1024, "deterministic_training": True}.items():
            if config.get(key) != value:
                raise ValueError(f"Reused control mismatch: {name}/{key}")
        if json.loads((old / "training_args.json").read_text())["learning_rate"] != 1e-6:
            raise ValueError("Reused control LR mismatch")
        for rate in rates:
            label = f"{name}_lr{rate:.0e}"
            reused = rate == 1e-6
            runs.append({"algorithm": name, "learning_rate": rate, "label": label, "reused": reused, "run_root": str(old if reused else root / "runs" / label), "evaluation_root": str(controls if reused else root / "evaluations" / label), "training_jobs": [f"{job}_{index}" for job in [11938500, 11938501, 11938502]] if reused else [], "evaluation_job": None})
    for script in ["run_lr_pilot.slurm", "evaluate_lr_pilot.slurm", "summarize_lr_pilots.slurm"]:
        subprocess.run(["bash", "-n", f"scripts/rl/{script}"], check=True)
        subprocess.run(["sbatch", "--test-only", f"scripts/rl/{script}"], check=True)
    new_runs = sum(not r["reused"] for r in runs)
    if not new_runs:
        raise ValueError("Grid must include at least one new learning rate")
    print(f"{new_runs} new pilots: algorithms={names}, rates={rates}; reuse LR1e-6. Same512 rows,128 updates,seed0. Four safe checkpoint segments/run, then dev300. Standby; no independent-task limit.")
    print(f"Slurm tasks: {4 * new_runs} training + {3 * new_runs} evaluation + 1 summary = {7 * new_runs + 1}; sbatch submissions: {5 * new_runs + 1}")
    if not args.submit:
        print("Dry run passed; no jobs submitted")
        return
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"), "root": str(root), "reused_evaluation_root": str(controls), "runs": runs, "summary_job": None, "precision": "FP32 weights/BF16 compute", "training_stop_rule": "unchanged endoftext training generations", "evaluation_stop_ids": [151643, 151645], "segments_per_run": 4, "account": "ai-forge", "partition": "a100-80gb", "qos": "standby", "task_throttle": None, "selection": "Development accuracy, ties lower evaluation generated tokens, then lower LR; provisional until second-seed confirmation"}
    def submit(script, env, dependency=None):
        command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
        if "RL_RUN_LABEL" in env:
            prefix = "lr_eval" if script == "evaluate_lr_pilot.slurm" else "lr_train"
            command.append(f"--job-name={prefix}_{env['RL_RUN_LABEL']}")
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        command.append(f"scripts/rl/{script}")
        return subprocess.check_output(command, env=env, text=True).strip().split(";")[0]
    for run in runs:
        if run["reused"]:
            continue
        env = dict(os.environ, RL_LR_ROOT=str(root), RL_ALGORITHM=run["algorithm"], RL_RATE=str(run["learning_rate"]), RL_RUN_LABEL=run["label"])
        parent = None
        for _ in range(4):
            parent = submit("run_lr_pilot.slurm", env, parent)
            run["training_jobs"].append(parent)
            write_json(root / "submission.json", manifest)
        run["evaluation_job"] = submit("evaluate_lr_pilot.slurm", env, parent)
        write_json(root / "submission.json", manifest)
        print(f"{run['label']}: training={run['training_jobs']}, evaluation={run['evaluation_job']}", flush=True)
    evaluations = ":".join(r["evaluation_job"] for r in runs if not r["reused"])
    manifest["summary_job"] = submit("summarize_lr_pilots.slurm", dict(os.environ, RL_LR_ROOT=str(root)), evaluations)
    write_json(root / "submission.json", manifest)
    print(f"Summary: {manifest['summary_job']}", flush=True)


if __name__ == "__main__":
    main()
