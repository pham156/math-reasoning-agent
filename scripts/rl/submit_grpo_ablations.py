"""Host stdlib submitter for seed confirmation and independent GRPO KL/clip gates."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from src.training.rl.ablation_controls import CASES, CONTROL, check_saved, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    scratch = Path("/scratch/gilbreth/pham156/math-reasoning-agent")
    if not root.is_relative_to(scratch / "outputs"):
        raise ValueError("All artifacts must be under project scratch outputs")
    if (root / "submission.json").exists():
        raise ValueError("Manifest exists; inspect it rather than duplicate submissions")
    study = scratch / "outputs/rl_1p5b_study"
    original = study / "pilot_20261013_000313"
    controls = study / "controls_exploration_20261014"
    upper = json.loads((study / "grpo_lr_upper_20261016/submission.json").read_text())
    reused = []
    for label, rate, run_root, evaluation_root, jobs in (
        ("seed0_lr1e-06", 1e-6, original / "pilot/grpo", controls, ["11938500_0", "11938501_0", "11938502_0"]),
        ("seed0_lr3e-06", 3e-6, study / "grpo_lr_upper_20261016/runs/grpo_lr3e-06", study / "grpo_lr_upper_20261016/evaluations/grpo_lr3e-06", next(r["training_jobs"] for r in upper["runs"] if r["learning_rate"] == 3e-6)),
    ):
        result = json.loads((run_root / "smoke_result.json").read_text())
        expected = {**CONTROL, "learning_rate": rate}
        check_saved(label, result["arguments"], json.loads((run_root / "training_args.json").read_text()), expected)
        if result["global_step"] != 128 or not (run_root / "final_model/config.json").exists():
            raise ValueError("Reused model incomplete")
        for key, value in {"model_path": str(scratch / "models/Qwen2.5-Math-1.5B"), "split_dir": str(scratch / "data/rl_1p5b_pilot_v1"), "train_samples": 512, "eval_samples": 20, "gradient_accumulation_steps": 16, "deterministic_training": True}.items():
            if result["arguments"].get(key) != value:
                raise ValueError(f"Reused control setting mismatch: {key}")
        reused.append({"label": label, "settings": expected, "reused": True, "run_root": str(run_root), "evaluation_root": str(evaluation_root), "training_jobs": jobs})
    if not json.loads((original / "resume_gate/grpo/resume_check.json").read_text())["passed"]:
        raise ValueError("Original GRPO restart control did not pass")
    probe = root / "probes.json"
    if not probe.exists() or len(json.loads(probe.read_text())["rows"]) != 8:
        raise ValueError("Prepare the eight fixed drift probes before submission")
    subprocess.run(["bash", "-n", "scripts/rl/run_grpo_ablation.slurm"], check=True)
    for extra in ([], ["--array=0-2"]):
        subprocess.run(["sbatch", "--test-only", *extra, "scripts/rl/run_grpo_ablation.slurm"], check=True)
    print("Six new GRPO pilots: seed1 LR1e-6/3e-6; seed0 KL.001/.01 or clip.1/.3. Same512 training problems,128updates; reuse two seed0 LR controls.")
    print("49 Slurm tasks:4 independent objective/restart gates +24 training segments +18 eval tasks +2 reused-control drift checks +1 summary. No task throttle or cross-experiment training barrier.")
    if not args.submit:
        print("Dry run passed; no jobs submitted")
        return
    runs = reused + [{"label": label, "settings": settings(label), "reused": False, "run_root": str(root / "runs" / label), "evaluation_root": str(root / "evaluations" / label), "training_jobs": []} for label in CASES]
    manifest = {"submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"), "runs": runs, "root": str(root), "controls": str(controls), "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(), "account": "ai-forge", "partition": "a100-80gb", "qos": "standby", "segments_per_run": 4, "training_samples": 512, "training_updates": 128, "evaluation_samples": 300, "task_throttle": None, "note": "LR is a paired seed0/1 comparison. KL/clipping use original seed0 LR1e-6 anchor, changing one factor only. All start from untouched base model. No external-benchmark model selection."}
    def save():
        (root / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")
    def submit(phase, run, dependency=None, array=None):
        command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", f"--job-name=grpo_{phase}_{run['label']}"]
        if dependency:
            command.append("--dependency=afterok:" + dependency)
        if array:
            command.extend(["--array=" + array, "--output=/home/pham156/math-reasoning-agent/logs/%A_%a_%x.out", "--error=/home/pham156/math-reasoning-agent/logs/%A_%a_%x.err"])
        if phase == "summary":
            command.extend(["--time=00:15:00", "--mem=8G", "--cpus-per-task=1"])
        env = dict(os.environ, RL_ABLATION_ROOT=str(root), RL_RUN_LABEL=run["label"], RL_PHASE=phase, RL_MODEL_PATH=str(Path(run.get("run_root", root)) / "final_model"))
        return subprocess.check_output([*command, "scripts/rl/run_grpo_ablation.slurm"], env=env, text=True).strip().split(";")[0]
    save()
    dependencies = []
    for run in runs:
        if run["reused"]:
            run["drift_job"] = submit("drift", run)
            dependencies.append(run["drift_job"])
        else:
            parent = None
            if run["label"].startswith(("kl_", "clip_")):
                parent = run["gate_job"] = submit("gate", run)
                save()
            for _ in range(4):
                parent = submit("train", run, parent)
                run["training_jobs"].append(parent)
                save()
            run["evaluation_job"] = submit("eval", run, parent, "0-2")
            dependencies.append(run["evaluation_job"])
        save()
        print(run["label"], {key: value for key, value in run.items() if key.endswith("job") or key == "training_jobs"}, flush=True)
    manifest["summary_job"] = submit("summary", {"label": "all"}, ":".join(dependencies))
    save()
    print("Summary:", manifest["summary_job"], flush=True)


if __name__ == "__main__":
    main()
