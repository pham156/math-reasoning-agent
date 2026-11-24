"""Scoped retention for an explicitly opted-in reward/exploration submission."""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil


OUTPUTS = Path(os.environ.get("SCRATCH_ROOT", "/scratch/gilbreth/pham156/math-reasoning-agent")) / "outputs"


def load_policy(root):
    root = Path(root).resolve()
    file = root / "retention_policy.json"
    if not file.exists():
        return None
    policy = json.loads(file.read_text())
    if not root.is_relative_to(OUTPUTS.resolve()) or root == OUTPUTS.resolve() or policy != {
        "version": 1, "root": str(root), "active_pilot_checkpoints": 1,
        "cleanup_gates_after_validation": True, "cleanup_pilots_after": "both_study_summaries",
        "keep_final_models": True, "completed_pilot_checkpoints": 1,
    }:
        raise ValueError("Unrecognized or out-of-scope checkpoint retention policy")
    if not (root / "submission.json").is_file():
        raise ValueError("Retention requires an existing submission manifest")
    return policy


def context(path):
    path = Path(path).absolute()
    for root in path.parents:
        policy = load_policy(root)
        if policy is None:
            continue
        if path.resolve() != path:
            raise ValueError("Retention refuses symlinked run paths")
        parts = path.relative_to(root).parts
        if len(parts) not in (3, 4) or parts[0] not in ("reward", "exploration") or parts[1] not in ("runs", "gates"):
            raise ValueError("Retention is restricted to this submission's new runs and gates")
        manifest = json.loads((root / "submission.json").read_text())
        cases = manifest["studies"][parts[0]]["runs"]
        case = next((r for r in cases if r["label"] == parts[2] and not r["reused"]), None)
        if case is None or Path(case["run_root"]).resolve() != root / parts[0] / "runs" / parts[2]:
            raise ValueError("Cannot clean a reused or unregistered condition")
        if (parts[1] == "runs" and len(parts) != 3) or (len(parts) == 4 and parts[3] not in ("continuous", "resumed")):
            raise ValueError("Unexpected condition directory")
        return root, parts, case
    return None


def save_limit(run_dir):
    found = context(run_dir)
    return 1 if found and found[1][1] == "runs" else 2


def checkpoint_due(step, target, boundary, stop_requested):
    return step > 0 and step % boundary == 0 and (stop_requested or step >= target)


def checkpoint_dirs(run_dir):
    run_dir = Path(run_dir).absolute()
    if run_dir.resolve() != run_dir:
        raise ValueError("Refusing a symlinked checkpoint ancestor")
    parent = run_dir / "checkpoints"
    if parent.is_symlink():
        raise ValueError("Refusing a symlinked checkpoint directory")
    found = []
    for item in parent.iterdir() if parent.exists() else []:
        # Transformers may emit a model card alongside checkpoint directories.
        if item.name == "README.md" and item.is_file() and not item.is_symlink():
            continue
        if not re.fullmatch(r"checkpoint-\d+", item.name) or not item.is_dir() or item.is_symlink():
            raise ValueError("Unexpected checkpoint entry; inspect rather than delete")
        found.append(item)
    return found


def weights(model, require_complete=True):
    model = Path(model).absolute()
    if model.resolve() != model:
        raise ValueError("Refusing a symlinked model export")
    found = sorted(set(model.glob("*.safetensors")) | set(model.glob("pytorch_model*.bin")))
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size == 0 for p in found):
        raise ValueError("Invalid exported model shard")
    for index in [*model.glob("*.safetensors.index.json"), *model.glob("pytorch_model*.bin.index.json")]:
        if index.is_symlink():
            raise ValueError("Refusing a symlinked model index")
        names = set(json.loads(index.read_text())["weight_map"].values())
        if any(Path(name).name != name or (require_complete and not (model / name).is_file()) for name in names):
            raise ValueError("Model export is missing a required shard")
    return found


def write_receipt(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cleanup_gate(gate, config):
    from src.training.rl.study_controls import validate_gate

    found = context(gate)
    if found is None:
        return
    root, parts, case = found
    if parts[1] != "gates" or len(parts) != 3 or config != case["settings"]:
        raise ValueError("Gate cleanup configuration mismatch")
    gate = root.joinpath(*parts)
    with (gate / ".storage_cleanup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_gate(gate, config)
        boundary = json.loads((gate / "boundary_check.json").read_text())
        final = json.loads((gate / "resume_check.json").read_text())
        if not boundary["passed"] or not final["same_parameter_keys"] or not math.isfinite(final["max_parameter_difference"]) or final["max_parameter_difference"] > 1e-7 or not final["deterministic_training"]:
            raise ValueError("Gate evidence is incomplete; preserve all artifacts")
        evidence = [gate / "boundary_check.json", gate / "resume_check.json"]
        for name in ("continuous", "resumed"):
            evidence.extend(gate / name / f for f in ("run_config.json", "smoke_result.json", "training_args.json", "rollouts.jsonl"))
        hashes = {str(p.relative_to(gate)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
        receipt = gate / "storage_cleanup.json"
        if receipt.exists() and json.loads(receipt.read_text())["evidence_sha256"] != hashes:
            raise ValueError("Gate evidence changed since cleanup")
        targets = []
        for name in ("continuous", "resumed"):
            run = gate / name
            targets.extend(checkpoint_dirs(run))
            targets.extend(weights(run / "final_model", require_complete=not receipt.exists()))
            targets.extend((run / "final_model").glob("*.safetensors.index.json"))
            targets.extend((run / "final_model").glob("pytorch_model*.bin.index.json"))
        payload = {"status": "deleting", "evidence_sha256": hashes,
                   "scope": "Validated gate checkpoints and weight shards only; small metadata/rollouts preserved"}
        write_receipt(receipt, payload)
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        write_receipt(receipt, {**payload, "status": "complete"})


def cleanup_completed_studies(root):
    from src.training.rl.study_controls import updates, validate_gate, validate_run

    root = Path(root).resolve()
    if load_policy(root) is None:
        return
    with (root / ".checkpoint_cleanup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((root / study / f"summary.{suffix}").exists() for study in ("reward", "exploration") for suffix in ("json", "md")):
            return
        manifest = json.loads((root / "submission.json").read_text())
        runs = []
        for study in ("reward", "exploration"):
            summary = json.loads((root / study / "summary.json").read_text())
            cases = manifest["studies"][study]["runs"]
            if {r["label"] for r in summary["runs"]} != {r["label"] for r in cases}:
                raise ValueError("Summary is incomplete; preserve checkpoints")
            for case in cases:
                row = next(r for r in summary["runs"] if r["label"] == case["label"])
                if row["n"] != 300 or set(row["pass_at_k"]) != {"1", "4", "8"} or row["settings"] != case["settings"]:
                    raise ValueError("Evaluation summary mismatch; preserve checkpoints")
                if case["reused"]:
                    continue
                run = Path(case["run_root"])
                found = context(run)
                if found is None or found[1][:2] != (study, "runs"):
                    raise ValueError("Unexpected cleanup target")
                validate_run(run, case["settings"])
                progress = json.loads((run / "progress.json").read_text())
                if progress["status"] != "completed" or progress["global_step"] != updates(case["settings"]) or not weights(run / "final_model"):
                    raise ValueError("No completed final model; preserve resume state")
                gate = root / study / "gates" / case["label"]
                validate_gate(gate, case["settings"])
                runs.append((run, gate, case["settings"]))
        targets, retained = [], []
        for run, _, config in runs:
            checkpoints = sorted(checkpoint_dirs(run), key=lambda p: int(p.name.split("-")[-1]))
            if not checkpoints:
                raise ValueError("Missing final resumable checkpoint; preserve all artifacts")
            latest = checkpoints[-1]
            step = updates(config)
            required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
            if (latest.name != f"checkpoint-{step}" or
                    any(not (latest / name).is_file() or (latest / name).is_symlink() or
                        (latest / name).stat().st_size == 0 for name in required) or
                    json.loads((latest / "trainer_state.json").read_text())["global_step"] != step or
                    not weights(latest)):
                raise ValueError("Incomplete final resumable checkpoint; preserve all artifacts")
            retained.append(str(latest))
            targets.extend(checkpoints[:-1])
        payload = {"status": "deleting", "scope": "Older new pilot checkpoints only; latest resumable checkpoint, final models/results and reused controls preserved",
                   "retained_checkpoints": retained,
                   "run_roots": [str(run) for run, _, _ in runs]}
        receipt = root / "checkpoint_cleanup.json"
        write_receipt(receipt, payload)
        for _, gate, config in runs:
            cleanup_gate(gate, config)
        for path in targets:
            shutil.rmtree(path)
        write_receipt(receipt, {**payload, "status": "complete"})
