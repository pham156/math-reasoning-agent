"""Dense (every-16-update) replay of the seed0 LR3e-6 GRPO trajectory, 0 -> 512.

The original runs kept one rotating checkpoint, so intermediate weights are gone. Three
independent deterministic replays regenerate them using the already validated stop/resume
path (``--stop_after_step``): A from the base model to 128, B forked from the preserved
128 checkpoint to 256, C forked from the validated reconstructed 256 checkpoint to 512.
After every stop the loadable weights are hard-linked into ``exports/step-N`` (immune to
checkpoint rotation; no extra disk) and validated before any evaluation may use them:
all rollout records and per-update loss/grad-norm/entropy must equal the original run,
and wherever original weights still exist (112, 128, 256, 512) the max absolute parameter
difference must be <= 1e-7.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from src.core.config import output_dir
from src.training.rl.checkpoint_storage import write_receipt

STUDY = output_dir("rl_1p5b_study")
ORIGINAL_128 = STUDY / "grpo_lr_upper_20261016/runs/grpo_lr3e-06"
ORIGINAL_512 = STUDY / "grpo_extended_20261105/run"
REPLAY_256 = STUDY / "checkpoint_benchmarks_20261106/replay"
SEGMENTS = {"a": (0, 128, ORIGINAL_128), "b": (128, 256, ORIGINAL_512), "c": (256, 512, ORIGINAL_512)}
WEIGHT_REFERENCES = {112: ORIGINAL_128 / "checkpoints/checkpoint-112", 128: ORIGINAL_128 / "checkpoints/checkpoint-128",
                     256: REPLAY_256 / "checkpoints/checkpoint-256", 512: ORIGINAL_512 / "checkpoints/checkpoint-512"}
RESUME_ONLY = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin"}
THRESHOLD = 1e-7


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def segment_of(step):
    for name, (start, end, _) in SEGMENTS.items():
        if start < step <= end:
            return name
    raise ValueError(f"Step {step} outside 1..512")


def compare_history(reference, replay, step):
    def metrics(path):
        return {r["step"]: {k: r[k] for k in ("loss", "grad_norm", "entropy") if k in r}
                for r in read(path)["log_history"] if "loss" in r and r["step"] <= step}
    left, right = metrics(reference), metrics(replay)
    if left.keys() != right.keys() or len(left) != step:
        raise ValueError("Replay training history incomplete")
    for index, values in left.items():
        if values.keys() != right[index].keys() or any(not math.isclose(v, right[index][k], rel_tol=0, abs_tol=1e-7) for k, v in values.items()):
            raise ValueError(f"Replay training metrics diverged at update {index}")


def max_weight_difference(left, right):
    from safetensors import safe_open

    def tensors(folder):
        out = {}
        for shard in sorted(Path(folder).glob("model*.safetensors")):
            handle = safe_open(str(shard), framework="pt")
            out.update({k: (handle, k) for k in handle.keys()})
        return out
    a, b = tensors(left), tensors(right)
    if a.keys() != b.keys() or not a:
        raise ValueError("Parameter keys differ")
    worst = 0.0
    for key in a:
        x, y = a[key][0].get_tensor(key), b[key][0].get_tensor(key)
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Parameter shape/dtype differs: {key}")
        worst = max(worst, float((x.double() - y.double()).abs().max()))
        if not math.isfinite(worst):
            raise ValueError("Non-finite parameter difference")
    return worst


def export_and_validate(root, step):
    name = segment_of(step)
    run = root / f"replay_{name}"
    reference_run = SEGMENTS[name][2]
    progress = read(run / "progress.json")
    if progress["global_step"] != step:
        raise ValueError(f"Replay {name} is at {progress['global_step']}, not {step}")
    checkpoint = Path(progress["checkpoint"])
    export = root / "exports" / f"step-{step}"
    if not export.exists():
        tmp = export.with_name(export.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=True)
        for item in checkpoint.iterdir():
            if item.name in RESUME_ONLY or item.is_dir():
                continue
            if not (tmp / item.name).exists():
                os.link(item, tmp / item.name)
        tmp.rename(export)
    reference = read_lines(reference_run / "rollouts.jsonl")
    current = read_lines(run / "rollouts.jsonl")
    # The October control predates extra reward-component logging; compare every field it recorded.
    fields = set(reference[0])
    if len(current) != step * 16 or any(not fields <= set(b) or any(a[k] != b[k] for k in fields) for a, b in zip(reference[:step * 16], current)):
        raise ValueError(f"Replay rollouts differ from the original run by step {step}")
    compare_history(reference_run / "train_history.json", run / "train_history.json", step)
    receipt = {"step": step, "segment": name, "rollouts_identical": True, "training_metrics_matched": True,
               "reference_run": str(reference_run), "export": str(export),
               "reference_rollouts_sha256": hashlib.sha256((reference_run / "rollouts.jsonl").read_bytes()).hexdigest()}
    if step in WEIGHT_REFERENCES:
        difference = max_weight_difference(export, WEIGHT_REFERENCES[step])
        receipt.update(weight_reference=str(WEIGHT_REFERENCES[step]), max_abs_weight_difference=difference)
        if difference > THRESHOLD:
            raise ValueError(f"Step {step} weights differ from preserved original by {difference}")
    write_receipt(export / "validation.json", receipt)
    print(json.dumps(receipt))


def fork_paused(source, destination):
    """Fork a validated *paused* replay (the reconstructed 256 run) without mutating it."""
    import shutil

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("Destination exists")
    progress = read(source / "progress.json")
    arguments = read(source / "run_config.json")["arguments"]
    if (progress["status"] != "paused" or progress["global_step"] != 256 or arguments["max_steps"] != 512 or
            arguments.get("training_passes") != 4 or arguments["learning_rate"] != 3e-6 or
            not (source.parent / "replay_256_check.json").exists()):
        raise ValueError("Source is not the validated paused 256 replay")
    checkpoint = Path(progress["checkpoint"])
    if any(not (checkpoint / f).is_file() for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")):
        raise ValueError("Source checkpoint lacks resume state")
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, destination / item.name)
    shutil.copytree(checkpoint, destination / "checkpoints" / checkpoint.name, copy_function=os.link)
    local = destination / "checkpoints" / checkpoint.name
    metadata = read(source / "run_config.json")
    metadata["arguments"] = {**arguments, "output_dir": str(destination), "resume_from_checkpoint": str(local)}
    write_receipt(destination / "run_config.json", metadata)
    write_receipt(destination / "progress.json", {**progress, "checkpoint": str(local)})
    write_receipt(destination / "continuation.json", {"source": str(source), "source_step": 256, "copy": "hard-linked immutable checkpoint files; logs copied"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("root", type=Path)
    v.add_argument("step", type=int)
    f = sub.add_parser("fork_c")
    f.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        export_and_validate(args.root, args.step)
    else:
        fork_paused(REPLAY_256, args.root / "replay_c")


if __name__ == "__main__":
    main()
