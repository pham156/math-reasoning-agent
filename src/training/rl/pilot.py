"""Shared-data, correctness-only pilots using the validated four-method backend."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from src.core.config import data_dir, ensure_scratch_path, model_dir, output_dir
from src.evaluation.answer import extract_final_answer
from src.training.grpo.train import write_json
from src.training.rl.methods import METHODS
from src.training.rl.smoke import conversation, run


def split_rows(rows, train_n, dev_n, seed=42):
    """Reserve category/level-balanced development rows before training selection."""
    groups = {}
    for row in rows:
        groups.setdefault((row["type"], row["level"]), []).append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    development = []
    keys = sorted(groups)
    while len(development) < dev_n:
        before = len(development)
        for key in keys:
            if groups[key] and len(development) < dev_n:
                development.append(groups[key].pop())
        if len(development) == before:
            raise ValueError("Insufficient development examples")
    training = [row for key in keys for row in groups[key]]
    rng.shuffle(training)
    if len(training) < train_n:
        raise ValueError("Insufficient audited training examples")
    return training[:train_n], development


def prepare_shared(args, tokenizer, run_dir):
    from datasets import Dataset, load_from_disk
    from math_verify import parse

    shared = ensure_scratch_path(args.split_dir, "shared split")
    manifest_path = shared / "manifest.json"
    if args.prepare_splits:
        if shared.exists() and any(shared.iterdir()):
            raise ValueError("Split directory already exists; reuse it without --prepare_splits")
        source = load_from_disk(args.dataset_dir)["train"]
        rows, seen = [], set()
        for index, row in enumerate(source):
            if row.get("level") not in {"Level 3", "Level 4", "Level 5"}:
                continue
            problem_hash = hashlib.sha256(" ".join(row["problem"].split()).encode()).hexdigest()
            answer = extract_final_answer(row["solution"])
            try:
                parseable = bool(answer.strip() and parse(r"\boxed{" + answer + "}"))
            except Exception:
                parseable = False
            prompt = conversation(row["problem"])
            length = len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
            if not parseable or length > args.max_prompt_length or problem_hash in seen:
                continue
            seen.add(problem_hash)
            rows.append({"source_index": index, "problem_hash": problem_hash, "problem": row["problem"], "answer": answer, "type": row["type"], "level": row["level"], "prompt": prompt})
        training, dev = split_rows(rows, 512, 300)
        shared.mkdir(parents=True)
        Dataset.from_list(training).save_to_disk(str(shared / "train"))
        Dataset.from_list(dev).save_to_disk(str(shared / "development"))
        def ids(items):
            return [{k: row[k] for k in ("source_index", "problem_hash", "type", "level")} for row in items]
        write_json(manifest_path, {"split_seed": 42, "source_split": "train", "dataset": args.dataset_dir, "max_prompt_length": args.max_prompt_length, "model_path": args.model_path, "audited_eligible": len(rows), "training": ids(training), "development": ids(dev)})
    manifest = json.loads(manifest_path.read_text())
    if manifest["max_prompt_length"] != args.max_prompt_length or manifest["model_path"] != args.model_path:
        raise ValueError("Frozen split tokenizer/prompt configuration mismatch")
    train = load_from_disk(str(shared / "train"))
    dev = load_from_disk(str(shared / "development"))
    if args.train_samples > len(train) or args.eval_samples > len(dev):
        raise ValueError("Requested slice exceeds frozen split")
    write_json(run_dir / "dataset_manifest.json", manifest)
    return train.select(range(args.train_samples)), list(dev.select(range(args.eval_samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=list(METHODS), default="grpo")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-1.5B")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--split_dir", default=str(data_dir("rl_1p5b_pilot_v1")))
    parser.add_argument("--output_dir", default=str(output_dir("rl_1p5b_study/pilot")))
    parser.add_argument("--prepare_splits", action="store_true")
    parser.add_argument("--train_samples", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, choices=[16])
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--clip_lower", type=float)
    parser.add_argument("--clip_upper", type=float)
    parser.add_argument("--save_steps", type=int, default=16)
    parser.add_argument("--segment_seconds", type=int, default=10800)
    parser.add_argument("--stop_after_step", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.deterministic_training = True
    args.purpose = "Checkpointed pilot; reserved MATH-train development diagnostics, not final benchmark results"
    for label in ("model_path", "dataset_dir", "split_dir", "output_dir"):
        ensure_scratch_path(getattr(args, label), label)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning_rate must be finite and positive")
    if not math.isfinite(args.kl_beta) or args.kl_beta < 0 or (args.kl_beta and args.algorithm != "grpo"):
        parser.error("kl_beta must be finite/nonnegative; nonzero beta is restricted to GRPO")
    from src.training.rl.methods import method_settings
    try:
        method_settings(args.algorithm, args.clip_lower, args.clip_upper)
    except ValueError as error:
        parser.error(str(error))
    if args.eval_samples < 1 or args.segment_seconds < 0 or args.stop_after_step < 0 or args.train_samples % 16 or not 0 < args.max_steps <= args.train_samples // 4 or args.max_steps % 4 or args.save_steps <= 0 or args.save_steps % 4:
        parser.error("Require complete 16-prompt batches and four-update boundaries within one pass")
    run_dir = Path(args.output_dir)
    if args.prepare_splits:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        prepare_shared(args, tokenizer, run_dir)
        return
    args.resume_from_checkpoint = None
    progress_file = run_dir / "progress.json"
    if args.resume and progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["status"] == "completed":
            print("Already completed; continuation is a no-op")
            return
        args.resume_from_checkpoint = progress["checkpoint"]
    elif run_dir.exists() and any(run_dir.iterdir()):
        parser.error("Existing run without safe progress checkpoint; investigate rather than overwrite")
    if args.resume_from_checkpoint:
        previous = json.loads((run_dir / "run_config.json").read_text())["arguments"]
        if previous.get("learning_rate", 1e-6) != args.learning_rate:
            parser.error("Resume configuration changed: learning_rate")
        for key, default in (("kl_beta", 0.0), ("clip_lower", None), ("clip_upper", None)):
            if previous.get(key, default) != getattr(args, key):
                parser.error(f"Resume configuration changed: {key}")
        for key in ("algorithm", "model_path", "split_dir", "seed", "max_steps", "train_samples", "eval_samples", "max_completion_length", "max_prompt_length", "deterministic_training"):
            if previous.get(key) != getattr(args, key):
                parser.error(f"Resume configuration changed: {key}")
    run(args, run_dir, dataset_preparer=prepare_shared)


if __name__ == "__main__":
    main()
