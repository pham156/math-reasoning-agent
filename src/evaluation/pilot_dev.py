"""Greedy evaluation on the frozen, disjoint pilot development split."""

import argparse
import json
import os
import time
from pathlib import Path

from src.core.config import data_dir, ensure_scratch_path
from src.training.grpo.train import write_json
from src.training.rl.smoke import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_dir", default=str(data_dir("rl_1p5b_pilot_v1")))
    parser.add_argument("--segment_seconds", type=int, default=12600)
    args = parser.parse_args()
    if args.segment_seconds < 0:
        parser.error("segment_seconds must be nonnegative; zero disables the pause")
    started = time.monotonic()
    for key in ("model_path", "output_dir", "split_dir"):
        ensure_scratch_path(getattr(args, key), key)
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import enable_full_determinism

    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Submit on a CUDA GPU")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = list(load_from_disk(str(Path(args.split_dir) / "development")))
    if len(rows) != 300:
        raise ValueError("Expected the frozen 300-problem development split")
    path = root / f"{args.name}.jsonl"
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(existing) > len(rows):
            raise ValueError("Too many existing evaluation rows")
        for index, row in enumerate(existing):
            if row["source_index"] != rows[index]["source_index"] or row["model_path"] != args.model_path:
                raise ValueError("Existing evaluation does not match this model/split")
    if len(existing) == len(rows) and (root / f"{args.name}_summary.json").exists():
        print(f"{args.name}: already complete")
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    with path.open("a", encoding="utf-8") as stream:
        for row in rows[len(existing):]:
            if args.segment_seconds and time.monotonic() - started >= args.segment_seconds:
                print(f"{args.name}: paused safely at {len(existing)}/300; rerun to continue", flush=True)
                return
            result = evaluate(model, tokenizer, [row], 3000)[0]
            result.update({"name": args.name, "model_path": args.model_path, "type": row["type"], "level": row["level"], "input_tokens": len(tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            existing.append(result)
            print(f"{args.name}: {len(existing)}/300 correct={sum(r['correct'] for r in existing)}", flush=True)
    write_json(root / f"{args.name}_summary.json", {"name": args.name, "n": len(existing), "accuracy": sum(r["correct"] for r in existing) / len(existing), "avg_generated_tokens": sum(r["generated_tokens"] for r in existing) / len(existing), "avg_runtime": sum(r["runtime_seconds"] for r in existing) / len(existing), "truncated": sum(r["truncated"] for r in existing), "model_path": args.model_path, "split_dir": args.split_dir, "generation": {"do_sample": False, "max_new_tokens": 3000}, "grader": "same exact/Math-Verify grader as RL pilots"})


if __name__ == "__main__":
    main()
