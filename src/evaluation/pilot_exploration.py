"""Common-stop greedy controls and fixed-set eight-rollout pilot diagnostics."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from src.core.config import data_dir, ensure_scratch_path, model_dir
from src.evaluation.answer import extract_final_answer
from src.training.grpo.train import write_json
from src.training.rl.smoke import grade_solution


NAMES = ("baseline", "grpo", "dr_grpo", "dapo_style", "gspo", "sft")


def pass_at_k(n, correct, k):
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Invalid pass@k counts")
    return 1.0 if n - correct < k else 1 - math.comb(n - correct, k) / math.comb(n, k)


def stop_ids(tokenizer, generation_config):
    configured = generation_config.eos_token_id
    configured = configured if isinstance(configured, list) else [configured]
    chat_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_end is None or chat_end == tokenizer.unk_token_id:
        raise ValueError("Tokenizer lacks the required chat-end token")
    return sorted({x for x in [*configured, tokenizer.eos_token_id, chat_end] if x is not None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_root", required=True)
    parser.add_argument("--comparison_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["greedy", "sampled"], required=True)
    parser.add_argument("--name", choices=NAMES, required=True)
    parser.add_argument("--model_path", help="Explicit scratch checkpoint for controlled training variants")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--segment_seconds", type=int, default=12600)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--benchmark_file", help="Frozen external benchmark JSON; greedy only")
    args = parser.parse_args()
    shard_size = 100 if args.mode == "greedy" else 20
    total = 300 if args.mode == "greedy" else 100
    benchmark = None
    if args.benchmark_file:
        if args.mode != "greedy":
            parser.error("External benchmarks use greedy evaluation only")
        benchmark = json.loads(ensure_scratch_path(args.benchmark_file).read_text())
        total = len(benchmark["rows"])
        if not total or total % shard_size:
            parser.error("External benchmark must contain complete 100-problem shards")
    if not 0 <= args.shard < total // shard_size or args.segment_seconds < 0:
        parser.error("Invalid shard/deadline")
    if args.mode == "sampled" and args.name == "sft":
        parser.error("SFT remains a greedy-only supporting control")
    for key in ("pilot_root", "comparison_root", "output_dir"):
        ensure_scratch_path(getattr(args, key), key)
    model_path = model_dir("Qwen2.5-Math-1.5B") if args.name == "baseline" else Path(args.pilot_root) / "pilot" / args.name / "final_model"
    if args.name == "sft":
        model_path = Path(args.comparison_root) / "sft/final_model"
    if args.model_path:
        model_path = ensure_scratch_path(args.model_path, "model_path")
    root = Path(args.output_dir) / args.mode / args.name
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    from datasets import load_from_disk
    from transformers import AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    eos = stop_ids(tokenizer, GenerationConfig.from_pretrained(model_path, local_files_only=True))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dev = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/development")))
    train = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
    if len(dev) != 300 or set(dev["source_index"]) & set(train["source_index"]):
        raise ValueError("Frozen development split is invalid or overlaps training")
    source = benchmark["rows"] if benchmark else dev
    rows = list(source[args.shard * shard_size:(args.shard + 1) * shard_size]) if benchmark else list(dev.select(range(args.shard * shard_size, (args.shard + 1) * shard_size)))
    prompt_hashes = [hashlib.sha256(tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True).encode()).hexdigest() for r in rows]
    config = {"name": args.name, "mode": args.mode, "shard": args.shard, "model_path": str(model_path), "source_indices": [r["source_index"] for r in rows], "prompt_hashes": prompt_hashes, "stop_ids": eos, "max_new_tokens": 3000, "num_samples": 1 if args.mode == "greedy" else 8, "temperature": 0 if args.mode == "greedy" else 1.0, "top_p": 1.0, "top_k": 0, "seed_rule": "100000 + source_index for each sampled group", "precision": "FP32 parameters/BF16 compute", "grader": "pilot exact/Math-Verify", "sampling_batch": 1 if args.mode == "greedy" else 8}
    config_path = root / f"shard_{args.shard}_config.json"
    if benchmark:
        config["benchmark"] = benchmark["metadata"]
        config["benchmark_sha256"] = hashlib.sha256(Path(args.benchmark_file).read_bytes()).hexdigest()
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cannot resume a changed evaluation protocol")
    write_json(config_path, config)
    if args.prepare_only:
        print(json.dumps(config), flush=True)
        return
    path = root / f"shard_{args.shard}.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if [r["source_index"] for r in existing] != config["source_indices"][:len(existing)] or len(existing) > len(rows):
        raise ValueError("Existing shard has unexpected source rows")
    if len(existing) == len(rows):
        print("Shard already complete")
        return
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from transformers.trainer_utils import enable_full_determinism

    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires a BF16 CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.config.use_cache = True
    model.eval()
    with path.open("a", encoding="utf-8") as stream:
        for row, prompt_hash in zip(rows[len(existing):], prompt_hashes[len(existing):]):
            if args.segment_seconds and time.monotonic() - started >= args.segment_seconds:
                print("Paused safely between problems; rerun this shard to continue", flush=True)
                return
            prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            set_seed(100000 + row["source_index"])
            kwargs = {"do_sample": args.mode == "sampled", "num_return_sequences": config["num_samples"], "max_new_tokens": 3000, "eos_token_id": eos, "pad_token_id": tokenizer.pad_token_id, "use_cache": True}
            if args.mode == "sampled":
                kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
            group_start = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, **kwargs)
            elapsed = time.monotonic() - group_start
            candidates = []
            for index, sequence in enumerate(generated):
                tokens = sequence[inputs["input_ids"].shape[1]:].tolist()
                ended = next((i for i, token in enumerate(tokens) if token in eos), None)
                tokens = tokens[:ended + 1] if ended is not None else tokens
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                candidates.append({"sample_index": index, "raw_solution": raw, "answer": extract_final_answer(raw), "correct": bool(grade_solution(raw, row["answer"])), "generated_tokens": len(tokens), "truncated": ended is None and len(tokens) >= 3000, "ending_token_id": tokens[-1] if tokens else None})
            correct = sum(c["correct"] for c in candidates)
            payload = {"source_index": row["source_index"], "problem": row["problem"], "reference_answer": row["answer"], "type": row["type"], "level": row["level"], "prompt_hash": prompt_hash, "name": args.name, "mode": args.mode, "candidates": candidates, "num_correct": correct, "pass_at_k": {str(k): pass_at_k(len(candidates), correct, k) for k in ([1] if args.mode == "greedy" else [1, 4, 8])}, "runtime_seconds": elapsed, "input_tokens": inputs["input_ids"].shape[1], "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            existing.append(payload)
            print(f"{args.name} {args.mode} shard{args.shard}: {len(existing)}/{len(rows)} correct={correct}/{len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
