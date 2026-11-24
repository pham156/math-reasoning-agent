"""Rejection-sampling fine-tuning (RFT/RAFT) data from the unchanged base model.

Samples responses on exactly the frozen 512 RL training problems with the RL prompt,
grader, temperature 1 and 3000-token cap, then keeps verified-correct, terminated,
de-duplicated responses (at most ``--max_per_problem`` per problem). This follows
offline RFT (Yuan et al., 2023) / RAFT (Xiong et al., 2025, RLHFlow/Minimal-RL):
the SFT targets come from the policy's own distribution and are filtered by the same
verifiable reward used for RL, so no human-written solution style is imitated.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from src.core.config import data_dir, ensure_scratch_path, model_dir
from src.evaluation.answer import extract_final_answer
from src.evaluation.pilot_exploration import stop_ids
from src.training.grpo.train import write_json
from src.training.rl.smoke import conversation, grade_solution


TRAIN_N = 512


def shard_bounds(shard, num_shards, total=TRAIN_N):
    if num_shards < 1 or total % num_shards or not 0 <= shard < num_shards:
        raise ValueError("Shards must evenly partition the frozen training set")
    size = total // num_shards
    return shard * size, (shard + 1) * size


def normalized(text):
    return " ".join(text.split())


ALLOWED_UNICODE = set("–—‘’“”…·×÷±≤≥≠≈∞√π°′″→←⇒∈∉∪∩⊂⊆∑∏∫∠△⊥∥αβγδεθλμσφωΔΣΩ")


def boxed_end(text):
    """Index just past the closing brace of the last \\boxed{...}, or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for index in range(start + len("\\boxed"), len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return index + 1
    return None


def clean_ending(text, max_tail=40):
    """Reject answers that ramble or emit junk after the final boxed answer."""
    end = boxed_end(text)
    if end is None or len(text[end:].strip()) > max_tail:
        return False
    return all(ord(ch) < 128 or ch in ALLOWED_UNICODE for ch in text)


def select_targets(rows, max_per_problem, clean=False):
    """Keep correct, terminated, unique responses; deterministic order by sample index."""
    dataset, stats = [], {"problems": len(rows), "problems_with_target": 0, "responses": 0, "correct": 0, "truncated_correct": 0, "duplicates": 0}
    for row in rows:
        seen, kept = set(), []
        for candidate in sorted(row["candidates"], key=lambda c: c["sample_index"]):
            stats["responses"] += 1
            if not candidate["correct"]:
                continue
            stats["correct"] += 1
            if candidate["truncated"] or not candidate["terminated"]:
                stats["truncated_correct"] += 1
                continue
            if clean and not clean_ending(candidate["raw_solution"]):
                stats["unclean_correct"] = stats.get("unclean_correct", 0) + 1
                continue
            key = normalized(candidate["raw_solution"])
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            if len(kept) < max_per_problem:
                kept.append(candidate)
        stats["problems_with_target"] += bool(kept)
        dataset.extend({"source_index": row["source_index"], "problem": row["problem"], "solution": c["raw_solution"], "sample_index": c["sample_index"], "generated_tokens": c["generated_tokens"]} for c in kept)
    stats["targets"] = len(dataset)
    return dataset, stats


def generate(args):
    from datasets import load_from_disk
    from transformers import AutoTokenizer, GenerationConfig

    model_path = Path(args.model_path)
    train = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
    if len(train) != TRAIN_N:
        raise ValueError("Expected the frozen 512-problem training split")
    start, end = shard_bounds(args.shard, args.num_shards)
    rows = list(train.select(range(start, end)))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    eos = stop_ids(tokenizer, GenerationConfig.from_pretrained(model_path, local_files_only=True))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [tokenizer.apply_chat_template(conversation(r["problem"]), tokenize=False, add_generation_prompt=True) for r in rows]
    root = Path(args.output_dir) / "generations"
    root.mkdir(parents=True, exist_ok=True)
    config = {"model_path": str(model_path), "shard": args.shard, "num_shards": args.num_shards, "source_indices": [r["source_index"] for r in rows], "prompt_hashes": [hashlib.sha256(p.encode()).hexdigest() for p in prompts], "num_samples": args.num_samples, "temperature": args.temperature, "top_p": 1.0, "top_k": 0, "max_new_tokens": args.max_new_tokens, "stop_ids": eos, "seed_rule": "200000 + source_index per group", "precision": "FP32 parameters/BF16 compute", "grader": "pilot exact/Math-Verify"}
    config_path = root / f"shard_{args.shard}_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cannot resume a changed generation protocol")
    write_json(config_path, config)
    path = root / f"shard_{args.shard}.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if [r["source_index"] for r in existing] != config["source_indices"][:len(existing)]:
        raise ValueError("Existing shard has unexpected rows")
    if len(existing) == len(rows):
        print("Shard already complete")
        return
    if args.prepare_only:
        print(json.dumps({k: config[k] for k in ("shard", "num_samples", "temperature", "max_new_tokens", "stop_ids")}), len(rows), "problems")
        return
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from transformers.trainer_utils import enable_full_determinism

    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Requires a CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval()
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as stream:
        for row, prompt in zip(rows[len(existing):], prompts[len(existing):]):
            if args.segment_seconds and time.monotonic() - started >= args.segment_seconds:
                raise RuntimeError("Deadline reached; partial shard preserved, rerun to continue")
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            set_seed(200000 + row["source_index"])
            group_start = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, num_return_sequences=args.num_samples, max_new_tokens=args.max_new_tokens, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            candidates = []
            for index, sequence in enumerate(generated):
                tokens = sequence[inputs["input_ids"].shape[1]:].tolist()
                ended = next((i for i, token in enumerate(tokens) if token in eos), None)
                tokens = tokens[:ended + 1] if ended is not None else tokens
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                candidates.append({"sample_index": index, "raw_solution": raw, "answer": extract_final_answer(raw), "correct": bool(grade_solution(raw, row["answer"])), "generated_tokens": len(tokens), "terminated": ended is not None, "truncated": ended is None and len(tokens) >= args.max_new_tokens})
            payload = {"source_index": row["source_index"], "problem": row["problem"], "reference_answer": row["answer"], "type": row["type"], "level": row["level"], "candidates": candidates, "num_correct": sum(c["correct"] for c in candidates), "runtime_seconds": time.monotonic() - group_start, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            existing.append(payload)
            print(f"rft shard{args.shard}: {len(existing)}/{len(rows)} correct={payload['num_correct']}/{args.num_samples}", flush=True)


def build(args):
    root = Path(args.output_dir)
    rows = []
    configs = sorted((root / "generations").glob("shard_*_config.json"))
    num_shards = {json.loads(p.read_text())["num_shards"] for p in configs}
    if len(num_shards) != 1:
        raise ValueError("Missing or inconsistent generation shards")
    (count,) = num_shards
    for shard in range(count):
        config = json.loads((root / "generations" / f"shard_{shard}_config.json").read_text())
        part = [json.loads(line) for line in (root / "generations" / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip()]
        if [r["source_index"] for r in part] != config["source_indices"] or any(len(r["candidates"]) != config["num_samples"] for r in part):
            raise ValueError(f"Incomplete generation shard {shard}")
        rows.extend(part)
    if len(rows) != TRAIN_N or len({r["source_index"] for r in rows}) != TRAIN_N:
        raise ValueError("Generations do not cover the frozen training set exactly once")
    dataset, stats = select_targets(rows, args.max_per_problem, clean=args.clean)
    stats.update(max_per_problem=args.max_per_problem, base_sampled_pass_at_1=stats["correct"] / stats["responses"], source="unchanged Qwen2.5-Math-1.5B base; frozen RL training IDs only; no human-written solutions")
    if args.clean:
        stats["clean_filter"] = "<=40 non-space chars after the final \\boxed{...}; ASCII plus common math symbols only"
    write_json(root / ("rft_dataset_clean.json" if args.clean else "rft_dataset.json"), {"stats": stats, "rows": dataset})
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-1.5B")))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--num_shards", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--max_per_problem", type=int, default=4)
    parser.add_argument("--segment_seconds", type=int, default=12600)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--clean", action="store_true", help="build: also require a clean ending after the final boxed answer")
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()
    ensure_scratch_path(args.output_dir, "output_dir")
    ensure_scratch_path(args.model_path, "model_path")
    if args.build:
        build(args)
    elif args.shard is None:
        parser.error("--shard is required unless --build")
    else:
        generate(args)


if __name__ == "__main__":
    main()
