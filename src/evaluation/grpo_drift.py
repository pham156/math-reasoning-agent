"""Exact next-token distribution KL on fixed baseline-generated development prefixes."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from src.core.config import data_dir, ensure_scratch_path, model_dir
from src.training.grpo.train import write_json


def categorical_kl(policy_logits, reference_logits):
    import torch
    policy = policy_logits.float().log_softmax(-1)
    reference = reference_logits.float().log_softmax(-1)
    values = (policy.exp() * (policy - reference)).sum(-1)
    if not torch.isfinite(values).all() or (values < -1e-5).any():
        raise FloatingPointError("Invalid reference divergence")
    return values.clamp_min(0)


def prepare(path, controls):
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir("Qwen2.5-Math-1.5B"), local_files_only=True)
    dev = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/development")))
    baseline = [json.loads(line) for line in (controls / "greedy/baseline/shard_0.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for row, raw in zip(list(dev.select(range(8))), baseline[:8]):
        if row["source_index"] != raw["source_index"]:
            raise ValueError("Probe development IDs changed")
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        if hashlib.sha256(prompt.encode()).hexdigest() != raw["prompt_hash"]:
            raise ValueError("Probe prompt differs from baseline")
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion = tokenizer(raw["candidates"][0]["raw_solution"], add_special_tokens=False)["input_ids"][:256]
        if not completion:
            raise ValueError("Baseline probe has no answer tokens")
        count = min(16, len(completion))
        positions = sorted({len(prefix) - 1 + round(i * (len(completion) - 1) / max(count - 1, 1)) for i in range(count)})
        rows.append({"source_index": row["source_index"], "input_ids": prefix + completion, "positions": positions})
    payload = {"rows": rows, "vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(), "note": "Exact KL(policy||base) at fixed baseline-generated prefixes, up to256 completion tokens and16 positions/problem. Contexts are not sampled from the updated policy; this is diagnostic drift, not sequence-level or on-policy KL."}
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Refusing to change frozen drift probes")
    write_json(path, payload)
    print("Frozen eight development reference-drift probes")


def score(path, model_path, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import enable_full_determinism

    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Drift scoring requires BF16 CUDA")
    payload = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest() != payload["vocab_sha256"]:
        raise ValueError("Policy vocabulary differs from the reference")
    started = time.monotonic()
    models = [AutoModelForCausalLM.from_pretrained(p, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa").to("cuda").eval().requires_grad_(False) for p in (model_path, model_dir("Qwen2.5-Math-1.5B"))]
    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in payload["rows"]:
            ids = torch.tensor([row["input_ids"]], device="cuda")
            positions = torch.tensor(row["positions"], device="cuda")
            logits = [m(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=positions, use_cache=False).logits[0] for m in models]
            values = categorical_kl(*logits)
            results.append({"source_index": row["source_index"], "positions": len(values), "kl_sum": float(values.sum()), "mean_kl": float(values.mean())})
    write_json(output, {"model_path": str(model_path), "probe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mean_kl": sum(r["kl_sum"] for r in results) / sum(r["positions"] for r in results), "rows": results, "runtime_seconds": time.monotonic() - started, "definition": payload["note"]})
    print(f"Saved exact fixed-prefix reference drift to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_file", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ensure_scratch_path(args.probe_file)
    if args.prepare:
        prepare(args.probe_file, ensure_scratch_path(args.controls))
    else:
        score(args.probe_file, ensure_scratch_path(args.model_path), ensure_scratch_path(args.output))


if __name__ == "__main__":
    main()
