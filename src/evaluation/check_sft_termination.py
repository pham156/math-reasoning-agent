"""Audit SFT target terminators and ignored chat-end tokens in greedy outputs."""

import argparse
import json

from src.core.config import data_dir, ensure_scratch_path
from src.training.grpo.train import write_json
from src.training.rl.sft_pilot import tokenize_solution
from src.training.rl.smoke import grade_solution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 20:
        parser.error("Use a small diagnostic, limit 1-20")
    root = ensure_scratch_path(args.output_dir)
    model_path = ensure_scratch_path(args.model_path)
    root.mkdir(parents=True, exist_ok=True)
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import enable_full_determinism

    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    chat_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    config = json.loads((model_path / "generation_config.json").read_text())
    eos = config["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    training = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
    source = load_from_disk(str(data_dir("MATH")))["train"]
    tails = [tokenize_solution(tokenizer, row["problem"], source[row["source_index"]]["solution"])["input_ids"][-2:] for row in training]
    audit = {"generation_eos_ids": eos, "tokenizer_eos_id": tokenizer.eos_token_id, "chat_end_id": chat_end, "training_n": len(tails), "targets_ending_with_chat_end": sum(t[-2] == chat_end or t[-1] == chat_end for t in tails), "targets_ending_with_generation_eos": sum(t[-1] in eos for t in tails), "tail_example": tails[0], "tail_tokens": tokenizer.convert_ids_to_tokens(tails[0])}
    write_json(root / "token_audit.json", audit)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval()
    rows = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/development"))).select(range(args.limit))
    results = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sequence = model.generate(**inputs, do_sample=False, max_new_tokens=3000, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
        tokens = sequence[0, inputs["input_ids"].shape[1]:].tolist()
        first_chat_end = tokens.index(chat_end) if chat_end in tokens else None
        trimmed = tokens if first_chat_end is None else tokens[:first_chat_end + 1]
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        stopped = tokenizer.decode(trimmed, skip_special_tokens=True)
        result = {"source_index": row["source_index"], "problem": row["problem"], "reference_answer": row["answer"], "token_ids": tokens, "raw_with_special_tokens": tokenizer.decode(tokens, skip_special_tokens=False), "generated_tokens": len(tokens), "first_chat_end_position": first_chat_end, "tokens_after_chat_end": 0 if first_chat_end is None else len(tokens) - first_chat_end - 1, "original_correct": bool(grade_solution(raw, row["answer"])), "chat_stop_correct": bool(grade_solution(stopped, row["answer"])), "original_truncated": len(tokens) == 3000 and tokens[-1] not in eos, "chat_stop_truncated": len(trimmed) == 3000 and trimmed[-1] not in [*eos, chat_end]}
        results.append(result)
        write_json(root / "generations.json", {"examples": results})
        print(json.dumps({k: v for k, v in result.items() if k not in {"token_ids", "raw_with_special_tokens", "problem"}}), flush=True)
    summary = {"audit": audit, "n": len(results), "outputs_with_chat_end": sum(r["first_chat_end_position"] is not None for r in results), "outputs_continuing_after_chat_end": sum(r["tokens_after_chat_end"] > 0 for r in results), "original_correct": sum(r["original_correct"] for r in results), "chat_stop_correct": sum(r["chat_stop_correct"] for r in results), "original_truncated": sum(r["original_truncated"] for r in results), "chat_stop_truncated": sum(r["chat_stop_truncated"] for r in results), "interpretation": "Greedy generation with an added chat-end stop would keep the same prefix. If no chat-end token is emitted, adding that stop token cannot fix that output. Small diagnostic only; existing evaluation/model unchanged."}
    write_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
