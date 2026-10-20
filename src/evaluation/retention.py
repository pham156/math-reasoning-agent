"""Fixed non-math retention diagnostics, excluded from math LR selection."""

import argparse
import hashlib
import json
import math
import random
import time

from src.core.config import data_dir, ensure_scratch_path, model_dir, print_storage_summary
from src.training.grpo.train import write_json


NAMES = ("baseline", "grpo", "dr_grpo", "dapo_style", "gspo")
SOURCES = {"wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"), "boolq": ("google/boolq", "default", "validation"), "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test")}


def answer_logprob(logits, prefix_length, candidate):
    import torch
    if prefix_length < 1 or not candidate:
        raise ValueError("Nonempty prefix and candidate required")
    positions = logits[0, prefix_length - 1:prefix_length + len(candidate) - 1].float()
    targets = torch.tensor(candidate, device=positions.device)
    return float(positions.log_softmax(-1).gather(-1, targets[:, None]).sum())


def prepare(root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    print_storage_summary()
    path = root / "frozen.json"
    if path.exists():
        print(f"Reusing immutable retention data: {path}")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_dir("Qwen2.5-Math-1.5B"), local_files_only=True)
    rng = random.Random(42)
    sources, datasets = {}, {}
    for name, (repo, config, split) in SOURCES.items():
        revision = HfApi().dataset_info(repo).sha
        dataset = load_dataset(repo, config, split=split, revision=revision)
        sources[name] = {"repo": repo, "config": config, "split": split, "revision": revision, "fingerprint": dataset._fingerprint}
        datasets[name] = dataset
    text = "\n\n".join(datasets["wikitext"]["text"])
    tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    blocks = sorted(rng.sample(range(len(tokens) // 512), 32))
    items = [{"id": f"wikitext:{i}", "task": "wikitext", "format": "raw", "input_ids": tokens[i * 512:(i + 1) * 512]} for i in blocks]
    manifest_indices = {"wikitext_blocks": blocks}
    for task in ("boolq", "arc_easy"):
        dataset = datasets[task]
        indices = sorted(rng.sample(range(len(dataset)), 100))
        manifest_indices[task] = indices
        for index in indices:
            row = dataset[index]
            if task == "boolq":
                question = f"Passage: {row['passage']}\nQuestion: {row['question']}"
                options, gold = ["no", "yes"], int(row["answer"])
            else:
                question = f"Question: {row['question']}"
                options = row["choices"]["text"]
                gold = row["choices"]["label"].index(row["answerKey"])
            letters = [chr(65 + i) for i in range(len(options))]
            question += "\n" + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, options))
            for style in ("raw", "chat"):
                if style == "raw":
                    prompt = question + "\nAnswer:"
                    candidates = [" " + letter for letter in letters]
                else:
                    prompt = tokenizer.apply_chat_template([{"role": "system", "content": "Answer the question by selecting the correct option. Reply with only its letter."}, {"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
                    candidates = letters
                prefix = tokenizer.encode(prompt, add_special_tokens=False)
                ids = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
                if any(len(c) != 1 for c in ids):
                    raise ValueError("Expected single-token option labels for unbiased length comparison")
                if max(len(prefix) + len(c) for c in ids) > 4096:
                    raise ValueError("Retention example exceeds context cap; no silent truncation")
                items.append({"id": f"{task}:{index}:{style}", "task": task, "format": style, "source_index": index, "prompt": prompt, "input_ids": prefix, "candidate_ids": ids, "options": options, "gold": gold})
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, {"sources": sources, "seed": 42, "tokenizer": str(model_dir("Qwen2.5-Math-1.5B")), "tokenizer_vocab_hash": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(), "selection": manifest_indices, "items": items, "purpose": "Frozen retention diagnostics only; never LR/checkpoint selection", "ppl_protocol": "32 seeded nonoverlapping 512-token WikiText-2 raw test blocks; first token not scored; not full-test benchmark perplexity", "classification_protocol": "Zero-shot sum conditional log likelihood of single-token option labels, raw and neutral chat. Not generated-answer/instruction-following accuracy."})
    print(f"Prepared {len(items)} items: 32 language blocks plus 100 questions x2 tasks x2 prompt formats")


def evaluate(root, name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import enable_full_determinism

    frozen_path = data_dir("rl_retention_v1/frozen.json")
    frozen = json.loads(frozen_path.read_text())
    model_path = model_dir("Qwen2.5-Math-1.5B") if name == "baseline" else root.parent / "pilot_20261013_000313/pilot" / name / "final_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest() != frozen["tokenizer_vocab_hash"]:
        raise ValueError("Tokenizer changed; fixed token IDs no longer comparable")
    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires BF16 CUDA GPU")
    root.mkdir(parents=True, exist_ok=True)
    config = {"name": name, "model_path": str(model_path), "data_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(), "precision": "FP32 parameters/BF16 compute", "score": "teacher-forced next-token likelihood; no generation, no stop-token dependence"}
    config_path = root / f"{name}_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume protocol changed")
    write_json(config_path, config)
    path = root / f"{name}.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if [r["id"] for r in existing] != [r["id"] for r in frozen["items"][:len(existing)]] or len(existing) > len(frozen["items"]):
        raise ValueError("Existing retention rows differ")
    if len(existing) == len(frozen["items"]):
        print("Already complete")
        return
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval()
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as stream:
        for item in frozen["items"][len(existing):]:
            if time.monotonic() - started >= 12600:
                print("Safe pause; rerun this task to resume")
                return
            result = {k: item[k] for k in ("id", "task", "format")}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if item["task"] == "wikitext":
                    inputs = torch.tensor([item["input_ids"]], device="cuda")
                    output = model(input_ids=inputs, labels=inputs, use_cache=False)
                    result.update(nll_sum=float(output.loss) * (inputs.shape[1] - 1), scored_tokens=inputs.shape[1] - 1)
                else:
                    scores = []
                    for candidate in item["candidate_ids"]:
                        inputs = torch.tensor([item["input_ids"] + candidate], device="cuda")
                        output = model(input_ids=inputs, use_cache=False)
                        scores.append(answer_logprob(output.logits, len(item["input_ids"]), candidate))
                    prediction = max(range(len(scores)), key=scores.__getitem__)
                    result.update(gold=item["gold"], prediction=prediction, correct=prediction == item["gold"], choice_logprobs=scores)
            if any(not math.isfinite(float(v)) for v in result.get("choice_logprobs", [result.get("nll_sum", 0)])):
                raise FloatingPointError("Nonfinite retention score")
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            existing.append(result)
            if len(existing) % 25 == 0:
                print(f"{name}: {len(existing)}/{len(frozen['items'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--name", choices=NAMES)
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    if args.prepare:
        prepare(data_dir("rl_retention_v1"))
    else:
        if not args.name or not args.output_dir:
            parser.error("Evaluation requires --name and --output_dir")
        evaluate(ensure_scratch_path(args.output_dir), args.name)


if __name__ == "__main__":
    main()
