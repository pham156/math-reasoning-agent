"""Matched-data SFT controls on exactly the RL pilot training IDs.

Defaults reproduce the original one-epoch full-parameter ground-truth control. Variants:
``--data rft`` trains on verified base-model samples (see ``src.training.rl.rft_data``),
and ``--lora_rank`` trains a PEFT LoRA adapter that is merged into FP32 weights for
evaluation. LoRA imports ``peft`` from the isolated ``$SCRATCH_ROOT/env/peft-overlay`` package directory.
"""

import argparse
import json
from pathlib import Path

from src.core.config import data_dir, ensure_scratch_path, model_dir
from src.training.grpo.train import write_json
from src.training.rl.smoke import conversation

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
STOP_IDS = [151643, 151645]


def tokenize_solution(tokenizer, problem, solution, cap=4096):
    prompt = tokenizer.apply_chat_template(conversation(problem), tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(conversation(problem) + [{"role": "assistant", "content": solution}], tokenize=True, add_generation_prompt=False)
    if complete[:len(prompt)] != prompt:
        raise ValueError("Training prompt is not the exact inference prefix")
    if len(complete) > cap:
        raise ValueError("SFT example exceeds cap; do not silently drop/truncate shared examples")
    if len(complete) <= len(prompt):
        raise ValueError("Empty supervised response")
    return {"input_ids": complete, "attention_mask": [1] * len(complete), "labels": [-100] * len(prompt) + complete[len(prompt):]}


def training_pairs(args, shared, source):
    train_ids = set(shared["source_index"])
    if args.data == "ground_truth":
        pairs = []
        for row in shared:
            original = source[row["source_index"]]
            if original["problem"] != row["problem"]:
                raise ValueError("Source dataset changed")
            pairs.append((row["problem"], original["solution"]))
        return pairs, {"data": "MATH reference solutions (human-written)"}
    payload = json.loads(ensure_scratch_path(args.rft_dataset, "rft_dataset").read_text())
    problems = {r["source_index"]: r["problem"] for r in shared}
    pairs = []
    for row in payload["rows"]:
        if row["source_index"] not in train_ids or problems[row["source_index"]] != row["problem"]:
            raise ValueError("RFT target outside the frozen training IDs")
        pairs.append((row["problem"], row["solution"]))
    return pairs, {"data": "RFT: verified base-model samples", "rft_stats": payload["stats"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-1.5B")))
    parser.add_argument("--split_dir", default=str(data_dir("rl_1p5b_pilot_v1")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--data", choices=["ground_truth", "rft"], default="ground_truth")
    parser.add_argument("--rft_dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=0, help="0 = full-parameter training")
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()
    for key in ("output_dir", "model_path", "split_dir", "dataset_dir"):
        ensure_scratch_path(getattr(args, key), key)
    if (args.data == "rft") != bool(args.rft_dataset):
        parser.error("--rft_dataset is required exactly when --data rft")
    import torch
    from datasets import Dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from transformers.trainer_utils import enable_full_determinism

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    shared = load_from_disk(str(Path(args.split_dir) / "train"))
    source = load_from_disk(args.dataset_dir)["train"]
    if len(shared) != 512:
        raise ValueError("Expected 512 frozen training problems")
    pairs, provenance = training_pairs(args, shared, source)
    records = [tokenize_solution(tokenizer, problem, solution) for problem, solution in pairs]
    lora = {"rank": args.lora_rank, "alpha": args.lora_alpha or 2 * args.lora_rank, "dropout": 0.0, "targets": LORA_TARGETS, "merged_for_evaluation": True} if args.lora_rank else None
    config_record = {"model_path": args.model_path, "split_dir": args.split_dir, "training_ids": list(shared["source_index"]), "train_n": len(records), "epochs": args.epochs, "learning_rate": args.learning_rate, "lr_scheduler": args.lr_scheduler, "warmup_ratio": args.warmup_ratio, "batch_size": 1, "gradient_accumulation_steps": args.gradient_accumulation_steps, "max_length": 4096, "seed": 0, "loss": "assistant-only cross entropy (target ends with <|im_end|>)", "precision": "FP32 parameters/BF16 compute", "comparison": "matched problems/prompt/initialization; compute differs from RL", **provenance}
    if lora:
        config_record["lora"] = lora
    config_path = root / "sft_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config_record:
        raise ValueError("Refusing to resume an SFT run with a changed configuration")
    write_json(config_path, config_record)
    if args.prepare_only:
        lengths = [len(r["input_ids"]) for r in records]
        print(f"Validated {len(records)} exact-prompt SFT records ({provenance['data']}); max length={max(lengths)}, mean={sum(lengths) / len(lengths):.0f}")
        return
    if (root / "final_model" / "config.json").exists() and (root / "train_history.json").exists():
        print("SFT already completed; no-op")
        return
    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires a BF16 CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    if lora:
        import os
        import sys

        # peft lives in an isolated scratch overlay so the validated RL runtime is unchanged.
        overlay = Path(os.environ["SCRATCH_ROOT"]) / "env/peft-overlay"
        if str(overlay) not in sys.path:
            sys.path.append(str(overlay))
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    steps_per_epoch = -(-len(records) // args.gradient_accumulation_steps)
    save_steps = max(1, steps_per_epoch // 2)
    config = TrainingArguments(output_dir=str(root / "checkpoints"), num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler, warmup_ratio=args.warmup_ratio, adam_beta1=0.9, adam_beta2=0.95, weight_decay=0, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_steps=save_steps if args.data == "rft" or lora else 32, save_total_limit=1 if args.data == "rft" or lora else 2, report_to="none", seed=0, full_determinism=True, save_safetensors=not lora)
    checkpoints = sorted((root / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    trainer = Trainer(model=model, args=config, train_dataset=Dataset.from_list(records), data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100), processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    final = trainer.model
    if lora:
        trainer.model.save_pretrained(str(root / "adapter"))
        final = trainer.model.merge_and_unload()
    final.config.use_cache = True
    final.generation_config.eos_token_id = STOP_IDS
    final.save_pretrained(str(root / "final_model"))
    tokenizer.save_pretrained(str(root / "final_model"))
    write_json(root / "train_history.json", {"log_history": trainer.state.log_history, "metrics": result.metrics, "global_step": trainer.state.global_step})


if __name__ == "__main__":
    main()
