from __future__ import annotations

import argparse
import inspect

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.config import cache_dir, data_dir, model_dir, output_dir, print_storage_summary


def format_example(example: dict) -> str:
    return f"### Problem:\n{example['problem']}\n\n### Solution:\n{example['solution']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for Qwen2.5-Math-7B on MATH.")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--output_dir", default=str(output_dir("qwen2.5-math-7b-math-lora")))
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_train_samples", type=int, default=0)
    args = parser.parse_args()

    print_storage_summary()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=str(cache_dir() / "huggingface"), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map="auto",
        cache_dir=str(cache_dir() / "huggingface"),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    dataset = load_from_disk(args.dataset_dir)["train"]
    if args.max_train_samples and args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    dataset = dataset.map(lambda ex: {"text": format_example(ex)}, remove_columns=dataset.column_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=["wandb"] if "WANDB_PROJECT" in __import__("os").environ else [],
        dataset_text_field="text",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
