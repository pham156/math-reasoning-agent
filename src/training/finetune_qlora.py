from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from src.core.config import cache_dir, data_dir, ensure_scratch_path, model_dir, output_dir, print_storage_summary
from src.core.infer_baseline import build_prompt
from src.training.artifacts import save_training_artifacts


def format_example(example: dict) -> str:
    return f"### Problem:\n{example['problem']}\n\n### Solution:\n{example['solution']}"


def format_chat_eval_example(example: dict, tokenizer) -> str:
    prompt = build_prompt(example["problem"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["solution"]},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{prompt}\n{example['solution']}"


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
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt_format",
        choices=["legacy_math_sft", "chat_eval"],
        default="legacy_math_sft",
        help=(
            "legacy_math_sft uses the original plain ### Problem/### Solution text. "
            "chat_eval uses the same user prompt and tokenizer chat template as evaluation."
        ),
    )
    args = parser.parse_args()
    if args.resume_from_checkpoint == "":
        args.resume_from_checkpoint = None

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print_storage_summary()
    ensure_scratch_path(args.output_dir, "output_dir")

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
    model.gradient_checkpointing_enable()

    dataset = load_from_disk(args.dataset_dir)["train"]
    if args.max_train_samples and args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    if args.prompt_format == "chat_eval":
        dataset = dataset.map(
            lambda ex: {"text": format_chat_eval_example(ex, tokenizer)},
            remove_columns=dataset.column_names,
        )
    else:
        dataset = dataset.map(lambda ex: {"text": format_example(ex)}, remove_columns=dataset.column_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_params = inspect.signature(SFTConfig.__init__).parameters
    length_arg = "max_seq_length" if "max_seq_length" in sft_params else "max_length"
    training_kwargs = {
        length_arg: args.max_seq_length,
    }

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=["wandb"] if "WANDB_PROJECT" in __import__("os").environ else [],
        dataset_text_field="text",
        **training_kwargs,
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
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    artifacts = save_training_artifacts(args.output_dir, trainer.state.log_history, train_result.metrics)
    run_config_path = Path(args.output_dir) / "qlora_run_config.json"
    run_config = {
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "prompt_format": args.prompt_format,
        "max_train_samples": args.max_train_samples,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "seed": args.seed,
    }
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter to {args.output_dir}")
    print(f"Saved run config: {run_config_path}")
    for name, path in artifacts.items():
        if path:
            print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
