from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.core.config import cache_dir, data_dir, ensure_scratch_path, model_dir, output_dir, print_storage_summary
from src.core.infer_baseline import build_prompt
from src.evaluation.answer import extract_final_answer
from src.training.rewards.math_correctness import final_answer_reward


def trl_capability() -> dict:
    try:
        import trl
    except Exception as exc:
        return {
            "trl_importable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "has_grpo_trainer": False,
            "has_grpo_config": False,
        }
    return {
        "trl_importable": True,
        "trl_version": getattr(trl, "__version__", "unknown"),
        "has_grpo_trainer": hasattr(trl, "GRPOTrainer"),
        "has_grpo_config": hasattr(trl, "GRPOConfig"),
        "grpo_trainer_signature": str(inspect.signature(trl.GRPOTrainer.__init__))
        if hasattr(trl, "GRPOTrainer")
        else None,
        "grpo_config_params": list(inspect.signature(trl.GRPOConfig.__init__).parameters)
        if hasattr(trl, "GRPOConfig")
        else [],
    }


def write_json(path: str | Path, payload: dict) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def _parse_level_number(value: Any) -> int | None:
    text = str(value or "").strip().lower().replace("level", "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _load_allowed_indices(path: str | None) -> set[int] | None:
    if not path:
        return None
    allowed: set[int] = set()
    manifest_path = Path(path).expanduser()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "problem_index" in row:
                allowed.add(int(row["problem_index"]))
            elif "source_index" in row:
                allowed.add(int(row["source_index"]))
    return allowed


def make_train_dataset(
    dataset_dir: str,
    max_train_samples: int,
    seed: int,
    min_level: int | None = None,
    max_level: int | None = None,
    allowed_indices_path: str | None = None,
):
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_dir)["train"]
    allowed_indices = _load_allowed_indices(allowed_indices_path)

    def add_source_index(example: dict, index: int) -> dict:
        result = dict(example)
        result["source_index"] = index
        return result

    dataset = dataset.map(add_source_index, with_indices=True)
    original_count = len(dataset)
    if min_level is not None or max_level is not None:
        min_value = min_level if min_level is not None else -10**9
        max_value = max_level if max_level is not None else 10**9
        dataset = dataset.filter(
            lambda row: (level := _parse_level_number(row.get("level"))) is not None
            and min_value <= level <= max_value
        )
    if allowed_indices is not None:
        dataset = dataset.filter(lambda row: int(row["source_index"]) in allowed_indices)
    filtered_count = len(dataset)
    if max_train_samples > 0:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_train_samples, len(dataset))))

    def format_row(example: dict) -> dict:
        return {
            "prompt": [{"role": "user", "content": build_prompt(example["problem"])}],
            "answer": extract_final_answer(example["solution"]),
            "problem": example["problem"],
            "solution": example["solution"],
            "type": example.get("type", ""),
            "level": example.get("level", ""),
            "source_index": example.get("source_index"),
        }

    keep = ["prompt", "answer", "problem", "solution", "type", "level", "source_index"]
    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    dataset = dataset.select_columns([column for column in keep if column in dataset.column_names])
    dataset.info.description = json.dumps(
        {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "selected_count": len(dataset),
            "min_level": min_level,
            "max_level": max_level,
            "allowed_indices_path": allowed_indices_path,
            "allowed_indices_count": len(allowed_indices) if allowed_indices is not None else None,
        },
        ensure_ascii=False,
    )
    return dataset


def summarize_history(log_history: list[dict]) -> dict:
    rewards = [
        float(entry[key])
        for entry in log_history
        for key in ("reward", "rewards/mean", "train/reward", "reward_mean")
        if key in entry and entry[key] is not None
    ]
    lengths = [
        float(entry[key])
        for entry in log_history
        for key in ("completion_length", "response_length", "completions/mean_length", "train/completion_length")
        if key in entry and entry[key] is not None
    ]
    kls = [
        float(entry[key])
        for entry in log_history
        for key in ("kl", "objective/kl", "train/kl")
        if key in entry and entry[key] is not None
    ]
    zero_variance_rate = _history_metric_mean(log_history, ("frac_reward_zero_std",))
    return {
        "num_log_entries": len(log_history),
        "reward_mean": mean(rewards) if rewards else None,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0 if rewards else None,
        "response_length_mean": mean(lengths) if lengths else None,
        "kl_mean": mean(kls) if kls else None,
        "reward_group_std_mean": _history_metric_mean(log_history, ("reward_std", "rewards/std")),
        "zero_variance_group_rate": zero_variance_rate,
        "mixed_reward_group_rate": 1.0 - zero_variance_rate if zero_variance_rate is not None else None,
        "entropy_mean": _history_metric_mean(log_history, ("entropy",)),
        "grad_norm_mean": _history_metric_mean(log_history, ("grad_norm",)),
    }


def _history_metric_mean(log_history: list[dict], keys: tuple[str, ...]) -> float | None:
    values = [float(entry[key]) for entry in log_history for key in keys if entry.get(key) is not None]
    return mean(values) if values else None


def save_curve_csv(path: Path, log_history: list[dict]) -> None:
    preferred = ["step", "loss", "reward", "reward_std", "frac_reward_zero_std", "completions/mean_length", "kl", "entropy", "grad_norm", "learning_rate", "epoch"]
    discovered = sorted({key for entry in log_history for key in entry})
    keys = preferred + [key for key in discovered if key not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for entry in log_history:
            f.write(",".join(str(entry.get(key, "")) for key in keys) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Small RLVR/GRPO run with final-answer rewards.")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--output_dir", default=str(output_dir("grpo_rlvr_qwen2.5_math7b_instruct_math500")))
    parser.add_argument("--max_train_samples", type=int, default=500)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0, help="Reference-policy KL coefficient.")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_level", type=int, default=None)
    parser.add_argument("--max_level", type=int, default=None)
    parser.add_argument("--allowed_indices_jsonl", default=None)
    parser.add_argument("--capability_only", action="store_true")
    args = parser.parse_args()

    print_storage_summary()
    output_path = ensure_scratch_path(args.output_dir, "output_dir")
    output_path.mkdir(parents=True, exist_ok=True)

    capability = trl_capability()
    capability["requested_experiment"] = vars(args)
    write_json(output_path / "trl_grpo_capability.json", capability)
    if args.capability_only:
        print(json.dumps(capability, indent=2))
        return
    if not capability.get("has_grpo_trainer") or not capability.get("has_grpo_config"):
        raise RuntimeError(
            "Installed TRL does not expose GRPOTrainer/GRPOConfig. "
            f"Wrote capability report to {output_path / 'trl_grpo_capability.json'}. "
            "Upgrade the project environment to a GRPO-capable TRL before launching RLVR."
        )

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=str(cache_dir() / "huggingface"),
        trust_remote_code=True,
        padding_side="left",
    )
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

    train_dataset = make_train_dataset(
        args.dataset_dir,
        args.max_train_samples,
        args.seed,
        min_level=args.min_level,
        max_level=args.max_level,
        allowed_indices_path=args.allowed_indices_jsonl,
    )
    train_dataset.save_to_disk(str(output_path / "train_dataset_snapshot"))
    try:
        dataset_filter_metadata = json.loads(train_dataset.info.description or "{}")
    except json.JSONDecodeError:
        dataset_filter_metadata = {}

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config_params = inspect.signature(GRPOConfig.__init__).parameters
    requested_config = {
        "output_dir": str(output_path),
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "seed": args.seed,
        "report_to": [],
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
    }
    grpo_kwargs = {key: value for key, value in requested_config.items() if key in config_params}
    training_args = GRPOConfig(**grpo_kwargs)

    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "reward_funcs": final_answer_reward,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    log_history = [dict(entry) for entry in trainer.state.log_history]
    write_json(output_path / "train_history.json", {"log_history": log_history})
    write_json(
        output_path / "train_summary.json",
        {
            "capability": capability,
            "train_metrics": train_result.metrics,
            "history_summary": summarize_history(log_history),
            "dataset_filter_metadata": dataset_filter_metadata,
        },
    )
    save_curve_csv(output_path / "train_curves.csv", log_history)
    print(f"Saved GRPO/RLVR adapter and training artifacts to {output_path}")


if __name__ == "__main__":
    main()
