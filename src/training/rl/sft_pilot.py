"""One-epoch full-parameter SFT control on exactly the RL pilot training IDs."""

import argparse
from pathlib import Path

from src.core.config import data_dir, ensure_scratch_path, model_dir
from src.training.grpo.train import write_json
from src.training.rl.smoke import conversation


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-1.5B")))
    parser.add_argument("--split_dir", default=str(data_dir("rl_1p5b_pilot_v1")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()
    for key in ("output_dir", "model_path", "split_dir", "dataset_dir"):
        ensure_scratch_path(getattr(args, key), key)
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
    records = []
    for row in shared:
        original = source[row["source_index"]]
        if original["problem"] != row["problem"]:
            raise ValueError("Source dataset changed")
        records.append(tokenize_solution(tokenizer, row["problem"], original["solution"]))
    write_json(root / "sft_config.json", {"model_path": args.model_path, "split_dir": args.split_dir, "training_ids": list(shared["source_index"]), "train_n": len(records), "epochs": 1, "learning_rate": 1e-5, "batch_size": 1, "gradient_accumulation_steps": 4, "max_length": 4096, "seed": 0, "loss": "assistant-only cross entropy", "precision": "FP32 parameters/BF16 compute", "comparison": "matched data/prompt/initialization; compute differs from RL"})
    if args.prepare_only:
        print(f"Validated {len(records)} exact-prompt SFT records; max length={max(len(r['input_ids']) for r in records)}")
        return
    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires a BF16 CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    config = TrainingArguments(output_dir=str(root / "checkpoints"), num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-5, lr_scheduler_type="constant", adam_beta1=0.9, adam_beta2=0.95, weight_decay=0, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_steps=32, save_total_limit=2, report_to="none", seed=0, full_determinism=True)
    checkpoints = sorted((root / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    trainer = Trainer(model=model, args=config, train_dataset=Dataset.from_list(records), data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100), processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    trainer.save_model(str(root / "final_model"))
    tokenizer.save_pretrained(str(root / "final_model"))
    write_json(root / "train_history.json", {"log_history": trainer.state.log_history, "metrics": result.metrics, "global_step": trainer.state.global_step})


if __name__ == "__main__":
    main()
