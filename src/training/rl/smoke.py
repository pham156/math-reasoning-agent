"""Small full-parameter reasoning training smoke test, not a benchmark run."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from src.core.config import data_dir, ensure_scratch_path, model_dir, output_dir, print_storage_summary
from src.evaluation.answer import extract_committed_answer, extract_final_answer
from src.training.grpo.train import summarize_history, write_json
from src.training.rl.methods import METHODS, check_method_config, method_settings
from src.training.rl.checkpoint_storage import checkpoint_due, save_limit
from src.training.rl.reward_design import batch_metrics, reward_components


SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def conversation(problem: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]


def epoch_repeat_sampler_class():
    import torch
    from trl.trainer.utils import RepeatSampler

    class EpochRepeatSampler(RepeatSampler):
        def __init__(self, *args, resume_epoch=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.resume_epoch = resume_epoch
            self.epoch = resume_epoch

        def set_epoch(self, epoch):
            if epoch < 0 or self.seed is None:
                raise ValueError("Repeated-pass sampling requires a seeded, nonnegative epoch")
            # Accelerate's skipped-batch loader starts its iteration counter at zero.
            self.epoch = max(epoch, self.resume_epoch)

        def __iter__(self):
            if self.shuffle:
                # Replay the private generator's earlier permutations without changing global RNG.
                self.generator.manual_seed(self.seed)
                for _ in range(self.epoch):
                    torch.randperm(self.num_samples, generator=self.generator)
            return super().__iter__()

    return EpochRepeatSampler


def smoke_trainer_class():
    from trl import GRPOTrainer

    if importlib.metadata.version("trl") != "0.25.1":
        raise RuntimeError("Use the pinned smoke runtime: this adapter is validated for TRL 0.25.1")

    class SmokeTrainer(GRPOTrainer):
        def _get_train_sampler(self, dataset=None):
            sampler = super()._get_train_sampler(dataset)
            if not getattr(self, "epoch_addressed_sampler", False):
                return sampler
            return epoch_repeat_sampler_class()(
                data_source=sampler.data_source, mini_repeat_count=sampler.mini_repeat_count,
                batch_size=sampler.batch_size, repeat_count=sampler.repeat_count,
                shuffle=sampler.shuffle, seed=sampler.seed,
                resume_epoch=self._step // (len(sampler.data_source) * sampler.mini_repeat_count),
            )

        def _compute_loss(self, model, inputs):
            # TRL 0.25.1's DAPO denominator spans generation, even when several updates consume it.
            if self.loss_type == "dapo" and model.training:
                inputs = dict(inputs)
                inputs["num_items_in_batch"] = inputs["num_items_in_batch"] * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            return super()._compute_loss(model, inputs)

    return SmokeTrainer


STRICT_GRADER = "closed-boxed-v1"


@lru_cache(maxsize=4096)
def grade_solution(text: str, answer: str) -> float:
    """Historical lenient grader used by the pilot reward."""
    return answers_match(extract_final_answer(text), answer)


@lru_cache(maxsize=4096)
def grade_solution_strict(text: str, answer: str) -> float:
    """Reporting-only grader requiring a closed final boxed answer."""
    return answers_match(extract_committed_answer(text) or "", answer)


def answers_match(prediction: str, answer: str) -> float:
    from math_verify import parse, verify

    if not prediction.strip() or not answer.strip():
        return 0.0
    if "".join(prediction.split()) == "".join(answer.split()):
        return 1.0
    try:
        gold = parse(r"\boxed{" + answer + "}")
        candidate = parse(r"\boxed{" + prediction + "}")
        return float(bool(gold and candidate and verify(gold, candidate)))
    except Exception:
        return 0.0


def prepare_dataset(args, tokenizer, run_dir: Path):
    from datasets import Dataset, load_from_disk
    from math_verify import parse

    original = load_from_disk(args.dataset_dir)["train"]
    indices = list(range(len(original)))
    random.Random(args.seed).shuffle(indices)
    rows = []
    for index in indices:
        row = original[index]
        if row.get("level") not in {"Level 3", "Level 4", "Level 5"}:
            continue
        answer = extract_final_answer(row["solution"])
        if not answer.strip() or not parse(r"\boxed{" + answer + "}"):
            continue
        prompt = conversation(row["problem"])
        serialized = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(serialized, add_special_tokens=False)) > args.max_prompt_length:
            continue
        rows.append({"prompt": prompt, "problem": row["problem"], "answer": answer, "type": row["type"], "level": row["level"], "source_index": index})
        if len(rows) == args.train_samples + args.eval_samples:
            break
    if len(rows) < args.train_samples + args.eval_samples:
        raise ValueError("Not enough parseable Level 3-5 MATH examples for the smoke slice")
    manifest = [{"source_index": row["source_index"], "type": row["type"], "level": row["level"], "problem_hash": hashlib.sha256(row["problem"].encode()).hexdigest()} for row in rows]
    write_json(run_dir / "dataset_manifest.json", {"dataset": args.dataset_dir, "source_split": "train", "seed": args.seed, "training": manifest[:args.train_samples], "held_out_smoke": manifest[args.train_samples:]})
    train = Dataset.from_list(rows[:args.train_samples])
    train.save_to_disk(str(run_dir / "train_dataset"))
    return train, rows[args.train_samples:]


def evaluate(model, tokenizer, rows: list[dict], cap: int) -> list[dict]:
    import torch

    results = []
    was_training = model.training
    previous_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=cap, pad_token_id=tokenizer.pad_token_id)
        completion = generated[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(completion, skip_special_tokens=True)
        results.append({"source_index": row["source_index"], "problem": row["problem"], "reference_answer": row["answer"], "raw_solution": text, "extracted_answer": extract_final_answer(text), "correct": bool(grade_solution(text, row["answer"])), "generated_tokens": len(completion), "truncated": len(completion) >= cap and int(completion[-1]) != tokenizer.eos_token_id, "runtime_seconds": time.monotonic() - started})
    model.config.use_cache = previous_cache
    model.train(was_training)
    return results


def run(args, run_dir: Path, dataset_preparer=prepare_dataset) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from transformers.trainer_utils import enable_full_determinism
    from trl import GRPOConfig

    deterministic = getattr(args, "deterministic_training", False)
    if deterministic:
        enable_full_determinism(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        set_seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required; submit the Gilbreth smoke Slurm script")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, padding_side="left")
    if not tokenizer.chat_template:
        raise ValueError("Qwen tokenizer is missing its chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # FP32 trainable weights preserve small updates; forward/backward compute uses BF16 autocast.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    if args.max_prompt_length + args.max_completion_length > model.config.max_position_embeddings:
        raise ValueError("Prompt and completion caps exceed the model context window")
    model.config.use_cache = False
    train, held_out = dataset_preparer(args, tokenizer, run_dir)
    resume = getattr(args, "resume_from_checkpoint", None)
    baseline_path = run_dir / "baseline.json"
    if resume:
        baseline = json.loads(baseline_path.read_text())["examples"]
    else:
        baseline = evaluate(model, tokenizer, held_out, args.max_completion_length)
        write_json(baseline_path, {"examples": baseline})
    generations = getattr(args, "num_generations", 4)
    generation_batch = args.gradient_accumulation_steps * generations
    clip_lower, clip_upper = getattr(args, "clip_lower", None), getattr(args, "clip_upper", None)
    objective = method_settings(args.algorithm, clip_lower, clip_upper)
    beta = getattr(args, "kl_beta", 0.0)
    checkpoint_limit = getattr(args, "checkpoint_limit", None) or save_limit(run_dir)
    reference_kwargs = {"dtype": "float32", "attn_implementation": "sdpa", "local_files_only": True} if beta else None
    config = GRPOConfig(
        output_dir=str(run_dir / "checkpoints"), max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=generation_batch, num_generations=generations, num_iterations=1,
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        learning_rate=getattr(args, "learning_rate", 1e-6), lr_scheduler_type="constant", warmup_steps=0,
        adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.0, max_grad_norm=1.0,
        temperature=getattr(args, "temperature", 1.0), top_p=1.0, top_k=0, beta=beta,
        model_init_kwargs=reference_kwargs,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no" if checkpoint_limit == 1 else "steps",
        save_steps=getattr(args, "save_steps", args.max_steps), save_total_limit=checkpoint_limit,
        report_to="none", push_to_hub=False, use_vllm=False, seed=args.seed, full_determinism=deterministic,
        mask_truncated_completions=False, **objective,
    )
    check_method_config(args.algorithm, config, clip_lower, clip_upper)
    if config.steps_per_generation != generation_batch or config.steps_per_generation // args.gradient_accumulation_steps != generations:
        raise ValueError("Unexpected rollout boundary in installed TRL configuration")
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "trl", "datasets", "accelerate", "math-verify"]}
    metadata = {"algorithm": args.algorithm, "status": "running", "arguments": vars(args), "objective": objective, "kl_beta": beta, "reward": "R0: extracted final-answer exact/Math-Verify correctness", "precision": {"trainable_parameters": "float32", "compute": "bfloat16", "peft": False, "quantization": False}, "gpu": torch.cuda.get_device_name(), "versions": versions, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "generation_batch_responses": generation_batch, "optimizer_batch_responses": args.gradient_accumulation_steps}
    metadata["dapo_normalizer_adapter"] = args.algorithm == "dapo_style"
    metadata["reward_design"] = getattr(args, "reward_design", "correctness")
    metadata["reward"] = metadata["reward_design"]
    metadata["rollout_boundary_updates"] = generations
    metadata["sampler"] = "epoch_addressed_repeat_v1" if getattr(args, "training_passes", 1) > 1 else "legacy_repeat"
    metadata["determinism"] = {"requested": deterministic, "algorithms_enabled": torch.are_deterministic_algorithms_enabled(), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "cudnn_benchmark": torch.backends.cudnn.benchmark, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32}
    revision_file = Path(args.model_path) / ".cache/huggingface/download/config.json.metadata"
    if revision_file.exists():
        metadata["model_revision"] = revision_file.read_text().splitlines()[0]
    write_json(run_dir / "run_config.json", metadata)
    write_json(run_dir / "training_args.json", config.to_dict())
    monitor = next(parameter for name, parameter in model.named_parameters() if name.endswith("self_attn.q_proj.weight"))
    before = monitor.detach().flatten()[:4096].float().cpu().clone()

    def accuracy_reward(completions, answer, problem, completion_ids, **kwargs):
        from math_verify import parse

        def parseable(value):
            try:
                return bool(parse(r"\boxed{" + value + "}"))
            except Exception:
                return False

        records = []
        with (run_dir / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
            for completion, reference, question, ids in zip(completions, answer, problem, completion_ids, strict=True):
                text = completion if isinstance(completion, str) else "\n".join(turn["content"] for turn in completion)
                components = reward_components(text, grade_solution(text, reference), len(ids), args.max_completion_length, metadata["reward_design"], parseable)
                record = {"problem": question, "reference_answer": reference, "completion": text, "extracted_answer": extract_final_answer(text), **components}
                records.append(record)
                stream.write(json.dumps(record) + "\n")
        metrics = batch_metrics(records, generations)
        for key, value in metrics.items():
            trainer._metrics["train"]["reward_components/" + key].append(value)
        with (run_dir / "reward_groups.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": trainer.state.global_step, "responses": len(records), "groups": len(records) // generations, **metrics}) + "\n")
        return [r["reward"] for r in records]

    class FiniteMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm", "kl", "entropy", "clip_ratio/region_mean"):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Non-finite {key} at step {state.global_step}")

        def on_step_end(self, args, state, control, **kwargs):
            deadline = getattr(run_args, "segment_seconds", 0)
            stop_step = getattr(run_args, "stop_after_step", 0)
            stop = (deadline and time.monotonic() - started >= deadline) or (stop_step and state.global_step >= stop_step)
            # Stop only after every slice of the current rollout batch has been consumed.
            if checkpoint_due(state.global_step, args.max_steps, generations, stop):
                control.should_save = True
                if stop:
                    control.should_training_stop = True
            return control

    run_args = args
    trainer = smoke_trainer_class()(model=model, args=config, reward_funcs=accuracy_reward, train_dataset=train, processing_class=tokenizer, callbacks=[FiniteMetrics()])
    trainer.epoch_addressed_sampler = getattr(args, "training_passes", 1) > 1
    if beta:
        if trainer.ref_model is None:
            raise RuntimeError("Nonzero KL requires a fixed reference model")
        trainer.ref_model.requires_grad_(False)
        trainer.ref_model.eval()
        if any(p.dtype != torch.float32 for p in trainer.ref_model.parameters()):
            raise RuntimeError("Reference policy must use FP32 weights like the trainable policy")
        metadata["reference"] = {"model_path": args.model_path, "frozen": True, "precision": "FP32 weights/BF16 compute"}
        write_json(run_dir / "run_config.json", metadata)
    if resume:
        saved_step = json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"]
        if saved_step % generations:
            raise ValueError("Continuation requires a complete rollout-boundary checkpoint")
        trainer._step = saved_step * args.gradient_accumulation_steps
    torch.cuda.reset_peak_memory_stats()
    trained = trainer.train(resume_from_checkpoint=resume)
    if beta and (any(p.requires_grad for p in trainer.ref_model.parameters()) or not any("kl" in row for row in trainer.state.log_history)):
        raise RuntimeError("KL reference is not frozen or measured training KL is missing")
    delta = float((monitor.detach().flatten()[:4096].float().cpu() - before).abs().max())
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{trainer.state.global_step}"
    if not (checkpoint / "trainer_state.json").exists():
        raise RuntimeError("Training did not reach its target or save a resumable Trainer checkpoint")
    write_json(run_dir / "train_history.json", {"log_history": trainer.state.log_history})
    if trainer.state.global_step < args.max_steps:
        write_json(run_dir / "progress.json", {"status": "paused", "global_step": trainer.state.global_step, "target_steps": args.max_steps, "checkpoint": str(checkpoint)})
        return
    final_model = run_dir / "final_model"
    trainer.save_model(str(final_model))
    tokenizer.save_pretrained(str(final_model))
    if not (final_model / "config.json").exists() or not list(final_model.glob("*.safetensors")):
        raise RuntimeError("Full-parameter model save is incomplete")
    after = evaluate(trainer.model, tokenizer, held_out, args.max_completion_length)
    write_json(run_dir / "after.json", {"examples": after})
    write_json(run_dir / "train_history.json", {"log_history": trainer.state.log_history})
    metadata.update({"status": "passed", "global_step": trainer.state.global_step, "train_metrics": trained.metrics, "history_summary": summarize_history(trainer.state.log_history), "monitored_parameter_max_change": delta, "policy_update_observed": delta > 0, "baseline_correct": sum(row["correct"] for row in baseline), "after_correct": sum(row["correct"] for row in after), "evaluation_n": len(after), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "runtime_seconds": time.monotonic() - started, "checkpoint": str(checkpoint), "final_model": str(final_model), "limitation": "Tiny setup test; no benchmark accuracy or exact interrupted-resume claim"})
    metadata["limitation"] = getattr(args, "purpose", metadata["limitation"])
    write_json(run_dir / "smoke_result.json", metadata)
    write_json(run_dir / "progress.json", {"status": "completed", "global_step": trainer.state.global_step, "checkpoint": str(checkpoint)})
    print(json.dumps({key: metadata[key] for key in ["algorithm", "status", "global_step", "policy_update_observed", "runtime_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=list(METHODS), required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-1.5B")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--output_dir", default=str(output_dir("rl_1p5b_study/smoke")))
    parser.add_argument("--train_samples", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print_storage_summary()
    run_dir = ensure_scratch_path(args.output_dir, "smoke output_dir")
    ensure_scratch_path(args.model_path, "smoke model_path")
    ensure_scratch_path(args.dataset_dir, "smoke dataset_dir")
    if min(args.train_samples, args.eval_samples, args.max_steps, args.gradient_accumulation_steps) < 1 or args.gradient_accumulation_steps % 4:
        parser.error("Samples/steps must be positive and accumulation must be divisible by four")
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error("Use an empty run directory; existing smoke artifacts will not be overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args, run_dir)
    except Exception as exc:
        write_json(run_dir / "smoke_result.json", {"algorithm": args.algorithm, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "arguments": vars(args)})
        raise


if __name__ == "__main__":
    main()
