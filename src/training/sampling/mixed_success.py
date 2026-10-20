from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_from_disk

from src.core.config import data_dir, model_dir, output_dir, print_storage_summary
from src.core.infer_baseline import build_prompt
from src.evaluation.answer import extract_final_answer
from src.evaluation.run_eval import LoadedGenerator
from src.training.rewards.math_correctness import final_answer_reward_one


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "outputs":
        return output_dir(*candidate.parts[1:])
    return candidate.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest of MATH train examples with mixed rollout rewards.")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--output", default=str(output_dir("grpo_stabilization/mixed_success_manifest.jsonl")))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print_storage_summary()

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(args.dataset_dir)["train"]
    dataset = dataset.map(lambda _row, index: {"source_index": index}, with_indices=True)
    dataset = dataset.shuffle(seed=args.sample_seed)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    generator = LoadedGenerator(args.model_path)
    kept = 0
    total_start = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for local_index, row in enumerate(dataset):
            problem = row.get("problem", "")
            answer = extract_final_answer(row.get("solution", ""))
            rewards = []
            answers = []
            generated_tokens = []
            for _ in range(args.num_rollouts):
                raw, meta = generator.generate(
                    build_prompt(problem),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                rewards.append(final_answer_reward_one(raw, answer))
                answers.append(extract_final_answer(raw))
                generated_tokens.append(meta.get("generated_tokens"))
            reward_sum = sum(rewards)
            is_mixed = 0 < reward_sum < len(rewards)
            payload = {
                "problem_index": int(row["source_index"]),
                "shuffled_local_index": local_index,
                "problem": problem,
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "ground_truth_answer": answer,
                "rollout_rewards": rewards,
                "rollout_answers": answers,
                "rollout_generated_tokens": generated_tokens,
                "reward_mean": reward_sum / len(rewards) if rewards else 0.0,
                "mixed_rewards": is_mixed,
            }
            if is_mixed:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if not args.quiet:
                print(f"[{local_index + 1}/{len(dataset)}] kept={kept} rewards={rewards}")

    summary = {
        "output": str(output_path),
        "num_scored": len(dataset),
        "num_kept": kept,
        "mixed_reward_group_rate": kept / len(dataset) if len(dataset) else 0.0,
        "num_rollouts": args.num_rollouts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "runtime_sec": time.time() - total_start,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
