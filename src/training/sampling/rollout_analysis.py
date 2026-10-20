from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from src.core.config import model_dir, output_dir
from src.core.infer_baseline import build_prompt
from src.evaluation.answer import extract_final_answer
from src.evaluation.run_eval import LoadedGenerator
from src.training.rewards.math_correctness import canonical_final_answer, final_answer_reward_one
from src.training.sampling.math_subsets import build_subset


def _resolve_output(value: str) -> Path:
    path = Path(value)
    return output_dir() / Path(*path.parts[1:]) if path.parts and path.parts[0] == "outputs" else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen-policy mathematical exploration.")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--subset", choices=["math_random", "math_level_1_2", "math_level_3_5", "gsm8k"], default="math_random")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_subset(args.subset, args.limit, args.seed)
    generator = LoadedGenerator(args.model_path)
    target = _resolve_output(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    group_rows = []
    started = time.time()
    with target.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            reference = extract_final_answer(row.get("solution", row.get("answer", "")))
            rollouts = []
            for rollout_index in range(args.num_rollouts):
                raw, metadata = generator.generate(build_prompt(row["problem"] if "problem" in row else row["question"]), max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
                extracted = extract_final_answer(raw)
                rollouts.append({"rollout": rollout_index, "raw": raw, "answer": extracted, "normalized_answer": canonical_final_answer(extracted), "correct": bool(final_answer_reward_one(raw, reference)), "generated_tokens": metadata.get("generated_tokens")})
            correct = sum(item["correct"] for item in rollouts)
            answer_counts = Counter(item["normalized_answer"] for item in rollouts)
            payload = {"index": index, "subset": args.subset, "type": row.get("type", ""), "level": row.get("level", ""), "problem": row.get("problem", row.get("question", "")), "reference_answer": reference, "temperature": args.temperature, "num_rollouts": args.num_rollouts, "rollouts": rollouts, "num_correct": correct, "any_correct": correct > 0, "group_state": "all_correct" if correct == args.num_rollouts else "all_incorrect" if correct == 0 else "mixed", "unique_answer_count": len(answer_counts)}
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            group_rows.append(payload)
            print(f"[{index + 1}/{len(rows)}] {payload['group_state']} correct={correct}/{args.num_rollouts}")

    states = Counter(row["group_state"] for row in group_rows)
    all_rollouts = [item for row in group_rows for item in row["rollouts"]]
    summary = {"schema_version": "frozen_policy_exploration_v1", "config": vars(args), "n": len(group_rows), "pass_at_k": mean(row["any_correct"] for row in group_rows), "mean_correct_fraction": mean(row["num_correct"] / args.num_rollouts for row in group_rows), "all_correct_rate": states["all_correct"] / len(group_rows), "all_incorrect_rate": states["all_incorrect"] / len(group_rows), "mixed_rate": states["mixed"] / len(group_rows), "mean_unique_answers": mean(row["unique_answer_count"] for row in group_rows), "mean_response_tokens": mean(item["generated_tokens"] for item in all_rollouts if item["generated_tokens"] is not None), "runtime_sec": time.time() - started}
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
