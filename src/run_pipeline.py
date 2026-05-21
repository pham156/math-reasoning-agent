from __future__ import annotations

import argparse
import json

from src.config import model_dir, print_storage_summary
from src.self_critique_loop import run_solver_verifier_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end retrieval, solve, verify, revise pipeline.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--expected_answer", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.json:
        print_storage_summary()
    trace = run_solver_verifier_loop(
        problem=args.problem,
        solver_model_path=args.model_path,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_iterations=args.max_iterations,
        max_new_tokens=args.max_new_tokens,
        solver_temperature=args.temperature,
        expected_answer=args.expected_answer,
    )

    if args.json:
        print(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False))
        return

    print("\nRetrieved examples:")
    for i, ex in enumerate(trace.retrieved_examples, start=1):
        print(
            f"{i}. semantic={ex.get('semantic_score', 0.0):.4f} "
            f"rerank={ex.get('rerank_score', 0.0):.4f} "
            f"topics={','.join(ex.get('topics', []))} strategies={','.join(ex.get('strategies', []))}"
        )

    for step in trace.steps:
        label = "Initial solution" if step.iteration == 0 else f"Revised solution {step.iteration}"
        print(f"\n{label}:\n{step.solution}")
        print("\nVerifier output:\n" + json.dumps(step.verification.to_dict(), indent=2, ensure_ascii=False))

    print("\nFinal solution:\n" + trace.final_solution)


if __name__ == "__main__":
    main()
