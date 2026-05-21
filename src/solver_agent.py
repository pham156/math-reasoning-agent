from __future__ import annotations

import argparse

from src.config import model_dir
from src.infer_baseline import generate_solution
from src.retrieval_context import build_solver_prompt
from src.retrieve import retrieve_examples


def build_rag_prompt(problem: str, examples: list[dict]) -> str:
    return build_solver_prompt(problem, examples)


def solve_with_retrieval(
    problem: str,
    model_path: str = str(model_dir("Qwen2.5-Math-7B-Instruct")),
    top_k: int = 3,
    candidate_k: int = 30,
    num_candidates: int = 1,
    max_new_tokens: int = 1024,
    temperature: float = 0.4,
) -> dict:
    examples = retrieve_examples(problem, top_k=top_k, candidate_k=candidate_k)
    prompt = build_rag_prompt(problem, examples)
    candidates = []
    for _ in range(num_candidates):
        candidates.append(
            generate_solution(
                problem=problem,
                model_path=model_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt=prompt,
            )
        )
    return {"examples": examples, "prompt": prompt, "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG math solver agent.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()

    result = solve_with_retrieval(**vars(args))
    for i, candidate in enumerate(result["candidates"], start=1):
        print(f"\nCandidate {i}\n{'=' * 80}\n{candidate}")


if __name__ == "__main__":
    main()
