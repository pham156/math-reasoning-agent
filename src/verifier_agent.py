from __future__ import annotations

import argparse
import json

from src.config import model_dir
from src.verifier_backends import PromptVerifierBackend, VerifierBackend
from src.verifier_prompts import build_revision_prompt, build_verifier_prompt
from src.verifier_schema import VerificationResult


def verify_solution(
    problem: str,
    solution: str,
    model_path: str = str(model_dir("Qwen2.5-Math-7B-Instruct")),
    max_new_tokens: int = 900,
    temperature: float = 0.0,
    expected_answer: str | None = None,
    backend: VerifierBackend | None = None,
) -> dict:
    verifier = backend or PromptVerifierBackend(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return verifier.verify(problem, solution, expected_answer=expected_answer).to_dict()


def verify_solution_typed(
    problem: str,
    solution: str,
    model_path: str = str(model_dir("Qwen2.5-Math-7B-Instruct")),
    max_new_tokens: int = 900,
    temperature: float = 0.0,
    expected_answer: str | None = None,
    backend: VerifierBackend | None = None,
) -> VerificationResult:
    verifier = backend or PromptVerifierBackend(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return verifier.verify(problem, solution, expected_answer=expected_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a math solution and return structured critique JSON.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--expected_answer", default=None)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--max_new_tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--print_prompt", action="store_true")
    parser.add_argument("--print_revision_prompt", action="store_true")
    args = parser.parse_args()

    if args.print_prompt:
        print(build_verifier_prompt(args.problem, args.solution, expected_answer=args.expected_answer))
        return

    result = verify_solution_typed(
        problem=args.problem,
        solution=args.solution,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        expected_answer=args.expected_answer,
    )

    if args.print_revision_prompt:
        print(build_revision_prompt(args.problem, args.solution, result))
        return

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
