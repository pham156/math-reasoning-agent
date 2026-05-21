from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from src.config import model_dir
from src.infer_baseline import generate_solution
from src.retrieval_context import build_retrieval_context
from src.verifier_backends import PromptVerifierBackend, VerifierBackend
from src.verifier_prompts import build_revision_prompt
from src.verifier_schema import VerificationResult


@dataclass
class SolverVerifierStep:
    iteration: int
    solution: str
    verification: VerificationResult
    revised_from_previous: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["verification"] = self.verification.to_dict()
        return data


@dataclass
class SolverVerifierTrace:
    problem: str
    retrieved_examples: list[dict]
    initial_prompt: str
    steps: list[SolverVerifierStep] = field(default_factory=list)

    @property
    def final_solution(self) -> str:
        return self.steps[-1].solution if self.steps else ""

    @property
    def final_verification(self) -> VerificationResult | None:
        return self.steps[-1].verification if self.steps else None

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "retrieved_examples": self.retrieved_examples,
            "initial_prompt": self.initial_prompt,
            "steps": [step.to_dict() for step in self.steps],
            "final_solution": self.final_solution,
            "final_verification": self.final_verification.to_dict() if self.final_verification else None,
        }


def run_solver_verifier_loop(
    problem: str,
    solver_model_path: str,
    verifier: VerifierBackend | None = None,
    top_k: int = 3,
    candidate_k: int = 30,
    max_iterations: int = 2,
    max_new_tokens: int = 1024,
    solver_temperature: float = 0.4,
    revision_temperature: float = 0.2,
    expected_answer: str | None = None,
) -> SolverVerifierTrace:
    from src.solver_agent import solve_with_retrieval

    verifier = verifier or PromptVerifierBackend(model_path=solver_model_path)
    solve_result = solve_with_retrieval(
        problem=problem,
        model_path=solver_model_path,
        top_k=top_k,
        candidate_k=candidate_k,
        num_candidates=1,
        max_new_tokens=max_new_tokens,
        temperature=solver_temperature,
    )
    examples = solve_result["examples"]
    trace = SolverVerifierTrace(problem=problem, retrieved_examples=examples, initial_prompt=solve_result["prompt"])

    solution = solve_result["candidates"][0]
    for iteration in range(max_iterations + 1):
        verification = verifier.verify(problem, solution, expected_answer=expected_answer)
        trace.steps.append(
            SolverVerifierStep(
                iteration=iteration,
                solution=solution,
                verification=verification,
                revised_from_previous=iteration > 0,
            )
        )
        if verification.is_valid or iteration >= max_iterations:
            break

        retrieval_context = build_retrieval_context(problem, examples)
        revision_prompt = build_revision_prompt(
            problem=problem,
            solution=solution,
            verification=verification,
            retrieval_context=retrieval_context,
        )
        solution = generate_solution(
            problem=problem,
            model_path=solver_model_path,
            max_new_tokens=max_new_tokens,
            temperature=revision_temperature,
            prompt=revision_prompt,
        )

    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solver -> verifier -> revised solver loop.")
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

    for step in trace.steps:
        print(f"\nIteration {step.iteration}")
        print(f"Verdict: {step.verification.verdict} score={step.verification.score:.1f}")
        print(step.verification.critique)
    print("\nFinal solution:\n" + trace.final_solution)


if __name__ == "__main__":
    main()
