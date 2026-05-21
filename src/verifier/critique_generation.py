from __future__ import annotations

from src.core.infer_baseline import generate_solution
from src.verifier.prompts import build_critique_prompt
from src.verifier.schema import VerificationResult
from src.verifier.scoring import critique_from_issues


def generate_structured_critique(
    problem: str,
    solution: str,
    verification: VerificationResult,
    model_path: str | None = None,
    max_new_tokens: int = 500,
) -> str:
    if model_path is None:
        return critique_from_issues(verification.issues, verification.score, verification.verdict)

    return generate_solution(
        problem=problem,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        prompt=build_critique_prompt(problem, solution, verification),
    )
