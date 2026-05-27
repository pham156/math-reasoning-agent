from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.core.config import model_dir, output_dir, print_storage_summary
from src.core.infer_baseline import generate_solution
from src.retrieval.context import build_solver_prompt
from src.retrieval.math_taxonomy import enrich_example
from src.retrieval.retrieve import retrieve_examples
from src.verifier.backends import PromptVerifierBackend
from src.verifier.schema import VerificationResult, normalize_verification


TOY_EXAMPLES = [
    {
        "id": "toy-quadratic-1",
        "problem": "Find all real x such that x^2 - 5x + 6 = 0.",
        "solution": "Factor x^2 - 5x + 6 = (x - 2)(x - 3). Hence x = 2 or x = 3.",
        "type": "algebra",
        "level": "toy",
    },
    {
        "id": "toy-quadratic-2",
        "problem": "Solve x^2 - 7x + 12 = 0.",
        "solution": "Factor x^2 - 7x + 12 = (x - 3)(x - 4), so the solutions are x = 3 and x = 4.",
        "type": "algebra",
        "level": "toy",
    },
    {
        "id": "toy-linear-factor",
        "problem": "Find all real x such that (x - 1)(x + 2) = 0.",
        "solution": "By the zero product property, x - 1 = 0 or x + 2 = 0. Thus x = 1 or x = -2.",
        "type": "algebra",
        "level": "toy",
    },
]


def resolve_output_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested
    parts = requested.parts
    if parts and parts[0] == "outputs":
        return output_dir(Path(*parts[1:]) if len(parts) > 1 else "")
    return output_dir(requested)


def model_available(path: str | Path) -> bool:
    model_path = Path(path)
    return model_path.exists() and any((model_path / name).exists() for name in ("config.json", "tokenizer_config.json"))


def retrieve_or_toy(problem: str, top_k: int, candidate_k: int) -> tuple[list[dict], str]:
    try:
        examples = retrieve_examples(problem, top_k=top_k, candidate_k=candidate_k)
        if examples:
            return examples, "faiss"
    except Exception:
        pass

    toy = []
    for rank, example in enumerate(TOY_EXAMPLES[:top_k], start=1):
        enriched = enrich_example(example)
        enriched.update(
            {
                "semantic_score": 1.0 / rank,
                "rerank_score": 1.0 / rank,
                "score": 1.0 / rank,
                "faiss_rank": None,
                "retrieval_source": "toy",
            }
        )
        toy.append(enriched)
    return toy, "toy"


def mock_generate(problem: str, prompt: str) -> str:
    normalized = problem.lower().replace(" ", "")
    if "x^2-5x+6=0" in normalized or "x^{2}-5x+6=0" in normalized:
        return (
            "We solve the quadratic by factoring:\n"
            "x^2 - 5x + 6 = (x - 2)(x - 3).\n"
            "Thus (x - 2)(x - 3) = 0. By the zero product property, x = 2 or x = 3.\n"
            "Final answer: \\boxed{2,3}"
        )
    return (
        "Mock solution: use the retrieved examples to identify the relevant algebraic pattern, "
        "then solve step by step. Final answer: \\boxed{mock}"
    )


def mock_verify(problem: str, solution: str) -> VerificationResult:
    normalized = solution.lower().replace(" ", "")
    if "x=2" in normalized and "x=3" in normalized or "\\boxed{2,3}" in normalized:
        data = {
            "verdict": "valid",
            "score": 9.0,
            "confidence": 0.9,
            "issues": [],
            "algebra_checks": ["Verified factorization x^2 - 5x + 6 = (x - 2)(x - 3)."],
            "logical_checks": ["Used the zero product property to derive both roots."],
            "missing_steps": [],
            "critique": "The solution correctly factors the quadratic and lists both real roots.",
            "suggested_revision_plan": [],
        }
    else:
        data = {
            "verdict": "uncertain",
            "score": 5.0,
            "confidence": 0.5,
            "issues": [
                {
                    "category": "incomplete_proof",
                    "severity": 3,
                    "location": "final answer",
                    "description": "The mock verifier could not confirm the final answer.",
                    "suggested_fix": "Show the algebraic derivation and state all solutions.",
                }
            ],
            "critique": "The solution needs more explicit verification.",
            "suggested_revision_plan": ["Show the complete derivation.", "State the final answer clearly."],
        }
    return normalize_verification(data, raw=json.dumps(data))


def run_pipeline(
    problem: str,
    model_path: str,
    top_k: int,
    candidate_k: int,
    max_new_tokens: int,
    temperature: float,
    mock_llm: bool,
) -> dict:
    start = time.time()
    retrieved_examples, retrieval_source = retrieve_or_toy(problem, top_k=top_k, candidate_k=candidate_k)
    solver_prompt = build_solver_prompt(problem, retrieved_examples, max_examples=top_k)

    llm_mode = "mock" if mock_llm else "real"
    fallback_reason = None
    if not mock_llm and not model_available(model_path):
        llm_mode = "mock"
        fallback_reason = f"Model path is unavailable or incomplete: {model_path}"

    if llm_mode == "mock":
        raw_solution = mock_generate(problem, solver_prompt)
        verification = mock_verify(problem, raw_solution)
    else:
        try:
            raw_solution = generate_solution(
                problem=problem,
                model_path=model_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt=solver_prompt,
            )
            verification = PromptVerifierBackend(model_path=model_path).verify(problem, raw_solution)
        except Exception as exc:
            llm_mode = "mock"
            fallback_reason = f"Real model execution failed: {type(exc).__name__}: {exc}"
            raw_solution = mock_generate(problem, solver_prompt)
            verification = mock_verify(problem, raw_solution)

    final_solution = raw_solution
    end = time.time()
    return {
        "input_problem": problem,
        "retrieved_examples": retrieved_examples,
        "solver_prompt": solver_prompt,
        "raw_solution": raw_solution,
        "verifier_critique": verification.to_dict(),
        "final_solution": final_solution,
        "runtime_metadata": {
            "runtime_sec": end - start,
            "llm_mode": llm_mode,
            "fallback_reason": fallback_reason,
            "retrieval_source": retrieval_source,
            "model_path": model_path,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "output_schema": "smoke_test_v1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal end-to-end smoke-test pipeline.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print_storage_summary()

    result = run_pipeline(
        problem=args.problem,
        model_path=args.model_path,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mock_llm=args.mock_llm,
    )

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.quiet:
        print(f"Wrote smoke-test output to {output_path}")
        print(json.dumps(result["runtime_metadata"], indent=2))


if __name__ == "__main__":
    main()
