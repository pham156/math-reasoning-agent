from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

from src.core.config import data_dir, model_dir, output_dir, print_storage_summary, project_root
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


def normalize_retrieved_examples(examples: list[dict]) -> list[dict]:
    normalized = []
    for item in examples:
        similarity_score = float(item.get("semantic_score", item.get("score", 0.0)))
        rerank_score = float(item.get("rerank_score", item.get("score", 0.0)))
        normalized.append(
            {
                "id": item.get("id"),
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
                "type": item.get("type", item.get("category", "")),
                "level": item.get("level", ""),
                "similarity_score": similarity_score,
                "semantic_score": similarity_score,
                "score": similarity_score,
                "rerank_score": rerank_score,
                "faiss_rank": item.get("faiss_rank"),
                "topics": item.get("topics", []),
                "strategies": item.get("strategies", []),
                "theorem_patterns": item.get("theorem_patterns", []),
            }
        )
    return normalized


def toy_retrieve(problem: str, top_k: int) -> list[dict]:
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
    return toy


def math_dataset_candidates() -> list[Path]:
    candidates = []
    if os.environ.get("MATH_DATASET_DIR"):
        candidates.append(Path(os.environ["MATH_DATASET_DIR"]).expanduser())
    candidates.append(data_dir("MATH"))
    candidates.append(project_root() / "data" / "MATH")
    return candidates


def find_math_dataset_dir() -> Path | None:
    for candidate in math_dataset_candidates():
        if candidate.exists():
            return candidate.resolve()
    return None


def math_index_paths() -> tuple[Path, Path, Path]:
    faiss_dir = data_dir("faiss")
    return faiss_dir / "math.index", faiss_dir / "metadata.jsonl", faiss_dir / "index_config.json"


def ensure_math_index(auto_build: bool = True) -> dict:
    index_path, metadata_path, config_path = math_index_paths()
    dataset_dir = find_math_dataset_dir()
    metadata = {
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "index_config_path": str(config_path),
        "index_built_this_run": False,
    }
    if index_path.exists() and metadata_path.exists():
        return metadata

    if dataset_dir is None:
        candidates = "\n".join(f"  - {path}" for path in math_dataset_candidates())
        raise RuntimeError(
            "MATH dataset is not available. Expected one of:\n"
            f"{candidates}\n\n"
            "Download it with:\n"
            "  source scripts/setup_env.sh\n"
            "  python scripts/download_data.py\n"
            "Then build the index with:\n"
            "  python -m src.retrieval.build_index"
        )

    if not auto_build:
        raise RuntimeError(
            f"FAISS index is missing at {index_path}. Build it with:\n"
            "  source scripts/setup_env.sh\n"
            "  python -m src.retrieval.build_index"
        )

    try:
        from src.retrieval.build_index import build_faiss_index

        build_faiss_index(
            dataset_dir=dataset_dir,
            embedding_model=model_dir("all-MiniLM-L6-v2"),
            output_dir=data_dir("faiss"),
        )
    except Exception as exc:
        raise RuntimeError(
            "FAISS index is missing and automatic build failed.\n"
            f"Reason: {type(exc).__name__}: {exc}\n\n"
            "After installing retrieval dependencies, run:\n"
            "  source scripts/setup_env.sh\n"
            "  python -m src.retrieval.build_index"
        ) from exc

    metadata["index_built_this_run"] = True
    return metadata


def retrieve_math(problem: str, top_k: int, candidate_k: int) -> tuple[list[dict], dict]:
    metadata = ensure_math_index(auto_build=True)
    index_path = metadata["index_path"]
    metadata_path = metadata["metadata_path"]
    examples = retrieve_examples(
        problem,
        top_k=top_k,
        candidate_k=candidate_k,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    metadata["num_retrieved"] = len(examples)
    return examples, metadata


def retrieve_for_pipeline(problem: str, top_k: int, candidate_k: int, retrieval_source: str) -> tuple[list[dict], dict]:
    if retrieval_source == "toy":
        examples = toy_retrieve(problem, top_k)
        return examples, {"requested_source": "toy", "actual_source": "toy", "num_retrieved": len(examples)}

    if retrieval_source == "math":
        examples, metadata = retrieve_math(problem, top_k=top_k, candidate_k=candidate_k)
        metadata.update({"requested_source": "math", "actual_source": "math"})
        return examples, metadata

    try:
        examples, metadata = retrieve_math(problem, top_k=top_k, candidate_k=candidate_k)
        metadata.update({"requested_source": "auto", "actual_source": "math"})
        return examples, metadata
    except Exception as exc:
        examples = toy_retrieve(problem, top_k)
        return examples, {
            "requested_source": "auto",
            "actual_source": "toy",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "num_retrieved": len(examples),
        }


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
    retrieval_source: str,
) -> dict:
    start = time.time()
    retrieved_examples_raw, retrieval_metadata = retrieve_for_pipeline(
        problem,
        top_k=top_k,
        candidate_k=candidate_k,
        retrieval_source=retrieval_source,
    )
    retrieved_examples = normalize_retrieved_examples(retrieved_examples_raw)
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
        "retrieval_metadata": retrieval_metadata,
        "solver_prompt": solver_prompt,
        "raw_solution": raw_solution,
        "verifier_critique": verification.to_dict(),
        "final_solution": final_solution,
        "runtime_metadata": {
            "runtime_sec": end - start,
            "llm_mode": llm_mode,
            "fallback_reason": fallback_reason,
            "retrieval_source": retrieval_metadata.get("actual_source"),
            "retrieval_metadata": retrieval_metadata,
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
    parser.add_argument("--retrieval_source", choices=["auto", "toy", "math"], default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print_storage_summary()

    try:
        result = run_pipeline(
            problem=args.problem,
            model_path=args.model_path,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            mock_llm=args.mock_llm,
            retrieval_source=args.retrieval_source,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.quiet:
        print(f"Wrote smoke-test output to {output_path}")
        print(json.dumps(result["runtime_metadata"], indent=2))


if __name__ == "__main__":
    main()
