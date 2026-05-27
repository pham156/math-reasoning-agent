from __future__ import annotations

from pathlib import Path

from src.core.config import cache_dir, model_dir


def resolve_embedding_model(path_or_id: str | Path | None = None) -> str:
    if path_or_id is None:
        candidate = model_dir("all-MiniLM-L6-v2")
        return str(candidate) if candidate.exists() else "sentence-transformers/all-MiniLM-L6-v2"

    raw = str(path_or_id)
    candidate = Path(raw)
    if candidate.exists():
        return str(candidate)
    if "/" in raw and not raw.startswith("/"):
        return raw
    if candidate.name == "all-MiniLM-L6-v2":
        return "sentence-transformers/all-MiniLM-L6-v2"
    return raw


def load_embedding_model(path_or_id: str | Path | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(resolve_embedding_model(path_or_id), cache_folder=str(cache_dir() / "huggingface"), device="cpu")


def embedding_text(example: dict, mode: str = "problem_strategy") -> str:
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    topics = ", ".join(example.get("topics", []) or [example.get("type", "")])
    strategies = ", ".join(example.get("strategies", []))
    theorem_patterns = ", ".join(example.get("theorem_patterns", []))
    level = example.get("level", "")

    if mode == "problem_only":
        return problem
    if mode == "problem_solution":
        return f"Problem: {problem}\nSolution: {solution}"
    return (
        f"Topic: {topics}\n"
        f"Level: {level}\n"
        f"Likely strategies: {strategies}\n"
        f"Theorem patterns: {theorem_patterns}\n"
        f"Problem: {problem}\n"
        f"Solution sketch source: {solution[:1600]}"
    )
