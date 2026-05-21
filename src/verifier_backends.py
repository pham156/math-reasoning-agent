from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import model_dir
from src.infer_baseline import generate_solution
from src.verifier_prompts import build_verifier_prompt
from src.verifier_schema import VerificationResult, extract_json_object, fallback_verification, normalize_verification


class VerifierBackend(ABC):
    @abstractmethod
    def verify(self, problem: str, solution: str, expected_answer: str | None = None) -> VerificationResult:
        raise NotImplementedError


class PromptVerifierBackend(VerifierBackend):
    def __init__(
        self,
        model_path: str = str(model_dir("Qwen2.5-Math-7B-Instruct")),
        max_new_tokens: int = 900,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def verify(self, problem: str, solution: str, expected_answer: str | None = None) -> VerificationResult:
        prompt = build_verifier_prompt(problem, solution, expected_answer=expected_answer)
        raw = generate_solution(
            problem=problem,
            model_path=self.model_path,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            prompt=prompt,
        )
        try:
            parsed = extract_json_object(raw)
        except Exception as exc:
            return fallback_verification(raw, f"Verifier did not return parseable JSON: {exc}")
        return normalize_verification(parsed, raw=raw)


class PlaceholderTrainedVerifierBackend(VerifierBackend):
    """Adapter interface for a future classifier/reward-model verifier."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def verify(self, problem: str, solution: str, expected_answer: str | None = None) -> VerificationResult:
        raise NotImplementedError("Load a trained verifier model here and return VerificationResult.")
