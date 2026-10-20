from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from src.evaluation.answer import extract_final_answer, normalize_answer


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        pairs = (("(", ")"), ("[", "]"), ("{", "}"))
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
    return text


def _decimal_to_fraction(text: str) -> str | None:
    try:
        return str(Fraction(Decimal(text)))
    except (InvalidOperation, ValueError):
        return None


def _numeric_canonical(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        inner = text[:-1].strip()
        frac = _numeric_canonical(inner)
        if frac is None:
            return None
        return str(Fraction(frac) / 100)
    if re.fullmatch(r"[+-]?\d+/\d+", text):
        try:
            return str(Fraction(text))
        except ZeroDivisionError:
            return None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return _decimal_to_fraction(text)
    return None


def canonical_final_answer(text: str) -> str:
    """Canonical answer string for verifiable final-answer rewards.

    This intentionally starts from the project evaluator's extractor and normalizer,
    then adds reward-specific numeric and set canonicalization.
    """

    extracted = extract_final_answer(str(text))
    raw = extracted.strip().replace(r"\{", "{").replace(r"\}", "}")
    raw = raw.replace(r"\left", "").replace(r"\right", "").replace("$", "")
    raw_stripped = _strip_wrappers(raw)
    if raw.startswith("{") and raw.endswith("}"):
        items = [canonical_final_answer(item) for item in raw_stripped.split(",") if item.strip()]
        return "{" + ",".join(sorted(items)) + "}"
    if raw.startswith("[") and raw.endswith("]"):
        items = [canonical_final_answer(item) for item in raw_stripped.split(",") if item.strip()]
        return "[" + ",".join(items) + "]"
    raw_numeric = _numeric_canonical(raw_stripped.replace(r"\%", "%"))
    if raw_numeric is not None:
        return raw_numeric

    normalized = normalize_answer(extracted)
    normalized = normalized.replace(r"\%", "%").replace("%", "%")
    normalized = normalized.replace(r"\dfrac", r"\frac")
    normalized = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", normalized)
    normalized = normalized.replace("−", "-")
    normalized = normalized.strip()

    numeric = _numeric_canonical(normalized)
    if numeric is not None:
        return numeric

    stripped = _strip_wrappers(normalized)
    if stripped != normalized:
        numeric = _numeric_canonical(stripped)
        if numeric is not None:
            return numeric

    if normalized.startswith("{") and normalized.endswith("}"):
        items = [canonical_final_answer(item) for item in _strip_wrappers(normalized).split(",") if item.strip()]
        return "{" + ",".join(sorted(items)) + "}"

    if normalized.startswith("[") and normalized.endswith("]"):
        items = [canonical_final_answer(item) for item in _strip_wrappers(normalized).split(",") if item.strip()]
        return "[" + ",".join(items) + "]"

    return normalized


def final_answer_reward_one(completion: str, reference: str) -> float:
    return 1.0 if canonical_final_answer(completion) == canonical_final_answer(reference) else 0.0


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def final_answer_reward(completions: list[Any], answer: list[str] | str | None = None, **kwargs) -> list[float]:
    """TRL-compatible custom reward function.

    GRPOTrainer passes generated completions plus all extra dataset columns.
    The train dataset constructed by `train_grpo_rlvr.py` includes an `answer`
    column containing the reference final answer.
    """

    references = answer if answer is not None else kwargs.get("reference_answer")
    if references is None:
        raise ValueError("final_answer_reward requires an `answer` or `reference_answer` column.")
    if isinstance(references, str):
        references = [references] * len(completions)
    if len(references) != len(completions):
        raise ValueError(f"Reward got {len(completions)} completions but {len(references)} references.")
    return [
        final_answer_reward_one(_completion_to_text(completion), reference)
        for completion, reference in zip(completions, references)
    ]
