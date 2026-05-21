from __future__ import annotations

import re


BOX_PATTERNS = (
    r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    r"\\fbox\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
)


def extract_boxed_answer(text: str) -> str | None:
    for pattern in BOX_PATTERNS:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
    return None


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    final_patterns = (
        r"final answer\s*[:\-]\s*(.+)$",
        r"answer\s*[:\-]\s*(.+)$",
        r"therefore\s*,?\s*(.+)$",
    )
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    tail = "\n".join(lines[-5:])
    for pattern in final_patterns:
        match = re.search(pattern, tail, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().splitlines()[0].strip()
    return lines[-1] if lines else ""


def normalize_answer(answer: str) -> str:
    text = answer.strip()
    text = extract_boxed_answer(text) or text
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\ ": "",
        "$": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .,:;")
    return text.lower()


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(extract_final_answer(prediction)) == normalize_answer(extract_final_answer(reference))


def answer_record(prediction: str, reference: str) -> dict:
    pred_answer = extract_final_answer(prediction)
    ref_answer = extract_final_answer(reference)
    return {
        "predicted_answer": pred_answer,
        "reference_answer": ref_answer,
        "normalized_prediction": normalize_answer(pred_answer),
        "normalized_reference": normalize_answer(ref_answer),
        "exact_match": normalize_answer(pred_answer) == normalize_answer(ref_answer),
    }
