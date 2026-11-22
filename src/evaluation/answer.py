from __future__ import annotations

import re


BOX_COMMANDS = (r"\boxed", r"\fbox")


def _extract_balanced_braced(text: str, command: str) -> list[str]:
    answers = []
    start = 0
    while True:
        command_pos = text.find(command, start)
        if command_pos < 0:
            break
        brace_pos = text.find("{", command_pos + len(command))
        if brace_pos < 0:
            break
        depth = 0
        for pos in range(brace_pos, len(text)):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[brace_pos + 1 : pos].strip())
                    start = pos + 1
                    break
        else:
            break
    return answers


def extract_boxed_answer(text: str) -> str | None:
    for command in BOX_COMMANDS:
        matches = _extract_balanced_braced(text, command)
        if matches:
            return matches[-1].strip()
        unclosed = re.findall(rf"{re.escape(command)}\s*\{{([^{{}}\n\r$\\\]]+)", text, flags=re.DOTALL)
        if unclosed:
            return unclosed[-1].strip()
    return None


def closed_boxed_answers(text: str) -> list[str]:
    """Every balanced \\boxed{...} in order; unclosed (cut-off) boxes are ignored."""
    return [answer for answer in _extract_balanced_braced(text, r"\boxed") if answer]


def extract_committed_answer(text: str) -> str | None:
    """Strict extraction: the last closed \\boxed{...}, the format the system prompt requests.

    Unlike extract_final_answer there is no unclosed-box, '####', phrase, or last-line
    fallback. An output cut off inside a box still yields an earlier closed box, if any.
    """
    answers = closed_boxed_answers(text)
    return answers[-1] if answers else None


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    # GSM8K places "#### <answer>" on the final line. Only honor the marker there so
    # Markdown headings such as "#### Step 1" are never mistaken for the answer.
    if lines and "####" in lines[-1]:
        final = lines[-1].rsplit("####", 1)[-1].strip()
        if final:
            return final

    final_patterns = (
        r"final answer\s*[:\-]\s*(.+)$",
        r"answer\s*[:\-]\s*(.+)$",
        r"therefore\s*,?\s*(.+)$",
    )
    tail = "\n".join(lines[-5:])
    for pattern in final_patterns:
        match = re.search(pattern, tail, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().splitlines()[0].strip()
    return lines[-1] if lines else ""


def normalize_answer(answer: str) -> str:
    text = answer.strip()
    text = extract_boxed_answer(text) or text
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\Big": "",
        r"\big": "",
        r"\,": "",
        r"\!": "",
        r"\ ": "",
        r"\{": "{",
        r"\}": "}",
        "$": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\b(?:and|or)\b", ",", text, flags=re.IGNORECASE)
    text = text.replace(";", ",")
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .,:;")
    if "," in text:
        parts = [part.strip("{}()[] ") for part in text.split(",") if part.strip("{}()[] ")]
        if parts:
            text = ",".join(sorted(parts))
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
