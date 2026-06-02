from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


IssueCategory = Literal[
    "algebra_mistake",
    "logical_gap",
    "unjustified_claim",
    "incomplete_proof",
    "invalid_assumption",
    "answer_mismatch",
    "presentation",
]


@dataclass
class VerificationIssue:
    category: str
    severity: int
    location: str
    description: str
    suggested_fix: str

    @classmethod
    def from_obj(cls, obj: Any) -> "VerificationIssue":
        if isinstance(obj, str):
            return cls(
                category="logical_gap",
                severity=2,
                location="unspecified",
                description=obj,
                suggested_fix="Address this issue explicitly.",
            )
        if not isinstance(obj, dict):
            return cls(
                category="logical_gap",
                severity=2,
                location="unspecified",
                description=str(obj),
                suggested_fix="Address this issue explicitly.",
            )
        return cls(
            category=str(obj.get("category", "logical_gap")),
            severity=clamp_int(obj.get("severity", 2), 1, 5),
            location=str(obj.get("location", "unspecified")),
            description=str(obj.get("description", obj.get("issue", ""))).strip(),
            suggested_fix=str(obj.get("suggested_fix", obj.get("fix", ""))).strip(),
        )


@dataclass
class VerificationResult:
    verdict: str
    score: float
    confidence: float
    issues: list[VerificationIssue] = field(default_factory=list)
    critique: str = ""
    missing_steps: list[str] = field(default_factory=list)
    algebra_checks: list[str] = field(default_factory=list)
    logical_checks: list[str] = field(default_factory=list)
    suggested_revision_plan: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_valid(self) -> bool:
        return self.verdict == "valid" and self.score >= 8.0 and not any(issue.severity >= 4 for issue in self.issues)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["issues"] = [asdict(issue) for issue in self.issues]
        return data


def clamp_int(value: Any, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = lower
    return max(lower, min(upper, parsed))


def clamp_float(value: Any, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = lower
    return max(lower, min(upper, parsed))


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in verifier output.")
    return json.loads(match.group(0))


def normalize_verification(obj: dict, raw: str = "") -> VerificationResult:
    issues = [VerificationIssue.from_obj(issue) for issue in obj.get("issues", [])]
    if "score" in obj:
        score = clamp_float(obj.get("score"), 0.0, 10.0)
    else:
        from src.verifier.scoring import score_from_issues

        score = score_from_issues(issues)
    confidence = clamp_float(obj.get("confidence", 0.5), 0.0, 1.0)
    verdict = str(obj.get("verdict", "invalid")).lower().strip()
    if verdict not in {"valid", "invalid", "uncertain"}:
        from src.verifier.scoring import verdict_from_score

        verdict = verdict_from_score(score, issues)
    if score < 8.0 and verdict == "valid":
        verdict = "uncertain"
    if any(issue.severity >= 4 for issue in issues):
        verdict = "invalid"

    critique = str(obj.get("critique", "")).strip()
    if not critique:
        from src.verifier.scoring import critique_from_issues

        critique = critique_from_issues(issues, score, verdict)

    return VerificationResult(
        verdict=verdict,
        score=score,
        confidence=confidence,
        issues=issues,
        critique=critique,
        missing_steps=[str(x) for x in obj.get("missing_steps", [])],
        algebra_checks=[str(x) for x in obj.get("algebra_checks", [])],
        logical_checks=[str(x) for x in obj.get("logical_checks", [])],
        suggested_revision_plan=[str(x) for x in obj.get("suggested_revision_plan", [])],
        raw=raw,
    )


def fallback_verification(raw: str, reason: str) -> VerificationResult:
    normalized_raw = raw.lower()
    valid_markers = (
        "the solution is valid",
        "solution is valid",
        "\\boxed{\\text{valid}}",
        '"verdict": "valid"',
        '"verdict":"valid"',
    )
    invalid_markers = (
        "solution is invalid",
        "the solution is invalid",
        '"verdict": "invalid"',
        '"verdict":"invalid"',
    )
    if any(marker in normalized_raw for marker in valid_markers) and not any(
        marker in normalized_raw for marker in invalid_markers
    ):
        return VerificationResult(
            verdict="valid",
            score=8.0,
            confidence=0.6,
            issues=[],
            critique=raw.strip(),
            suggested_revision_plan=[],
            raw=raw,
        )

    return VerificationResult(
        verdict="invalid",
        score=0.0,
        confidence=0.0,
        issues=[
            VerificationIssue(
                category="presentation",
                severity=3,
                location="verifier_output",
                description=reason,
                suggested_fix="Rerun verification or inspect the raw verifier output.",
            )
        ],
        critique=raw.strip(),
        suggested_revision_plan=["Produce a solution with explicit algebra checks and justified logical steps."],
        raw=raw,
    )
