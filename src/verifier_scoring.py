from __future__ import annotations

from src.verifier_schema import VerificationIssue


CATEGORY_PENALTIES = {
    "algebra_mistake": 1.6,
    "logical_gap": 1.4,
    "unjustified_claim": 1.1,
    "incomplete_proof": 1.5,
    "invalid_assumption": 1.8,
    "answer_mismatch": 2.0,
    "presentation": 0.4,
}


def score_from_issues(issues: list[VerificationIssue]) -> float:
    if not issues:
        return 9.0
    penalty = 0.0
    for issue in issues:
        category_weight = CATEGORY_PENALTIES.get(issue.category, 1.0)
        penalty += category_weight * issue.severity
    return max(0.0, min(10.0, 10.0 - penalty))


def verdict_from_score(score: float, issues: list[VerificationIssue]) -> str:
    if any(issue.severity >= 4 for issue in issues):
        return "invalid"
    if score >= 8.0:
        return "valid"
    if score >= 6.0:
        return "uncertain"
    return "invalid"


def critique_from_issues(issues: list[VerificationIssue], score: float, verdict: str) -> str:
    if not issues:
        return f"Verdict: {verdict}. Score: {score:.1f}/10. No substantive mathematical issues were identified."

    ordered = sorted(issues, key=lambda issue: issue.severity, reverse=True)
    lines = [f"Verdict: {verdict}. Score: {score:.1f}/10.", "Main issues:"]
    for issue in ordered:
        lines.append(
            f"- [{issue.category}, severity {issue.severity}] {issue.location}: "
            f"{issue.description} Fix: {issue.suggested_fix}"
        )
    return "\n".join(lines)
