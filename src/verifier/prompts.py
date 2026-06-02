from __future__ import annotations

import json

from src.verifier.schema import VerificationResult


VERIFIER_JSON_SCHEMA = {
    "verdict": "valid | invalid | uncertain",
    "score": "number from 0 to 10",
    "confidence": "number from 0 to 1",
    "issues": [
        {
            "category": "algebra_mistake | logical_gap | unjustified_claim | incomplete_proof | invalid_assumption | answer_mismatch | presentation",
            "severity": "integer from 1 to 5",
            "location": "specific line, equation, or step",
            "description": "what is wrong or risky",
            "suggested_fix": "concrete repair",
        }
    ],
    "algebra_checks": ["specific algebraic transformations checked"],
    "logical_checks": ["specific implication or proof steps checked"],
    "missing_steps": ["steps needed for completeness"],
    "critique": "short structured critique",
    "suggested_revision_plan": ["ordered edits the solver should make"],
}


def build_verifier_prompt(problem: str, solution: str, expected_answer: str | None = None) -> str:
    answer_block = f"\nExpected final answer, if known:\n{expected_answer}\n" if expected_answer else ""
    return (
        "You are a strict verifier for olympiad and competition-style math solutions.\n"
        "Your job is not to solve from scratch unless needed for checking. Audit the proposed solution.\n\n"
        "Check for these failure modes:\n"
        "1. algebra mistakes: invalid transformations, sign errors, bad substitutions, arithmetic errors.\n"
        "2. logical gaps: conclusions that do not follow, skipped cases, circular reasoning.\n"
        "3. unjustified claims: assertions without proof, unproved lemmas, hidden assumptions.\n"
        "4. incomplete proofs: missing existence/uniqueness, boundary cases, final answer not established.\n"
        "5. answer mismatch: final answer does not match the argument or known expected answer.\n\n"
        "Be fair: do not mark a solution invalid for minor style issues if the math is complete.\n"
        "Return exactly one raw JSON object. Start with { and end with }. Do not include markdown, LaTeX, prose, or code fences.\n\n"
        f"{json.dumps(VERIFIER_JSON_SCHEMA, indent=2)}\n\n"
        "Example for a complete correct solution:\n"
        '{"verdict":"valid","score":9,"confidence":0.9,"issues":[],"algebra_checks":["checked core algebra"],'
        '"logical_checks":["checked implications"],"missing_steps":[],"critique":"The solution is mathematically complete.",'
        '"suggested_revision_plan":[]}\n\n'
        f"Problem:\n{problem}\n"
        f"{answer_block}\n"
        f"Proposed solution:\n{solution}\n\n"
        "Raw JSON object only:"
    )


def build_critique_prompt(problem: str, solution: str, verification: VerificationResult) -> str:
    return (
        "Convert the verifier result into a concise critique for a math solver.\n"
        "Prioritize mathematical correctness over style. Group comments by severity.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution:\n{solution}\n\n"
        f"Verifier result:\n{json.dumps(verification.to_dict(), indent=2)}\n\n"
        "Critique:"
    )


def build_revision_prompt(
    problem: str,
    solution: str,
    verification: VerificationResult,
    retrieval_context: str = "",
) -> str:
    context_block = f"\nHelpful retrieval context:\n{retrieval_context}\n" if retrieval_context else ""
    return (
        "You are revising a mathematical solution after verification.\n"
        "Fix the mathematical issues directly. Do not merely apologize or describe changes.\n"
        "Preserve correct parts of the original solution, but replace invalid or incomplete reasoning.\n"
        "Make algebra transformations explicit and justify each nontrivial claim.\n\n"
        f"Problem:\n{problem}\n"
        f"{context_block}\n"
        f"Original solution:\n{solution}\n\n"
        f"Verifier critique:\n{json.dumps(verification.to_dict(), indent=2)}\n\n"
        "Write the revised solution. End with a clearly marked final answer.\n\nRevised solution:"
    )
