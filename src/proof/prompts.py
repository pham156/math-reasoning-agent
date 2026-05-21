from __future__ import annotations

import json


DEFAULT_PROOF_STRATEGIES = [
    "induction",
    "contradiction",
    "invariant",
    "inequalities",
    "constructive proof",
    "combinatorial counting",
]


def build_strategy_prompt(problem: str, retrieval_context: str = "", num_strategies: int = 4) -> str:
    context = f"\nRetrieval context:\n{retrieval_context}\n" if retrieval_context else ""
    return (
        "Generate diverse proof strategies for the target math problem.\n"
        "Prefer these strategy families when relevant: induction, contradiction, invariant, inequalities, "
        "constructive proof, combinatorial counting.\n"
        "Return only JSON with a list under key strategies. Each item must include strategy, rationale, and first_step.\n\n"
        f"Problem:\n{problem}\n"
        f"{context}\n"
        f"Number of strategies: {num_strategies}\n\n"
        "JSON schema:\n"
        + json.dumps(
            {
                "strategies": [
                    {
                        "strategy": "contradiction",
                        "rationale": "why this strategy could work",
                        "first_step": "first concrete proof step",
                    }
                ]
            },
            indent=2,
        )
        + "\n\nJSON:"
    )


def build_branch_prompt(
    problem: str,
    strategy: str,
    claim: str,
    partial_solution: str,
    retrieval_context: str = "",
    branch_factor: int = 2,
) -> str:
    context = f"\nRetrieval context:\n{retrieval_context}\n" if retrieval_context else ""
    return (
        "Continue a proof-search branch for a math problem.\n"
        "Create distinct next reasoning branches. Each branch should make concrete mathematical progress, "
        "state a local claim or lemma, and include the next partial proof text.\n"
        "Return only JSON with key branches.\n\n"
        f"Problem:\n{problem}\n"
        f"Strategy: {strategy}\n"
        f"Current branch claim: {claim}\n"
        f"Current partial solution:\n{partial_solution or '[empty]'}\n"
        f"{context}\n"
        f"Branch factor: {branch_factor}\n\n"
        "JSON schema:\n"
        + json.dumps(
            {
                "branches": [
                    {
                        "strategy": strategy,
                        "claim": "local claim for this branch",
                        "partial_solution": "extended proof text",
                        "finish_likelihood": "low | medium | high",
                    }
                ]
            },
            indent=2,
        )
        + "\n\nJSON:"
    )


def build_completion_prompt(problem: str, node_solution: str, strategy: str, retrieval_context: str = "") -> str:
    context = f"\nRetrieval context:\n{retrieval_context}\n" if retrieval_context else ""
    return (
        "Complete this partial mathematical proof if possible. If the branch is flawed, repair it explicitly.\n"
        "End with a clearly marked final answer or final proof conclusion.\n\n"
        f"Problem:\n{problem}\n"
        f"Strategy: {strategy}\n"
        f"{context}\n"
        f"Partial proof:\n{node_solution}\n\n"
        "Completed proof:"
    )
