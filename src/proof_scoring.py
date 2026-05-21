from __future__ import annotations

import math
import re

from src.proof_tree import ProofTreeNode
from src.verifier_backends import VerifierBackend


PROGRESS_TERMS = (
    "therefore",
    "hence",
    "so",
    "it follows",
    "we get",
    "we have",
    "suppose",
    "assume",
    "base case",
    "inductive",
    "contradiction",
    "invariant",
    "construct",
    "count",
    "inequality",
)


def heuristic_branch_score(node: ProofTreeNode) -> float:
    text = f"{node.claim}\n{node.partial_solution}".lower()
    length = len(node.partial_solution)
    progress_hits = sum(1 for term in PROGRESS_TERMS if term in text)
    equation_hits = len(re.findall(r"[=<>]|\\le|\\ge|\\equiv|\\sum|\\prod", text))
    final_signal = any(term in text for term in ("qed", "done", "final answer", "therefore the proof is complete"))
    contradiction_penalty = 0.0
    if any(term in text for term in ("obvious", "clearly", "trivial")) and len(text) < 700:
        contradiction_penalty += 0.8
    if "..." in text or "[missing" in text:
        contradiction_penalty += 1.0

    length_score = min(2.0, math.log1p(max(length, 0)) / 4.0)
    progress_score = min(2.5, progress_hits * 0.35)
    equation_score = min(1.5, equation_hits * 0.12)
    completion_score = 1.5 if final_signal else 0.0
    depth_penalty = max(0.0, node.depth - 2) * 0.25
    return max(0.0, min(10.0, 3.0 + length_score + progress_score + equation_score + completion_score - contradiction_penalty - depth_penalty))


def score_branch(
    node: ProofTreeNode,
    verifier: VerifierBackend | None = None,
    expected_answer: str | None = None,
) -> ProofTreeNode:
    node.heuristic_score = heuristic_branch_score(node)
    node.score = node.heuristic_score
    if verifier and len(node.partial_solution.strip()) >= 80:
        verification = verifier.verify(node.problem, node.partial_solution, expected_answer=expected_answer)
        node.verifier_score = verification.score
        node.critique = verification.critique
        node.score = 0.45 * node.heuristic_score + 0.55 * verification.score
        if verification.is_valid:
            node.status = "complete"
        elif verification.verdict == "invalid" and verification.score < 3.0:
            node.status = "failed"
    return node


def prune_branches(nodes: list[ProofTreeNode], beam_width: int, min_score: float) -> tuple[list[ProofTreeNode], list[ProofTreeNode]]:
    ranked = sorted(nodes, key=lambda node: node.score, reverse=True)
    kept = []
    pruned = []
    for i, node in enumerate(ranked):
        if i < beam_width and node.score >= min_score and node.status != "failed":
            kept.append(node)
        else:
            node.status = "pruned" if node.status != "failed" else "failed"
            pruned.append(node)
    return kept, pruned
