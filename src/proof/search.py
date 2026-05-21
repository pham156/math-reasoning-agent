from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.core.config import log_dir, model_dir
from src.core.infer_baseline import generate_solution
from src.proof.prompts import DEFAULT_PROOF_STRATEGIES, build_branch_prompt, build_completion_prompt, build_strategy_prompt
from src.proof.scoring import prune_branches, score_branch
from src.proof.tree import ProofTreeNode, best_nodes, open_nodes
from src.proof.visualization import tree_to_json, tree_to_mermaid, tree_to_text
from src.verifier.backends import PromptVerifierBackend, VerifierBackend


@dataclass
class ProofSearchConfig:
    model_path: str
    max_depth: int = 2
    beam_width: int = 3
    branch_factor: int = 2
    num_strategies: int = 4
    min_score: float = 3.0
    max_new_tokens: int = 700
    strategy_temperature: float = 0.7
    branch_temperature: float = 0.5
    use_verifier: bool = False
    expected_answer: str | None = None


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")
    return json.loads(match.group(0))


def fallback_strategies(problem: str, num_strategies: int) -> list[dict]:
    strategies = DEFAULT_PROOF_STRATEGIES[:num_strategies]
    return [
        {
            "strategy": strategy,
            "rationale": f"Try a {strategy} approach and test whether it exposes useful structure.",
            "first_step": f"Begin with the standard setup for a {strategy} proof.",
        }
        for strategy in strategies
    ]


def generate_strategy_candidates(
    problem: str,
    model_path: str,
    retrieval_context: str = "",
    num_strategies: int = 4,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
) -> list[dict]:
    prompt = build_strategy_prompt(problem, retrieval_context, num_strategies=num_strategies)
    raw = generate_solution(
        problem=problem,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt=prompt,
    )
    try:
        parsed = extract_json_object(raw)
        candidates = parsed.get("strategies", [])
    except Exception:
        candidates = []
    if not candidates:
        candidates = fallback_strategies(problem, num_strategies)
    return candidates[:num_strategies]


def fallback_branches(node: ProofTreeNode, branch_factor: int) -> list[dict]:
    return [
        {
            "strategy": node.strategy,
            "claim": f"Develop the {node.strategy} path by proving the next necessary lemma.",
            "partial_solution": (
                f"{node.partial_solution}\n\n"
                f"Next, we continue the {node.strategy} approach. "
                "We identify the next claim needed and attempt to justify it rigorously."
            ).strip(),
            "finish_likelihood": "medium",
        }
        for _ in range(branch_factor)
    ]


def expand_branch(
    node: ProofTreeNode,
    model_path: str,
    retrieval_context: str,
    branch_factor: int,
    max_new_tokens: int,
    temperature: float,
) -> list[ProofTreeNode]:
    prompt = build_branch_prompt(
        problem=node.problem,
        strategy=node.strategy,
        claim=node.claim,
        partial_solution=node.partial_solution,
        retrieval_context=retrieval_context,
        branch_factor=branch_factor,
    )
    raw = generate_solution(
        problem=node.problem,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt=prompt,
    )
    try:
        parsed = extract_json_object(raw)
        branches = parsed.get("branches", [])
    except Exception:
        branches = []
    if not branches:
        branches = fallback_branches(node, branch_factor)

    children = []
    for branch in branches[:branch_factor]:
        child = ProofTreeNode(
            problem=node.problem,
            strategy=str(branch.get("strategy", node.strategy)),
            claim=str(branch.get("claim", "Continue this proof branch.")),
            partial_solution=str(branch.get("partial_solution", node.partial_solution)),
            metadata={"finish_likelihood": branch.get("finish_likelihood", "unknown"), "raw_parent_generation": raw},
        )
        node.add_child(child)
        children.append(child)
    node.status = "expanded"
    return children


def build_retrieval_context_for_search(problem: str, top_k: int = 3, candidate_k: int = 30) -> tuple[str, list[dict]]:
    try:
        from src.retrieval.context import build_retrieval_context
        from src.retrieval.retrieve import retrieve_examples
    except Exception:
        return "", []

    try:
        examples = retrieve_examples(problem, top_k=top_k, candidate_k=candidate_k)
    except Exception:
        return "", []
    return build_retrieval_context(problem, examples, max_examples=top_k), examples


def run_proof_search(
    problem: str,
    config: ProofSearchConfig,
    retrieval_context: str = "",
    verifier: VerifierBackend | None = None,
) -> ProofTreeNode:
    verifier = verifier or (PromptVerifierBackend(model_path=config.model_path) if config.use_verifier else None)
    root = ProofTreeNode(problem=problem, strategy="root", claim="Search for a proof.", partial_solution="", depth=0)

    strategy_candidates = generate_strategy_candidates(
        problem=problem,
        model_path=config.model_path,
        retrieval_context=retrieval_context,
        num_strategies=config.num_strategies,
        max_new_tokens=min(config.max_new_tokens, 600),
        temperature=config.strategy_temperature,
    )

    initial_nodes = []
    for candidate in strategy_candidates:
        child = ProofTreeNode(
            problem=problem,
            strategy=str(candidate.get("strategy", "unknown")),
            claim=str(candidate.get("rationale", "")),
            partial_solution=str(candidate.get("first_step", "")),
            metadata={"source": "strategy_generation"},
        )
        root.add_child(child)
        score_branch(child, verifier=None)
        initial_nodes.append(child)
    root.status = "expanded"

    kept, _ = prune_branches(initial_nodes, beam_width=config.beam_width, min_score=0.0)
    for node in kept:
        node.status = "open"

    while True:
        frontier = sorted(
            [node for node in open_nodes(root) if node.depth < config.max_depth],
            key=lambda item: item.score,
            reverse=True,
        )[: config.beam_width]
        if not frontier:
            break

        generated = []
        for node in frontier:
            children = expand_branch(
                node=node,
                model_path=config.model_path,
                retrieval_context=retrieval_context,
                branch_factor=config.branch_factor,
                max_new_tokens=config.max_new_tokens,
                temperature=config.branch_temperature,
            )
            for child in children:
                score_branch(child, verifier=verifier, expected_answer=config.expected_answer)
            generated.extend(children)

        complete = [node for node in generated if node.status == "complete"]
        if complete:
            break

        kept, _ = prune_branches(generated, beam_width=config.beam_width, min_score=config.min_score)
        for node in kept:
            node.status = "open"

    for node in best_nodes(root, limit=1):
        if node.status not in {"complete", "pruned", "failed"}:
            completion_prompt = build_completion_prompt(problem, node.partial_solution, node.strategy, retrieval_context)
            node.partial_solution = generate_solution(
                problem=problem,
                model_path=config.model_path,
                max_new_tokens=config.max_new_tokens,
                temperature=0.2,
                prompt=completion_prompt,
            )
            score_branch(node, verifier=verifier, expected_answer=config.expected_answer)
            if node.status == "open":
                node.status = "complete" if node.score >= 8.0 else "expanded"

    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight proof-search controller.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--branch_factor", type=int, default=2)
    parser.add_argument("--num_strategies", type=int, default=4)
    parser.add_argument("--min_score", type=float, default=3.0)
    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--use_verifier", action="store_true")
    parser.add_argument("--expected_answer", default=None)
    parser.add_argument("--format", choices=["text", "json", "mermaid"], default="text")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    retrieval_context = ""
    if args.use_retrieval:
        retrieval_context, _ = build_retrieval_context_for_search(args.problem)

    config = ProofSearchConfig(
        model_path=args.model_path,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        num_strategies=args.num_strategies,
        min_score=args.min_score,
        max_new_tokens=args.max_new_tokens,
        use_verifier=args.use_verifier,
        expected_answer=args.expected_answer,
    )
    root = run_proof_search(args.problem, config, retrieval_context=retrieval_context)

    if args.format == "json":
        rendered = tree_to_json(root)
    elif args.format == "mermaid":
        rendered = tree_to_mermaid(root)
    else:
        rendered = tree_to_text(root)
        best = best_nodes(root, limit=1)
        if best:
            rendered += "\n\nBest branch solution:\n" + best[0].partial_solution

    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = log_dir(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote proof-search output to {output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
