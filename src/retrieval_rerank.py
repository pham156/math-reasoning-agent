from __future__ import annotations

import math
import re
from collections import Counter

from src.math_taxonomy import labels_for_query, normalize_text


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_count(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))


def lexical_score(query: str, problem: str, solution: str = "") -> float:
    query_tokens = set(tokenize(query))
    problem_tokens = set(tokenize(problem))
    solution_tokens = set(tokenize(solution[:1000]))
    return 0.75 * jaccard(query_tokens, problem_tokens) + 0.25 * jaccard(query_tokens, solution_tokens)


def length_penalty(solution: str, target_chars: int = 1800) -> float:
    if not solution:
        return 0.0
    ratio = len(solution) / target_chars
    return math.exp(-abs(math.log(max(ratio, 0.05))))


def rerank_examples(
    problem: str,
    candidates: list[dict],
    top_k: int = 5,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    weights = weights or {
        "semantic": 0.55,
        "topic": 0.14,
        "strategy": 0.12,
        "theorem": 0.08,
        "lexical": 0.08,
        "solution_shape": 0.03,
    }
    query_labels = labels_for_query(problem)
    query_topics = query_labels.get("topics", [])
    query_strategies = query_labels.get("strategies", [])
    query_theorems = query_labels.get("theorem_patterns", [])

    reranked = []
    for rank, item in enumerate(candidates, start=1):
        semantic = float(item.get("semantic_score", item.get("score", 0.0)))
        topic = overlap_count(query_topics, item.get("topics", [])) / max(len(set(query_topics)), 1)
        strategy = overlap_count(query_strategies, item.get("strategies", [])) / max(len(set(query_strategies)), 1)
        theorem = overlap_count(query_theorems, item.get("theorem_patterns", [])) / max(len(set(query_theorems)), 1)
        lexical = lexical_score(problem, item.get("problem", ""), item.get("solution", ""))
        shape = length_penalty(item.get("solution", ""))
        rank_prior = 1.0 / math.sqrt(rank)

        final = (
            weights["semantic"] * semantic
            + weights["topic"] * topic
            + weights["strategy"] * strategy
            + weights["theorem"] * theorem
            + weights["lexical"] * lexical
            + weights["solution_shape"] * shape
        ) * (0.92 + 0.08 * rank_prior)

        enriched = dict(item)
        enriched["query_labels"] = query_labels
        enriched["rerank_features"] = {
            "semantic": semantic,
            "topic": topic,
            "strategy": strategy,
            "theorem": theorem,
            "lexical": lexical,
            "solution_shape": shape,
        }
        enriched["rerank_score"] = final
        reranked.append(enriched)

    return sorted(reranked, key=lambda row: row["rerank_score"], reverse=True)[:top_k]


def aggregate_retrieval_signals(problem: str, examples: list[dict]) -> dict:
    query_labels = labels_for_query(problem)
    topic_counts = Counter(topic for ex in examples for topic in ex.get("topics", []))
    strategy_counts = Counter(strategy for ex in examples for strategy in ex.get("strategies", []))
    theorem_counts = Counter(theorem for ex in examples for theorem in ex.get("theorem_patterns", []))
    return {
        "query_labels": query_labels,
        "retrieved_topics": topic_counts.most_common(5),
        "retrieved_strategies": strategy_counts.most_common(8),
        "retrieved_theorem_patterns": theorem_counts.most_common(8),
    }
