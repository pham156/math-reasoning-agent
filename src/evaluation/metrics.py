from __future__ import annotations

from collections import Counter
from statistics import mean

from src.proof.tree import iter_nodes


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def retrieval_stats(examples: list[dict]) -> dict:
    if not examples:
        return {
            "retrieval_top_k": 0,
            "mean_semantic_score": 0.0,
            "mean_rerank_score": 0.0,
            "top_topics": [],
            "top_strategies": [],
        }
    topic_counts = Counter(topic for ex in examples for topic in ex.get("topics", []))
    strategy_counts = Counter(strategy for ex in examples for strategy in ex.get("strategies", []))
    return {
        "retrieval_top_k": len(examples),
        "mean_semantic_score": safe_mean([float(ex.get("semantic_score", ex.get("score", 0.0))) for ex in examples]),
        "mean_rerank_score": safe_mean([float(ex.get("rerank_score", 0.0)) for ex in examples]),
        "top_topics": topic_counts.most_common(5),
        "top_strategies": strategy_counts.most_common(5),
    }


def verifier_stats(verification: dict | None) -> dict:
    if not verification:
        return {
            "verifier_used": False,
            "verifier_verdict": "",
            "verifier_score": None,
            "verifier_confidence": None,
            "num_issues": 0,
            "issue_categories": [],
            "max_issue_severity": 0,
        }
    issues = verification.get("issues", []) or []
    categories = Counter(str(issue.get("category", "unknown")) for issue in issues if isinstance(issue, dict))
    severities = [int(issue.get("severity", 0)) for issue in issues if isinstance(issue, dict)]
    return {
        "verifier_used": True,
        "verifier_verdict": verification.get("verdict", ""),
        "verifier_score": verification.get("score"),
        "verifier_confidence": verification.get("confidence"),
        "num_issues": len(issues),
        "issue_categories": categories.most_common(),
        "max_issue_severity": max(severities) if severities else 0,
    }


def branch_stats(root) -> dict:
    if root is None:
        return {
            "search_used": False,
            "num_nodes": 0,
            "num_complete": 0,
            "num_pruned": 0,
            "max_depth": 0,
            "best_branch_score": None,
            "mean_branch_score": 0.0,
            "strategy_counts": [],
        }
    nodes = list(iter_nodes(root))
    strategy_counts = Counter(node.strategy for node in nodes if node.strategy != "root")
    scores = [float(node.score) for node in nodes if node.depth > 0]
    return {
        "search_used": True,
        "num_nodes": len(nodes),
        "num_complete": sum(node.status == "complete" for node in nodes),
        "num_pruned": sum(node.status == "pruned" for node in nodes),
        "num_failed": sum(node.status == "failed" for node in nodes),
        "max_depth": max((node.depth for node in nodes), default=0),
        "best_branch_score": max(scores) if scores else None,
        "mean_branch_score": safe_mean(scores),
        "strategy_counts": strategy_counts.most_common(),
    }


def summarize_results(rows: list[dict]) -> dict:
    summary = {"num_examples": len(rows)}
    if not rows:
        return summary

    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary["overall"] = {
        "exact_accuracy": safe_mean([1.0 if row.get("exact_match") else 0.0 for row in rows]),
        "mean_runtime_sec": safe_mean([float(row.get("runtime_sec", 0.0)) for row in rows]),
    }
    summary["by_mode"] = {}
    for mode, mode_rows in by_mode.items():
        verifier_scores = [
            float(row["verifier"]["verifier_score"])
            for row in mode_rows
            if row.get("verifier", {}).get("verifier_score") is not None
        ]
        branch_scores = [
            float(row["branch"]["best_branch_score"])
            for row in mode_rows
            if row.get("branch", {}).get("best_branch_score") is not None
        ]
        summary["by_mode"][mode] = {
            "num_examples": len(mode_rows),
            "exact_accuracy": safe_mean([1.0 if row.get("exact_match") else 0.0 for row in mode_rows]),
            "mean_runtime_sec": safe_mean([float(row.get("runtime_sec", 0.0)) for row in mode_rows]),
            "mean_verifier_score": safe_mean(verifier_scores),
            "mean_best_branch_score": safe_mean(branch_scores),
            "mean_retrieval_rerank_score": safe_mean(
                [float(row.get("retrieval", {}).get("mean_rerank_score", 0.0)) for row in mode_rows]
            ),
        }
    return summary
