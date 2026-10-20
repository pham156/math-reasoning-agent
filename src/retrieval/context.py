from __future__ import annotations

from src.retrieval.rerank import aggregate_retrieval_signals
from src.retrieval.strategy_bank import describe_patterns


def truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n...[truncated]"


def format_strategy_summary(problem: str, examples: list[dict]) -> str:
    signals = aggregate_retrieval_signals(problem, examples)
    query_labels = signals["query_labels"]
    lines = [
        "Retrieval analysis:",
        f"- inferred target topics: {', '.join(query_labels.get('topics', []))}",
        f"- inferred target strategies: {', '.join(query_labels.get('strategies', []) or ['unknown'])}",
        f"- inferred theorem patterns: {', '.join(query_labels.get('theorem_patterns', []) or ['unknown'])}",
    ]
    if signals["retrieved_strategies"]:
        lines.append(
            "- strongest retrieved strategies: "
            + ", ".join(f"{name} ({count})" for name, count in signals["retrieved_strategies"])
        )
    if signals["retrieved_theorem_patterns"]:
        lines.append(
            "- useful theorem patterns: "
            + ", ".join(f"{name} ({count})" for name, count in signals["retrieved_theorem_patterns"])
        )
    strategy_names = [name for name, _ in signals["retrieved_strategies"]]
    theorem_names = [name for name, _ in signals["retrieved_theorem_patterns"]]
    pattern_descriptions = describe_patterns(
        list(dict.fromkeys(query_labels.get("strategies", []) + strategy_names))[:6],
        list(dict.fromkeys(query_labels.get("theorem_patterns", []) + theorem_names))[:6],
    )
    if pattern_descriptions:
        lines.append("Pattern hints:")
        lines.extend(pattern_descriptions)
    return "\n".join(lines)


def format_retrieved_example(example: dict, index: int, solution_chars: int = 1200) -> str:
    features = example.get("rerank_features", {})
    feature_text = ", ".join(f"{k}={v:.3f}" for k, v in features.items() if isinstance(v, float))
    return (
        f"Example {index}\n"
        f"Topic/subfield: {', '.join(example.get('topics', []) or [example.get('type', 'unknown')])}\n"
        f"Level: {example.get('level', '')}\n"
        f"Strategies: {', '.join(example.get('strategies', []) or ['unknown'])}\n"
        f"Theorem patterns: {', '.join(example.get('theorem_patterns', []) or ['none tagged'])}\n"
        f"Retrieval scores: semantic={example.get('semantic_score', example.get('score', 0.0)):.3f}, "
        f"rerank={example.get('rerank_score', 0.0):.3f}"
        + (f", features=({feature_text})" if feature_text else "")
        + "\n"
        f"Problem:\n{truncate(example.get('problem', ''), 900)}\n"
        f"Solution pattern:\n{truncate(example.get('solution', ''), solution_chars)}"
    )


def build_retrieval_context(
    problem: str,
    examples: list[dict],
    max_examples: int = 4,
    solution_chars: int = 1200,
) -> str:
    selected = examples[:max_examples]
    blocks = [format_strategy_summary(problem, selected)]
    blocks.extend(format_retrieved_example(ex, i, solution_chars=solution_chars) for i, ex in enumerate(selected, start=1))
    return "\n\n".join(blocks)


def build_solver_prompt(problem: str, examples: list[dict], max_examples: int = 4) -> str:
    context = build_retrieval_context(problem, examples, max_examples=max_examples)
    return (
        "You are a careful mathematical reasoning assistant.\n"
        "Use the retrieval context as mathematical hints, not as text to copy. First identify the likely topic, "
        "then choose a proof or solution strategy, then solve the target problem rigorously.\n\n"
        f"{context}\n\n"
        f"Target problem:\n{problem}\n\n"
        "Write a concise but complete step-by-step solution. End with a clearly marked final answer.\n\nSolution:"
    )


def format_compact_hint(example: dict, index: int) -> str:
    strategies = example.get("strategies", []) or ["unknown"]
    topics = example.get("topics", []) or [example.get("type", "unknown")]
    theorem_patterns = example.get("theorem_patterns", []) or []
    strategy = strategies[0]
    theorem_text = f"; theorem pattern: {theorem_patterns[0]}" if theorem_patterns else ""
    return (
        f"Hint {index}: category={example.get('type', 'unknown')}; level={example.get('level', 'unknown')}; "
        f"topic={topics[0]}; strategy={strategy}{theorem_text}; "
        f"similarity={example.get('similarity_score', example.get('score', 0.0)):.3f}. "
        "Use this only as a high-level direction, not as a solution template."
    )


def build_compact_solver_prompt(problem: str, examples: list[dict], max_examples: int = 4) -> str:
    selected = examples[:max_examples]
    signals = format_strategy_summary(problem, selected)
    hints = "\n".join(format_compact_hint(example, idx) for idx, example in enumerate(selected, start=1))
    return (
        "You are a careful mathematical reasoning assistant.\n"
        "Use the compact retrieval hints only to choose a strategy. Do not copy a retrieved solution; none is provided.\n\n"
        f"{signals}\n\n"
        f"Compact retrieval hints:\n{hints}\n\n"
        f"Target problem:\n{problem}\n\n"
        "Write a concise but complete step-by-step solution. End with a clearly marked final answer.\n\nSolution:"
    )
