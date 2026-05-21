from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from src.core.config import data_dir, log_dir
from src.retrieval.math_taxonomy import enrich_example
from src.retrieval.retrieve import MathRetriever


def overlap(a: list[str], b: list[str]) -> bool:
    return bool(set(a) & set(b))


def evaluate_one(query: dict, retrieved: list[dict], ks: list[int]) -> dict:
    row = {
        "id": query.get("id"),
        "type": query.get("type", ""),
        "level": query.get("level", ""),
        "topics": query.get("topics", []),
        "strategies": query.get("strategies", []),
        "theorem_patterns": query.get("theorem_patterns", []),
    }
    for k in ks:
        top = retrieved[:k]
        row[f"type_hit@{k}"] = any(item.get("type") == query.get("type") for item in top)
        row[f"topic_hit@{k}"] = any(overlap(query.get("topics", []), item.get("topics", [])) for item in top)
        row[f"strategy_hit@{k}"] = any(overlap(query.get("strategies", []), item.get("strategies", [])) for item in top)
        row[f"theorem_hit@{k}"] = any(overlap(query.get("theorem_patterns", []), item.get("theorem_patterns", [])) for item in top)
        row[f"mean_rerank_score@{k}"] = sum(item.get("rerank_score", 0.0) for item in top) / max(len(top), 1)
    row["top1_type"] = retrieved[0].get("type", "") if retrieved else ""
    row["top1_topics"] = retrieved[0].get("topics", []) if retrieved else []
    row["top1_strategies"] = retrieved[0].get("strategies", []) if retrieved else []
    return row


def summarize(rows: list[dict], ks: list[int]) -> dict:
    summary = {"num_queries": len(rows)}
    for k in ks:
        for metric in ("type_hit", "topic_hit", "strategy_hit", "theorem_hit"):
            key = f"{metric}@{k}"
            summary[key] = sum(bool(row[key]) for row in rows) / max(len(rows), 1)
        score_key = f"mean_rerank_score@{k}"
        summary[score_key] = sum(float(row[score_key]) for row in rows) / max(len(rows), 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval topic/strategy quality on MATH.")
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=50)
    parser.add_argument("--output_json", default=str(log_dir("retrieval_eval_summary.json")))
    parser.add_argument("--output_csv", default=str(log_dir("retrieval_eval_rows.csv")))
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)
    if args.split not in dataset:
        raise KeyError(f"Split {args.split!r} not found. Available splits: {list(dataset.keys())}")

    retriever = MathRetriever()
    examples = []
    for i, row in enumerate(dataset[args.split]):
        if args.num_queries and i >= args.num_queries:
            break
        example = {
            "id": f"{args.split}-{i}",
            "problem": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "type": row.get("type", row.get("category", "")),
            "level": row.get("level", ""),
        }
        examples.append(enrich_example(example))

    ks = sorted({1, 3, 5, args.top_k})
    rows = []
    for query in tqdm(examples, desc="Evaluating retrieval"):
        retrieved = retriever.retrieve(query["problem"], top_k=args.top_k, candidate_k=args.candidate_k)
        retrieved = [item for item in retrieved if item.get("id") != query.get("id")]
        rows.append(evaluate_one(query, retrieved, ks))

    summary = summarize(rows, ks)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary: {output_json}")
    print(f"Wrote per-query rows: {output_csv}")


if __name__ == "__main__":
    main()
