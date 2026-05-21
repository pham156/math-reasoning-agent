from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import data_dir, model_dir
from src.math_taxonomy import labels_for_query
from src.retrieval_context import build_retrieval_context, build_solver_prompt
from src.retrieval_embedding import embedding_text, load_embedding_model
from src.retrieval_rerank import rerank_examples


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MathRetriever:
    def __init__(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        embedding_model: str | Path | None = None,
        embedding_mode: str = "problem_strategy",
    ) -> None:
        self.index_path = Path(index_path or data_dir("faiss/math.index"))
        self.metadata_path = Path(metadata_path or data_dir("faiss/metadata.jsonl"))
        self.embedding_mode = embedding_mode
        self.model = load_embedding_model(embedding_model or model_dir("all-MiniLM-L6-v2"))
        import faiss

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = load_metadata(self.metadata_path)

    def _query_text(self, problem: str) -> str:
        labels = labels_for_query(problem)
        pseudo_example = {
            "problem": problem,
            "solution": "",
            "topics": labels.get("topics", []),
            "strategies": labels.get("strategies", []),
            "theorem_patterns": labels.get("theorem_patterns", []),
            "level": "",
        }
        return embedding_text(pseudo_example, mode=self.embedding_mode)

    def search(self, problem: str, candidate_k: int = 30) -> list[dict]:
        query = self.model.encode(
            [self._query_text(problem)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(query, candidate_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            item["faiss_rank"] = rank
            item["semantic_score"] = float(score)
            item["score"] = float(score)
            results.append(item)
        return results

    def retrieve(self, problem: str, top_k: int = 5, candidate_k: int = 30) -> list[dict]:
        candidates = self.search(problem, candidate_k=max(candidate_k, top_k))
        return rerank_examples(problem, candidates, top_k=top_k)


def retrieve_examples(
    problem: str,
    top_k: int = 5,
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    embedding_model: str | Path | None = None,
    candidate_k: int = 30,
    embedding_mode: str = "problem_strategy",
) -> list[dict]:
    retriever = MathRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
        embedding_mode=embedding_mode,
    )
    return retriever.retrieve(problem, top_k=top_k, candidate_k=candidate_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve similar solved MATH examples.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--show_json", action="store_true")
    parser.add_argument("--show_context", action="store_true")
    parser.add_argument("--show_solver_prompt", action="store_true")
    args = parser.parse_args()

    results = retrieve_examples(args.problem, args.top_k, candidate_k=args.candidate_k)
    if args.show_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    if args.show_context:
        print(build_retrieval_context(args.problem, results))
        return
    if args.show_solver_prompt:
        print(build_solver_prompt(args.problem, results))
        return

    for i, item in enumerate(results, start=1):
        print(
            f"\n[{i}] semantic={item.get('semantic_score', 0.0):.4f} "
            f"rerank={item.get('rerank_score', 0.0):.4f} type={item.get('type')} level={item.get('level')}"
        )
        print(f"Topics: {', '.join(item.get('topics', []))}")
        print(f"Strategies: {', '.join(item.get('strategies', []))}")
        print(f"Theorem patterns: {', '.join(item.get('theorem_patterns', []))}")
        print(f"Problem: {item.get('problem', '')}")
        print(f"Solution: {item.get('solution', '')[:1200]}")


if __name__ == "__main__":
    main()
