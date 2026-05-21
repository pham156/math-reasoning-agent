from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.config import data_dir, model_dir, print_storage_summary
from src.math_taxonomy import enrich_example
from src.retrieval_embedding import embedding_text, load_embedding_model


def iter_examples(dataset_root: Path, split: str):
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_root))
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found. Available splits: {list(dataset.keys())}")
    for i, row in enumerate(dataset[split]):
        yield {
            "id": f"{split}-{i}",
            "problem": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "type": row.get("type", row.get("category", "")),
            "level": row.get("level", ""),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index over MATH training examples.")
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--embedding_model", default=str(model_dir("all-MiniLM-L6-v2")))
    parser.add_argument("--output_dir", default=str(data_dir("faiss")))
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--embedding_mode",
        default="problem_strategy",
        choices=["problem_only", "problem_solution", "problem_strategy"],
        help="Text fields used for embedding. problem_strategy adds topic/strategy/theorem tags.",
    )
    args = parser.parse_args()

    print_storage_summary()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import faiss
    import numpy as np

    model = load_embedding_model(args.embedding_model)
    examples = [enrich_example(example) for example in iter_examples(dataset_dir, args.split)]
    texts = [embedding_text(example, mode=args.embedding_mode) for example in examples]

    embeddings = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[start : start + args.batch_size]
        emb = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb.astype("float32"))

    matrix = np.vstack(embeddings)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(output_dir / "math.index"))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    with (output_dir / "index_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_dir": str(dataset_dir),
                "split": args.split,
                "embedding_model": args.embedding_model,
                "embedding_mode": args.embedding_mode,
                "num_examples": len(examples),
                "embedding_dim": int(matrix.shape[1]),
            },
            f,
            indent=2,
        )

    print(f"Wrote FAISS index: {output_dir / 'math.index'}")
    print(f"Wrote metadata: {output_dir / 'metadata.jsonl'}")
    print(f"Wrote index config: {output_dir / 'index_config.json'}")


if __name__ == "__main__":
    main()
