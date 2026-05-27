from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.core.config import data_dir, model_dir, print_storage_summary
from src.retrieval.math_taxonomy import enrich_example
from src.retrieval.embedding import embedding_text, load_embedding_model


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


def build_faiss_index(
    dataset_dir: str | Path,
    embedding_model: str | Path,
    output_dir: str | Path,
    split: str = "train",
    batch_size: int = 64,
    embedding_mode: str = "problem_strategy",
) -> dict:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"MATH dataset not found at {dataset_dir}. Run: python scripts/download_data.py"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    import faiss
    import numpy as np

    model = load_embedding_model(embedding_model)
    examples = [enrich_example(example) for example in iter_examples(dataset_dir, split)]
    if not examples:
        raise ValueError(f"No examples found in split {split!r} at {dataset_dir}")
    texts = [embedding_text(example, mode=embedding_mode) for example in examples]

    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb.astype("float32"))

    matrix = np.vstack(embeddings)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_path = output_dir / "math.index"
    metadata_path = output_dir / "metadata.jsonl"
    config_path = output_dir / "index_config.json"

    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    config = {
        "dataset_dir": str(dataset_dir),
        "split": split,
        "embedding_model": str(embedding_model),
        "embedding_mode": embedding_mode,
        "num_examples": len(examples),
        "embedding_dim": int(matrix.shape[1]),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


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
    config = build_faiss_index(
        dataset_dir=args.dataset_dir,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        embedding_mode=args.embedding_mode,
    )

    print(f"Wrote FAISS index: {config['index_path']}")
    print(f"Wrote metadata: {config['metadata_path']}")
    print(f"Wrote index config: {Path(args.output_dir) / 'index_config.json'}")


if __name__ == "__main__":
    main()
