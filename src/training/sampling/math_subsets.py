from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from src.core.config import data_dir, ensure_scratch_path, output_dir


def _level_number(value: object) -> int | None:
    text = str(value or "").lower().replace("level", "").strip()
    return int(text) if text.isdigit() else None


def build_subset(name: str, limit: int, seed: int) -> Dataset:
    if name == "gsm8k":
        gsm8k_root = data_dir("GSM8K")
        try:
            loaded = load_from_disk(str(gsm8k_root))
            source = loaded["train"]
        except FileNotFoundError:
            try:
                source = load_from_disk(str(gsm8k_root / "train"))
            except FileNotFoundError:
                source = load_dataset("openai/gsm8k", "main", split="train")
    else:
        source = load_from_disk(str(data_dir("MATH")))["train"]
        source = source.map(lambda _row, index: {"source_index": index}, with_indices=True)
        if name == "math_level_1_2":
            source = source.filter(lambda row: (_level_number(row.get("level")) or 0) in {1, 2})
        elif name == "math_level_3_5":
            source = source.filter(lambda row: (_level_number(row.get("level")) or 0) in {3, 4, 5})
        elif name != "math_random":
            raise ValueError(f"Unknown subset: {name}")
    source = source.shuffle(seed=seed)
    return source.select(range(min(limit, len(source)))) if limit > 0 else source


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible RL training subsets.")
    parser.add_argument("--subset", choices=["math_random", "math_level_1_2", "math_level_3_5", "gsm8k"], required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    target = Path(args.output) if args.output else output_dir("rl_experiments") / "subsets" / f"{args.subset}_{args.limit}_seed{args.seed}"
    target = ensure_scratch_path(target, "output")
    subset = build_subset(args.subset, args.limit, args.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(target))
    manifest = {"subset": args.subset, "limit": args.limit, "seed": args.seed, "n": len(subset), "path": str(target)}
    (target.parent / f"{target.name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
