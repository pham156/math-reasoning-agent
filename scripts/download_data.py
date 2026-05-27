from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import data_dir, print_storage_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and save the MATH dataset to scratch.")
    parser.add_argument("--dataset", default="hendrycks/competition_math")
    parser.add_argument(
        "--fallback_dataset",
        default="SuperSecureHuman/competition_math_hf_dataset",
        help="Public MATH mirror used if the canonical dataset id is unavailable.",
    )
    parser.add_argument("--output_dir", default=str(data_dir("MATH")))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    print_storage_summary()
    print(f"Dataset: {args.dataset}")
    print(f"Saving dataset to: {output_dir}")

    if "/scratch/gilbreth/" not in str(output_dir):
        raise ValueError(f"Refusing to write dataset outside scratch: {output_dir}")

    from datasets import load_dataset

    try:
        dataset = load_dataset(args.dataset)
        dataset_name = args.dataset
    except Exception as exc:
        print(f"WARNING: failed to load {args.dataset}: {type(exc).__name__}: {exc}")
        print(f"Trying fallback dataset: {args.fallback_dataset}")
        dataset = load_dataset(args.fallback_dataset)
        dataset_name = args.fallback_dataset

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Saved dataset {dataset_name} to {output_dir}")


if __name__ == "__main__":
    main()
