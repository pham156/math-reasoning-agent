from __future__ import annotations

import json
import time

from src.core.config import data_dir, print_storage_summary


BENCHMARKS = [
    {
        "name": "MATH",
        "dataset_id": "local:$DATA_DIR/MATH",
        "target": "MATH",
        "already_local": True,
        "note": "Local competition_math-compatible dataset already downloaded for retrieval/evaluation.",
    },
    {
        "name": "MATH-500",
        "dataset_id": "HuggingFaceH4/MATH-500",
        "target": "MATH-500",
        "kwargs": {},
        "note": "Common 500-problem MATH subset.",
    },
    {
        "name": "GSM8K",
        "dataset_id": "openai/gsm8k",
        "target": "GSM8K",
        "kwargs": {"name": "main"},
        "note": "Full GSM8K dataset; evaluation preset samples GSM8K-100 from test.",
    },
    {
        "name": "AIME 2024",
        "dataset_id": "HuggingFaceH4/aime_2024",
        "target": "AIME_2024",
        "kwargs": {},
        "note": "30 AIME 2024 problems.",
    },
    {
        "name": "AIME 2025",
        "dataset_id": "math-ai/aime25",
        "target": "AIME_2025",
        "kwargs": {},
        "note": "30 AIME 2025 problems.",
    },
    {
        "name": "GPQA",
        "dataset_id": "Idavidrein/gpqa",
        "target": "GPQA",
        "kwargs": {},
        "optional": True,
        "note": "Official GPQA is gated on Hugging Face and may require authentication.",
    },
    {
        "name": "miniF2F Lean4",
        "dataset_id": "cat-searcher/minif2f-lean4",
        "target": "miniF2F-Lean4",
        "kwargs": {},
        "note": "Lean4 miniF2F validation/test formal proof benchmark.",
    },
]


def dataset_size(dataset) -> dict:
    if hasattr(dataset, "items"):
        return {split: len(split_dataset) for split, split_dataset in dataset.items()}
    return {"rows": len(dataset)}


def save_hf_dataset(spec: dict) -> dict:
    from datasets import load_dataset, load_from_disk

    target = data_dir(spec["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        dataset = load_from_disk(str(target))
        return {
            "name": spec["name"],
            "dataset_id": spec["dataset_id"],
            "target": str(target),
            "status": "present",
            "splits": dataset_size(dataset),
            "note": spec.get("note", ""),
        }

    kwargs = dict(spec.get("kwargs", {}))
    config_name = kwargs.pop("name", None)
    if config_name:
        dataset = load_dataset(spec["dataset_id"], config_name, **kwargs)
    else:
        dataset = load_dataset(spec["dataset_id"], **kwargs)
    dataset.save_to_disk(str(target))
    return {
        "name": spec["name"],
        "dataset_id": spec["dataset_id"],
        "target": str(target),
        "status": "downloaded",
        "splits": dataset_size(dataset),
        "note": spec.get("note", ""),
    }


def check_local_dataset(spec: dict) -> dict:
    target = data_dir(spec["target"])
    return {
        "name": spec["name"],
        "dataset_id": spec["dataset_id"],
        "target": str(target),
        "status": "present" if target.exists() else "missing",
        "note": spec.get("note", ""),
    }


def main() -> None:
    print_storage_summary()
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": str(data_dir()),
        "benchmarks": [],
    }

    for spec in BENCHMARKS:
        print(f"\n== {spec['name']} ==")
        if spec.get("already_local"):
            record = check_local_dataset(spec)
            print(json.dumps(record, indent=2))
            manifest["benchmarks"].append(record)
            continue

        try:
            record = save_hf_dataset(spec)
        except Exception as exc:
            record = {
                "name": spec["name"],
                "dataset_id": spec["dataset_id"],
                "target": str(data_dir(spec["target"])),
                "status": "failed_optional" if spec.get("optional") else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": spec.get("note", ""),
            }
            print(json.dumps(record, indent=2))
            manifest["benchmarks"].append(record)
            if not spec.get("optional"):
                raise
            continue

        print(json.dumps(record, indent=2))
        manifest["benchmarks"].append(record)

    manifest_path = data_dir("benchmark_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark manifest to {manifest_path}")


if __name__ == "__main__":
    main()
