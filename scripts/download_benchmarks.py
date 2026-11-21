from __future__ import annotations

import json
from pathlib import Path
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
        "note": "30 AIME 2024 problems. Held February 2024, before the September 2024 Qwen2.5-Math release, so it may be in that model's training data; not a held-out set.",
    },
    {
        "name": "AIME 2025 (legacy copy)",
        "dataset_id": "math-ai/aime25",
        "target": "AIME_2025",
        "kwargs": {},
        "note": "30 AIME 2025 problems used by the older 7B/QLoRA runs (benchmark preset aime2025). Its problem 2 wording differs from MathArena; the 1.5B pilot AIME check uses MathArena below.",
    },
    {
        "name": "AIME 2025 (MathArena)",
        "matharena": "aime2025",
        "note": "Pinned MathArena revision used by `src.evaluation.pilot_benchmarks --suite aime`.",
    },
    {
        "name": "AIME 2026 (MathArena)",
        "matharena": "aime2026",
        "note": "Pinned MathArena revision used by `src.evaluation.pilot_benchmarks --suite aime`.",
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


def revision_record_path(target: Path) -> Path:
    """Sidecar beside the saved dataset recording the Hub commit it was downloaded at."""
    return target.with_name(target.name + ".download.json")


def recorded_revision(target: Path, dataset_id: str) -> str:
    """The recorded Hub revision, or "unrecorded" for copies saved before revisions were kept."""
    record_path = revision_record_path(target)
    if not record_path.exists():
        return "unrecorded"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["dataset_id"] != dataset_id:
        raise ValueError(f"{target} was downloaded from {record['dataset_id']}, expected {dataset_id}")
    return record["revision"]


def save_hf_dataset(spec: dict) -> dict:
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi

    target = data_dir(spec["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        dataset = load_from_disk(str(target))
        return {
            "name": spec["name"],
            "dataset_id": spec["dataset_id"],
            "target": str(target),
            "status": "present",
            "revision": recorded_revision(target, spec["dataset_id"]),
            "splits": dataset_size(dataset),
            "note": spec.get("note", ""),
        }

    # Resolve the current commit once and download exactly that, so the record cannot drift.
    revision = HfApi().dataset_info(spec["dataset_id"]).sha
    kwargs = dict(spec.get("kwargs", {}))
    config_name = kwargs.pop("name", None)
    if config_name:
        dataset = load_dataset(spec["dataset_id"], config_name, revision=revision, **kwargs)
    else:
        dataset = load_dataset(spec["dataset_id"], revision=revision, **kwargs)
    dataset.save_to_disk(str(target))
    revision_record_path(target).write_text(json.dumps({"dataset_id": spec["dataset_id"], "revision": revision}, indent=2) + "\n", encoding="utf-8")
    return {
        "name": spec["name"],
        "dataset_id": spec["dataset_id"],
        "target": str(target),
        "status": "downloaded",
        "revision": revision,
        "splits": dataset_size(dataset),
        "note": spec.get("note", ""),
    }


def save_matharena(spec: dict) -> dict:
    from src.evaluation.benchmarks import MATHARENA_AIME, load_matharena_aime

    rows, metadata = load_matharena_aime(spec["matharena"])
    return {
        "name": spec["name"],
        "dataset_id": metadata["dataset_id"],
        "target": str(data_dir(f"MathArena_{spec['matharena']}")),
        "status": "present" if metadata["source"] == "local_disk" else "downloaded",
        "revision": MATHARENA_AIME[spec["matharena"]]["revision"],
        "splits": {"train": len(rows)},
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

    manifest_path = data_dir("benchmark_manifest.json")
    try:
        for spec in BENCHMARKS:
            print(f"\n== {spec['name']} ==")
            if spec.get("already_local"):
                record = check_local_dataset(spec)
                print(json.dumps(record, indent=2))
                manifest["benchmarks"].append(record)
                continue

            try:
                record = save_matharena(spec) if spec.get("matharena") else save_hf_dataset(spec)
            except Exception as exc:
                record = {
                    "name": spec["name"],
                    "dataset_id": spec.get("dataset_id", spec.get("matharena", "")),
                    "target": str(data_dir(spec["target"] if "target" in spec else f"MathArena_{spec['matharena']}")),
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
    finally:
        # Written even when a required download fails, so completed entries stay recorded.
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote benchmark manifest to {manifest_path}")


if __name__ == "__main__":
    main()
