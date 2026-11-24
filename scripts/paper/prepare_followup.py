"""Validate completed follow-up evidence for paper/poster, explicitly excluding the failed condition."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.core.config import project_root
from src.evaluation.checkpoint_ablations import digest, load_condition, metrics, read
from src.evaluation.checkpoint_followup import control_rows, validate_condition, verify_campaign, verify_smoke
from src.evaluation.saved_candidate_ablations import analyze_rows


MODELS = ("base", "step128", "step256", "step512")
EXCLUDED = "step512_math500_original_sampled"


def validate_payload(data):
    expected = {(model, "development", variant) for model in MODELS for variant in ("reworded", "concise")}
    expected |= {("base", "math500", "original"), ("base", "gsm8k100", "original"), ("step512", "gsm8k100", "original")}
    keys = [(r["checkpoint"], r["dataset"], r["prompt_variant"]) for r in data["completed"]]
    if len(keys) != 11 or set(keys) != expected:
        raise ValueError("Expected exactly eleven complete follow-up conditions")
    for row in data["completed"]:
        if row["n"] != (500 if row["dataset"] == "math500" else 100):
            raise ValueError("Partial cohort cannot be reported as complete")
        if row["samples_per_problem"] != (1 if row["dataset"] == "development" else 8):
            raise ValueError("Wrong candidate count")
    sampled = [(r["checkpoint"], r["temperature"]) for r in data["saved"]]
    if len(sampled) != 12 or set(sampled) != {(m, t) for m in MODELS for t in (.4, .7, 1.0)}:
        raise ValueError("Incomplete saved-response analysis")
    if data["excluded_conditions"] != [EXCLUDED]:
        raise ValueError("The failed MATH-500 condition must remain explicitly excluded")


def load_export(path):
    data = read(path)
    validate_payload(data)
    for file, record in data["sources"].items():
        if digest(file) != record["sha256"]:
            raise ValueError(f"Follow-up evidence changed: {file}")
    return data


def prepare(campaign, destination):
    if destination.exists():
        cached = load_export(destination)
        print(f"Reused validated follow-up export: {len(cached['completed'])} complete conditions")
        return
    manifest = read(campaign / "campaign.json")
    verify_campaign(manifest)
    verify_smoke(campaign, manifest)
    sources = {}

    def record(hashes):
        sources.update({str(path): {"path": str(path), "sha256": value} for path, value in hashes.items()})

    completed = []
    for condition in manifest["conditions"]:
        if condition["key"] == EXCLUDED:
            continue
        rows, hashes = validate_condition(manifest, condition)
        controls = control_rows(manifest, condition)
        if ([r["source_index"] for r in rows] != [r["source_index"] for r in controls] or
                any(a["problem"] != b["problem"] or a["reference_answer"] != b["reference_answer"]
                    for a, b in zip(rows, controls))):
            raise ValueError("Follow-up control cohort mismatch")
        if condition["dataset"] != "development" and [r["prompt_hash"] for r in rows] != [r["prompt_hash"] for r in controls]:
            raise ValueError("External prompts changed")
        record(hashes)
        result = {key: condition[key] for key in ("key", "dataset", "checkpoint", "prompt_variant", "temperature")}
        result.update(metrics(rows))
        result["control_accuracy"] = sum(r["candidates"][0]["correct"] for r in controls) / len(controls)
        result["control_tokens"] = sum(r["candidates"][0]["generated_tokens"] for r in controls) / len(controls)
        completed.append(result)
    parent_path = Path(manifest["source_campaign"]) / "campaign.json"
    parent = read(parent_path)
    saved = []
    for condition in parent["conditions"]:
        rows, hashes = load_condition(condition, parent["cohort"])
        record(hashes)
        if condition["mode"] == "sampled":
            saved.append({"checkpoint": condition["checkpoint"], "temperature": condition["temperature"],
                          **analyze_rows(rows)})
    # Include reused external control configs/outputs and frozen inputs in the evidence chain.
    record(manifest["source_hashes"])
    for path in (campaign / "campaign.json", campaign / "smoke_passed.json", parent_path, Path(__file__)):
        record({str(path): digest(path)})
    data = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "completed": completed,
            "saved": saved, "excluded_conditions": [EXCLUDED], "sources": sources,
            "scope": "This publication snapshot excludes the incomplete step512 MATH-500 consensus cohort and later replacements. Original completed greedy MATH-500 results are unaffected. Saved-response results are retrospective development analyses; external selector remains frozen strict majority with first-candidate fallback."}
    validate_payload(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Prepared eleven complete conditions and twelve recomputed saved-response analyses: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=project_root() / "results/checkpoint_followup_20261115")
    parser.add_argument("--output", type=Path, default=project_root() / "docs/final/generated/followup_evidence.json")
    args = parser.parse_args()
    prepare(args.campaign, args.output)


if __name__ == "__main__":
    main()
