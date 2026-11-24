"""Freeze AIME 2025+2026 (60 problems) and summarize checkpoint/SFT external evaluations.

AIME is tiny (30 problems/year; one problem = 3.3 points), so each model gets greedy
accuracy plus eight samples at temperature 1 (mean@8 = sampled pass@1, and pass@8).
The prompt, grader, stop tokens and 3000-token cap match every other evaluation.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from src.core.config import data_dir, ensure_scratch_path
from src.evaluation.pilot_benchmarks import problem_hash, summarize_rows
from src.training.grpo.train import write_json

AIME_YEARS = (("2025", "AIME_2025", "math-ai/aime25"), ("2026", "AIME_2026", "math-ai/aime26"))
GREEDY_SHARD, SAMPLED_SHARD = 20, 10


def prepare(root):
    from datasets import load_from_disk
    from src.training.rl.smoke import conversation

    train = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
    train_hashes = {problem_hash(r["problem"]) for r in train}
    manifest = json.loads(data_dir("benchmark_manifest_selected.json").read_text())
    revisions = {b["dataset_id"]: b.get("download_revision") for b in manifest["benchmarks"]}
    rows = []
    for year, folder, dataset_id in AIME_YEARS:
        split = load_from_disk(str(data_dir(folder)))["test"]
        if len(split) != 30:
            raise ValueError(f"Expected 30 AIME {year} problems")
        for index, row in enumerate(split):
            answer = str(row["answer"]).strip()
            if not answer.isdigit() or not 0 <= int(answer) <= 999:
                raise ValueError(f"Non-AIME answer in {year}/{index}")
            if problem_hash(row["problem"]) in train_hashes:
                raise ValueError("AIME problem overlaps training data")
            rows.append({"source_index": int(year) * 100 + index, "problem_id": f"aime{year}-{row['id']}", "problem": row["problem"], "answer": answer, "solution": "", "type": f"AIME {year}", "level": "AIME", "prompt": conversation(row["problem"])})
    if len({problem_hash(r["problem"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate AIME problems")
    metadata = {"benchmark": "aime2025_2026", "datasets": {d: revisions.get(d) for _, _, d in AIME_YEARS}, "n": len(rows), "training_overlap": 0, "greedy_shard_size": GREEDY_SHARD, "sampled_shard_size": SAMPLED_SHARD, "allow_sampled": True, "purpose": "External evaluation only; never used for training, tuning or checkpoint selection"}
    payload = {"metadata": metadata, "rows": rows}
    path = root / "aime_frozen.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Refusing to change frozen AIME benchmark")
    write_json(path, payload)
    print(f"Frozen AIME 2025+2026: {len(rows)} problems; no training overlap")


def read_shards(directory, frozen_path, mode, model_path):
    frozen = json.loads(frozen_path.read_text())
    meta = frozen["metadata"]
    size = meta.get(f"{mode}_shard_size", 100)
    expected = [r["source_index"] for r in frozen["rows"]]
    digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    rows = []
    for shard in range(len(expected) // size):
        config = json.loads((directory / f"shard_{shard}_config.json").read_text())
        part = [json.loads(line) for line in (directory / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip()]
        if (config["benchmark_sha256"] != digest or Path(config["model_path"]).resolve() != Path(model_path).resolve() or
                config["max_new_tokens"] != 3000 or [r["source_index"] for r in part] != expected[shard * size:(shard + 1) * size] or
                [r["prompt_hash"] for r in part] != config["prompt_hashes"]):
            raise ValueError(f"Incomplete or mismatched shard: {directory}/{shard}")
        rows.extend(part)
    return rows


def sampled_summary(rows):
    n = len(rows)
    return {"n": n, "mean_at_8": sum(r["pass_at_k"]["1"] for r in rows) / n, "pass_at_8": sum(r["pass_at_k"]["8"] for r in rows) / n,
            "avg_generated_tokens": sum(c["generated_tokens"] for r in rows for c in r["candidates"]) / (8 * n),
            "truncated_fraction": sum(c["truncated"] for r in rows for c in r["candidates"]) / (8 * n)}


def summarize(root):
    m = json.loads((root / "submission.json").read_text())
    frozen = Path(m["aime_frozen"])
    results = []
    for label, model in m["models"].items():
        base = root / "evaluations" / label / "aime"
        greedy = read_shards(next((base / "greedy").iterdir()), frozen, "greedy", model)
        entry = {"model": label, "model_path": model, "greedy": summarize_rows(greedy), "greedy_by_year": {}}
        groups = defaultdict(list)
        for r in greedy:
            groups[r["type"]].append(r)
        entry["greedy_by_year"] = {k: summarize_rows(v) for k, v in sorted(groups.items())}
        if (base / "sampled").exists():
            sampled = read_shards(next((base / "sampled").iterdir()), frozen, "sampled", model)
            entry["sampled"] = sampled_summary(sampled)
            groups = defaultdict(list)
            for r in sampled:
                groups[r["type"]].append(r)
            entry["sampled_by_year"] = {k: sampled_summary(v) for k, v in sorted(groups.items())}
        results.append(entry)
    note = "AIME 2025+2026 (60 problems), greedy and 8 samples at T=1, 3000-token cap, common stop, exact/Math-Verify grader. Single training seed; one AIME problem is 3.3 points per year, so small differences are noise."
    write_json(root / "summary.json", {"runs": results, "note": note})
    lines = ["# AIME 2025/2026 checkpoint evaluation", "", note, "", "| Model | Greedy 2025 | Greedy 2026 | Greedy all | mean@8 all | pass@8 all | Greedy tokens | Truncated (greedy) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        y = r["greedy_by_year"]
        s = r.get("sampled", {})
        lines.append(f"| {r['model']} | {y['AIME 2025']['accuracy']:.1%} | {y['AIME 2026']['accuracy']:.1%} | {r['greedy']['accuracy']:.1%} | {s.get('mean_at_8', float('nan')):.1%} | {s.get('pass_at_8', float('nan')):.1%} | {r['greedy']['avg_generated_tokens']:.0f} | {r['greedy']['truncated']} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    prepare(root) if args.prepare else summarize(root)


if __name__ == "__main__":
    main()
