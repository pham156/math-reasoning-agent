"""Freeze external pilot benchmarks and summarize complete common-stop runs."""

import argparse
from collections import defaultdict
import hashlib
import json

from src.core.config import data_dir, ensure_scratch_path
from src.training.grpo.train import write_json


NAMES = ("baseline", "grpo", "dr_grpo", "dapo_style", "gspo")
BENCHMARKS = {"math500": 500, "gsm8k100": 100}


def problem_hash(problem):
    return hashlib.sha256(" ".join(problem.split()).encode()).hexdigest()


def prepare(root):
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi
    from src.evaluation.benchmarks import load_math500, load_gsm8k100
    from src.evaluation.answer import extract_final_answer
    from src.training.rl.smoke import conversation

    train = load_from_disk(str(data_dir("rl_1p5b_pilot_v1/train")))
    train_hashes = {problem_hash(r["problem"]) for r in train}
    gsm_path = data_dir("GSM8K")
    if not (gsm_path / "dataset_dict.json").exists():
        revision = HfApi().dataset_info("openai/gsm8k").sha
        dataset = load_dataset("openai/gsm8k", "main", revision=revision)
        dataset.save_to_disk(str(gsm_path))
        write_json(root / "gsm8k_download.json", {"dataset_id": "openai/gsm8k", "revision": revision})
    for name, loader in (("math500", load_math500), ("gsm8k100", load_gsm8k100)):
        source, metadata = loader(limit=BENCHMARKS[name], sample_seed=42, sampling="random")
        rows = []
        for row in source:
            answer = row["ground_truth_answer_override"] or extract_final_answer(row["ground_truth_solution"])
            if not answer or problem_hash(row["problem"]) in train_hashes:
                raise ValueError(f"Empty answer or training overlap: {row['problem_id']}")
            rows.append({"source_index": row["problem_index"], "problem_id": row["problem_id"], "problem": row["problem"], "answer": answer, "solution": row["ground_truth_solution"], "type": row["type"], "level": row["level"], "prompt": conversation(row["problem"])})
        if len(rows) != BENCHMARKS[name] or len({problem_hash(r["problem"]) for r in rows}) != len(rows):
            raise ValueError("Benchmark count or uniqueness mismatch")
        metadata.update(training_overlap=0, training_count=len(train), purpose="External transfer evaluation; do not use for tuning", previously_inspected_in_7b_project=True)
        if name == "gsm8k100" and (root / "gsm8k_download.json").exists():
            metadata["dataset_revision"] = json.loads((root / "gsm8k_download.json").read_text())["revision"]
        payload = {"metadata": metadata, "rows": rows}
        path = root / f"{name}_frozen.json"
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError("Refusing to change frozen benchmark")
        write_json(path, payload)
        print(f"Frozen {name}: {len(rows)} problems; no normalized-text training overlap", flush=True)


def summarize_rows(rows):
    n = len(rows)
    return {"n": n, "accuracy": sum(r["candidates"][0]["correct"] for r in rows) / n,
            "truncated": sum(r["candidates"][0]["truncated"] for r in rows),
            "avg_generated_tokens": sum(r["candidates"][0]["generated_tokens"] for r in rows) / n,
            "avg_input_tokens": sum(r["input_tokens"] for r in rows) / n,
            "avg_runtime": sum(r["runtime_seconds"] for r in rows) / n}


def summarize(root):
    results = []
    for benchmark, total in BENCHMARKS.items():
        frozen_path = root / f"{benchmark}_frozen.json"
        frozen = json.loads(frozen_path.read_text())
        expected = [r["source_index"] for r in frozen["rows"]]
        baseline = None
        baseline_hashes = None
        for name in NAMES:
            rows, hashes = [], []
            for shard in range(total // 100):
                directory = root / benchmark / "greedy" / name
                config = json.loads((directory / f"shard_{shard}_config.json").read_text())
                if config["benchmark_sha256"] != hashlib.sha256(frozen_path.read_bytes()).hexdigest():
                    raise ValueError("Benchmark inputs changed")
                part = [json.loads(line) for line in (directory / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip()]
                if [r["source_index"] for r in part] != expected[shard * 100:(shard + 1) * 100]:
                    raise ValueError(f"Incomplete or changed shard: {benchmark}/{name}/{shard}")
                if [r["prompt_hash"] for r in part] != config["prompt_hashes"] or any(len(r["candidates"]) != 1 for r in part):
                    raise ValueError("Prompt or candidate mismatch")
                rows.extend(part)
                hashes.extend(config["prompt_hashes"])
            correct = [r["candidates"][0]["correct"] for r in rows]
            if baseline is None:
                baseline, baseline_hashes = correct, hashes
            if hashes != baseline_hashes:
                raise ValueError("Model prompt mismatch")
            result = {"benchmark": benchmark, "name": name, **summarize_rows(rows),
                      "fixed": [i for i, a, b in zip(expected, baseline, correct) if not a and b],
                      "broken": [i for i, a, b in zip(expected, baseline, correct) if a and not b]}
            for key in ("type", "level"):
                groups = defaultdict(list)
                for row in rows:
                    groups[row[key] or "unspecified"].append(row)
                result[f"by_{key}"] = {k: summarize_rows(v) for k, v in groups.items()}
            results.append(result)
    write_json(root / "summary.json", {"runs": results, "note": "One training seed; previously inspected benchmarks, not untouched confirmation data. Greedy common-stop, max3000, exact/Math-Verify grader."})
    lines = ["# External math benchmark evaluation", "", "One training seed; no retrieval or verifier. These benchmarks were inspected during older 7B experiments and are not untouched confirmation sets.", "", "| Benchmark | Model | n | Accuracy | Truncated | Avg tokens | Avg seconds | Fixes / breaks vs base |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['benchmark']} | {r['name']} | {r['n']} | {r['accuracy']:.1%} | {r['truncated']} | {r['avg_generated_tokens']:.1f} | {r['avg_runtime']:.1f} | {len(r['fixed'])} / {len(r['broken'])} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


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
