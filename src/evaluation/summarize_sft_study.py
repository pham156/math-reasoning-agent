"""Summarize SFT variants (ground-truth vs RFT data, full vs LoRA) on dev300 and external sets."""

import argparse
import json
from pathlib import Path

from src.core.config import ensure_scratch_path, output_dir
from src.evaluation.aime_benchmarks import read_shards
from src.evaluation.pilot_benchmarks import summarize_rows
from src.training.grpo.train import write_json

STUDY = output_dir("rl_1p5b_study")


def only_subdir(folder):
    (child,) = [p for p in Path(folder).iterdir() if p.is_dir()]
    return child


def dev300_rows(directory, model_path):
    rows = []
    for shard in range(3):
        config = json.loads((directory / f"shard_{shard}_config.json").read_text())
        part = [json.loads(line) for line in (directory / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip()]
        if (Path(config["model_path"]).resolve() != Path(model_path).resolve() or len(part) != 100 or
                [r["source_index"] for r in part] != config["source_indices"] or config["max_new_tokens"] != 3000):
            raise ValueError(f"Incomplete dev300 shard: {directory}/{shard}")
        rows.extend(part)
    return rows


def references():
    controls = json.loads((STUDY / "controls_exploration_20261014/summary.json").read_text())["greedy_controls"]
    external = json.loads((STUDY / "external_benchmarks_20261015/summary.json").read_text())["runs"]
    out = {}
    for name, label in (("baseline", "No RL (base)"), ("grpo", "GRPO LR1e-6, 128 updates")):
        out[label] = {"dev300": next(r["accuracy"] for r in controls if r["name"] == name)}
        out[label].update({r["benchmark"]: r["accuracy"] for r in external if r["name"] == name})
    out["Original SFT (GT, full, LR1e-5)"] = {"dev300": next(r["accuracy"] for r in controls if r["name"] == "sft")}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    m = json.loads((root / "submission.json").read_text())
    results = []
    for arm in m["arms"]:
        model = arm["model_path"]
        evals = root / "evaluations" / arm["label"]
        entry = {"label": arm["label"], "description": arm["description"], "model_path": model}
        if (root / "arms" / arm["label"] / "sft_config.json").exists():
            entry["sft_config"] = json.loads((root / "arms" / arm["label"] / "sft_config.json").read_text())
        if (evals / "dev300").exists():
            entry["dev300"] = summarize_rows(dev300_rows(only_subdir(evals / "dev300/greedy"), model))
        for bench, frozen in m["benchmarks"].items():
            if not (evals / bench).exists() and m.get("optional_benchmarks"):
                continue
            entry[bench] = summarize_rows(read_shards(only_subdir(evals / bench / "greedy"), Path(frozen), "greedy", model))
        results.append(entry)
    note = "Single seed; matched 512 training problems and prompt. Greedy, common stop (endoftext + im_end), 3000-token cap, exact/Math-Verify grader. RFT targets are verified base-model samples, not human solutions."
    write_json(root / "summary.json", {"arms": results, "references": references(), "note": note})
    def pct(d, k):
        return f"{d[k]['accuracy']:.1%}" if k in d else "---"

    lines = ["# SFT study", "", note, "", "| Arm | Train examples | Dev300 | MATH-500 | GSM8K-100 | AIME 25+26 | Dev300 tokens | Dev300 truncated |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        dev = r.get("dev300", {})
        lines.append(f"| {r['label']} | {r.get('sft_config', {}).get('train_n', '---')} | {pct(r, 'dev300')} | {pct(r, 'math500')} | {pct(r, 'gsm8k100')} | {pct(r, 'aime')} | {dev.get('avg_generated_tokens', float('nan')):.0f} | {dev.get('truncated', '---')} |")
    lines += ["", "References (existing results): " + json.dumps(references())]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
