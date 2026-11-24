"""Frozen prompt-sensitivity and external-consensus evaluations; no training."""

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from src.core.config import ensure_scratch_path
from src.evaluation.answer import extract_final_answer
from src.evaluation.checkpoint_ablations import digest, load_condition, metrics, read, verify_sources, write
from src.evaluation.pilot_exploration import stop_ids


PROMPTS = {
    "original": r"Please reason step by step, and put your final answer within \boxed{}.",
    "reworded": r"Solve this mathematics problem step by step. Put the final answer in \boxed{}.",
    "concise": r"Solve the problem with concise, justified mathematical reasoning. Put the final answer in \boxed{}.",
}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def input_rows(manifest, condition):
    if condition["dataset"] == "development":
        source = read(Path(manifest["source_campaign"]) / "campaign.json")["cohort"]
        rows = [{**r, "answer": r["reference_answer"]} for r in source]
    else:
        rows = read(manifest["benchmarks"][condition["dataset"]])["rows"]
    selected = []
    for row in rows[:condition["n"]]:
        messages = [{"role": "system", "content": manifest["prompts"][condition["prompt_variant"]]},
                    {"role": "user", "content": row["problem"]}]
        if condition["dataset"] != "development" and messages != row["prompt"]:
            raise ValueError("External confirmation must preserve frozen benchmark prompts")
        selected.append({key: row[key] for key in ("source_index", "problem", "answer", "type", "level")}
                        | {"prompt": messages})
    if len(selected) != condition["n"] or len({r["source_index"] for r in selected}) != len(selected):
        raise ValueError("Wrong input cohort")
    return selected


@lru_cache(maxsize=4)
def tokenizer_for(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prompt_hashes(rows, tokenizer):
    return [hashlib.sha256(tokenizer.apply_chat_template(r["prompt"], tokenize=False,
            add_generation_prompt=True).encode()).hexdigest() for r in rows]


def shard_rows(manifest, condition, shard):
    if shard not in condition["shards"]:
        raise ValueError("Unknown shard")
    rows = input_rows(manifest, condition)
    size = condition["shard_size"]
    return rows[shard * size:(shard + 1) * size]


def validate_condition(manifest, condition):
    rows, evidence = load_condition(condition, n=condition["n"])
    expected = input_rows(manifest, condition)
    tokenizer = tokenizer_for(condition["model_path"])
    hashes = prompt_hashes(expected, tokenizer)
    for row, source, prompt_hash in zip(rows, expected, hashes):
        if (any(row[key] != source[key] for key in ("source_index", "problem", "type", "level")) or
                row["reference_answer"] != source["answer"] or row["prompt_hash"] != prompt_hash):
            raise ValueError("Changed source, answer or rendered chat-template prompt")
    for shard in condition["shards"]:
        folder = Path(condition["root"]) / condition["mode"] / condition["name"]
        config = read(folder / f"shard_{shard}_config.json")
        if (config["condition_sha256"] != object_hash(condition) or
                config["input_sha256"] != object_hash(shard_rows(manifest, condition, shard))):
            raise ValueError("Changed frozen condition/input records")
    return rows, evidence


def generate(manifest, condition, shard):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
    from transformers.trainer_utils import enable_full_determinism
    from src.training.rl.smoke import grade_solution

    model_path = ensure_scratch_path(condition["model_path"])
    folder = ensure_scratch_path(condition["root"]) / condition["mode"] / condition["name"]
    folder.mkdir(parents=True, exist_ok=True)
    rows = shard_rows(manifest, condition, shard)
    tokenizer = tokenizer_for(str(model_path))
    hashes = prompt_hashes(rows, tokenizer)
    eos = stop_ids(tokenizer, GenerationConfig.from_pretrained(model_path, local_files_only=True))
    if eos != [151643, 151645]:
        raise ValueError("Changed common-stop convention")
    count = 8 if condition["mode"] == "sampled" else 1
    config = {"name": condition["name"], "mode": condition["mode"], "shard": shard,
              "model_path": str(model_path), "source_indices": [r["source_index"] for r in rows],
              "prompt_hashes": hashes, "stop_ids": eos, "max_new_tokens": condition["cap"],
              "num_samples": count, "sampling_batch": count, "temperature": condition["temperature"],
              "top_p": 1.0, "top_k": 0, "seed_rule": "100000 + source_index for each sampled group",
              "precision": "FP32 parameters/BF16 compute", "grader": "pilot exact/Math-Verify",
              "condition_sha256": object_hash(condition), "input_sha256": object_hash(rows)}
    config_path = folder / f"shard_{shard}_config.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("Refuse changed evaluation resume")
    write(config_path, config)
    path = folder / f"shard_{shard}.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if ([r["source_index"] for r in existing] != config["source_indices"][:len(existing)] or
            [r["prompt_hash"] for r in existing] != hashes[:len(existing)] or len(existing) > len(rows)):
        raise ValueError("Existing output is not the expected prefix")
    if len(existing) == len(rows):
        return
    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.config.use_cache = True
    model.eval()
    started = time.monotonic()
    with path.open("a") as stream:
        for row, prompt_hash in zip(rows[len(existing):], hashes[len(existing):]):
            if time.monotonic() - started >= 12600:
                raise RuntimeError("Shard incomplete at safe deadline; partial results preserved")
            prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            set_seed(100000 + row["source_index"])
            kwargs = {"do_sample": count > 1, "num_return_sequences": count,
                      "max_new_tokens": condition["cap"], "eos_token_id": eos,
                      "pad_token_id": tokenizer.pad_token_id, "use_cache": True}
            if count > 1:
                kwargs.update(temperature=condition["temperature"], top_p=1.0, top_k=0)
            tick = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, **kwargs)
            elapsed = time.monotonic() - tick
            candidates = []
            for i, sequence in enumerate(generated):
                tokens = sequence[inputs["input_ids"].shape[1]:].tolist()
                ended = next((j for j, token in enumerate(tokens) if token in eos), None)
                tokens = tokens[:ended + 1] if ended is not None else tokens
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                candidates.append({"sample_index": i, "raw_solution": raw, "answer": extract_final_answer(raw),
                    "correct": bool(grade_solution(raw, row["answer"])), "generated_tokens": len(tokens),
                    "truncated": ended is None and len(tokens) >= condition["cap"],
                    "ending_token_id": tokens[-1] if tokens else None})
            payload = {key: row[key] for key in ("source_index", "problem", "type", "level")}
            payload.update(reference_answer=row["answer"], prompt_hash=prompt_hash, name=condition["name"],
                mode=condition["mode"], candidates=candidates, num_correct=sum(c["correct"] for c in candidates),
                runtime_seconds=elapsed, input_tokens=inputs["input_ids"].shape[1],
                slurm_job_id=os.environ.get("SLURM_JOB_ID"))
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            print(f"{condition['key']} shard{shard}: source={row['source_index']}", flush=True)


def verify_campaign(manifest):
    verify_sources(manifest)
    parent = read(Path(manifest["source_campaign"]) / "campaign.json")
    verify_sources(parent)
    if manifest["prompts"] != PROMPTS or manifest["models"] != parent["models"]:
        raise ValueError("Changed prompt definitions or retained checkpoints")
    validate_design(manifest)


def validate_design(manifest):
    expected = {(label, "development", variant, "greedy", 0.0, 100)
                for label in ("base", "step128", "step256", "step512") for variant in ("reworded", "concise")}
    expected |= {(label, benchmark, "original", "sampled", 0.7, n)
                 for label in ("base", "step512") for benchmark, n in (("math500", 500), ("gsm8k100", 100))}
    actual = [(c["checkpoint"], c["dataset"], c["prompt_variant"], c["mode"], c["temperature"], c["n"])
              for c in manifest["conditions"]]
    if len(actual) != 12 or set(actual) != expected:
        raise ValueError("Changed or duplicate experimental design")
    tasks = []
    for i, condition in enumerate(manifest["conditions"]):
        size = 100 if condition["mode"] == "greedy" else 20
        if (condition["cap"] != 3000 or condition["shard_size"] != size or
                condition["model_path"] != manifest["models"][condition["checkpoint"]] or
                condition["shards"] != list(range(condition["n"] // size))):
            raise ValueError("Changed generation budget, checkpoint or sharding")
        tasks.extend({"condition": i, "shard": shard,
                      "phase": "prompt" if condition["dataset"] == "development" else "external"}
                     for shard in condition["shards"])
    if manifest["tasks"] != tasks:
        raise ValueError("Task map differs from frozen conditions")


def run_task(campaign, task):
    manifest = read(campaign / "campaign.json")
    verify_campaign(manifest)
    smoke = task is None
    if not smoke:
        verify_smoke(campaign, manifest)
    entries = [(c, 0) for c in manifest["smoke_conditions"]] if smoke else [
        (manifest["conditions"][manifest["tasks"][task]["condition"]], manifest["tasks"][task]["shard"])]
    for condition, shard in entries:
        index = (manifest["smoke_conditions"] if smoke else manifest["conditions"]).index(condition)
        command = ["bash", "scripts/rl/smoke_python.sh", "-m", "src.evaluation.checkpoint_followup",
                   str(campaign), "--phase", "generate", "--condition", str(index), "--shard", str(shard)]
        if smoke:
            command.append("--smoke_condition")
        subprocess.run(command, check=True)
    if smoke:
        subprocess.run(["bash", "scripts/rl/smoke_python.sh", "-m", "src.evaluation.checkpoint_followup",
                        str(campaign), "--phase", "validate_smoke"], check=True)


def validate_smoke(campaign, manifest):
    evidence, summaries = {}, []
    for condition in manifest["smoke_conditions"]:
        rows, hashes = validate_condition(manifest, condition)
        evidence.update(hashes)
        summaries.append({"key": condition["key"], **metrics(rows)})
        if condition["key"] in manifest["smoke_references"]:
            reference = manifest["smoke_references"][condition["key"]]
            old, hashes = load_condition(reference, n=2)
            evidence.update(hashes)
            for before, after in zip(old, rows):
                if (before["prompt_hash"] != after["prompt_hash"] or
                        before["candidates"] != after["candidates"]):
                    raise ValueError("New inference path differs from prior GPU smoke")
    from src.evaluation.saved_candidate_ablations import subset_metrics
    probe = rows[0]["candidates"]
    if len(probe) != 8 or not 0 <= subset_metrics(probe, 4, "plurality")["accuracy"] <= 1:
        raise ValueError("Smoke final-selection validator failed")
    write(campaign / "smoke_passed.json", {"passed": True, "runs": summaries,
          "prior_smoke_candidates_identical": True, "evidence_sha256": evidence})


def verify_smoke(campaign, manifest):
    receipt = read(campaign / "smoke_passed.json")
    if (not receipt["passed"] or not receipt.get("prior_smoke_candidates_identical") or
            len(receipt["runs"]) != len(manifest["smoke_conditions"])):
        raise ValueError("Smoke incomplete")
    for path, expected in receipt["evidence_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Smoke artifacts changed")


def control_rows(manifest, condition):
    if condition["dataset"] == "development":
        parent = read(Path(manifest["source_campaign"]) / "campaign.json")
        control = next(c for c in parent["conditions"] if c["checkpoint"] == condition["checkpoint"]
                       and c["mode"] == "greedy" and c["cap"] == 3000)
        return load_condition(control, parent["cohort"])[0]
    from src.evaluation.checkpoint_benchmarks import benchmark_rows
    control = manifest["external_controls"][condition["checkpoint"]]
    return benchmark_rows(Path(control["root"]) / condition["dataset"], control["name"],
                          condition["model_path"], Path(manifest["benchmarks"][condition["dataset"]]))


def summarize(campaign):
    from collections import defaultdict
    from src.evaluation.checkpoint_ablations import selection
    manifest = read(campaign / "campaign.json")
    verify_campaign(manifest)
    verify_smoke(campaign, manifest)
    results, evidence, diagnostics = [], {}, []
    for condition in manifest["conditions"]:
        rows, hashes = validate_condition(manifest, condition)
        evidence.update(hashes)
        controls = control_rows(manifest, condition)
        if ([r["source_index"] for r in rows] != [r["source_index"] for r in controls] or
                any(a["problem"] != b["problem"] or a["reference_answer"] != b["reference_answer"]
                    for a, b in zip(rows, controls))):
            raise ValueError("Control cohort differs")
        if condition["dataset"] != "development" and [r["prompt_hash"] for r in rows] != [r["prompt_hash"] for r in controls]:
            raise ValueError("External confirmation changed prompts")
        baseline = [r["candidates"][0]["correct"] for r in controls]
        chosen = [r["candidates"][selection(r["candidates"])["selected_index"]]["correct"] for r in rows]
        result = {key: condition[key] for key in ("key", "dataset", "checkpoint", "prompt_variant", "temperature")}
        result.update(metrics(rows))
        result.update(control_greedy_accuracy=sum(baseline) / len(baseline),
            control_avg_completion_tokens=sum(r["candidates"][0]["generated_tokens"] for r in controls) / len(controls),
            fixes_vs_control=[r["source_index"] for r, a, b in zip(rows, baseline, chosen) if not a and b],
            breaks_vs_control=[r["source_index"] for r, a, b in zip(rows, baseline, chosen) if a and not b])
        for field in ("type", "level"):
            groups = defaultdict(list)
            for row in rows:
                groups[row[field] or "unspecified"].append(row)
            result["by_" + field] = {k: metrics(v) for k, v in groups.items()}
        results.append(result)
        for row in rows:
            diagnostics.append({"condition": condition["key"], "source_index": row["source_index"],
                **selection(row["candidates"]), "candidate_answers": [c["answer"] for c in row["candidates"]],
                "candidate_correct": [c["correct"] for c in row["candidates"]]})
    if len(results) != 12:
        raise ValueError("Expected eight prompt conditions plus four external conditions")
    raw = ensure_scratch_path(manifest["raw_root"])
    (raw / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in diagnostics))
    note = ("Prompt variants: greedy, same 100 development questions and checkpoint-specific original-prompt controls. "
            "External: one frozen T=0.7, N=8, cap3000 consensus setting, strict majority with first-candidate fallback. "
            "Base is a matched-decoding control; external scores do not select temperatures/prompts/checkpoints. "
            "Previously inspected external cohorts are transfer checks, not untouched confirmation sets. "
            "No new training, retrieval or verifier. Sampled mean pass@1, first candidate, consensus and oracle are distinct.")
    write(campaign / "summary.json", {"note": note, "runs": results, "evidence_sha256": evidence})
    lines = ["# Checkpoint Follow-up", "", note, "",
             "| Dataset | Checkpoint | Prompt | n | Greedy control | Selected accuracy | Sampled pass@1 | Completion tokens/problem | Fixes / breaks |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['checkpoint']} | {r['prompt_variant']} | {r['n']} | {r['control_greedy_accuracy']:.1%} | {r['selected_accuracy']:.1%} | {r['pass_at_1']:.2%} | {r['avg_problem_tokens']:.1f} | {len(r['fixes_vs_control'])} / {len(r['breaks_vs_control'])} |")
    (campaign / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--phase", choices=("check", "smoke", "validate_smoke", "task", "generate", "summary"), required=True)
    parser.add_argument("--task", type=int)
    parser.add_argument("--condition", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--smoke_condition", action="store_true")
    args = parser.parse_args()
    manifest = read(args.campaign / "campaign.json")
    verify_campaign(manifest)
    if args.phase == "generate":
        if not args.smoke_condition:
            verify_smoke(args.campaign, manifest)
        condition = manifest["smoke_conditions" if args.smoke_condition else "conditions"][args.condition]
        generate(manifest, condition, args.shard)
    elif args.phase == "summary":
        summarize(args.campaign)
    elif args.phase == "validate_smoke":
        validate_smoke(args.campaign, manifest)
    elif args.phase in ("smoke", "task"):
        if args.phase == "task" and (args.task is None or not 0 <= args.task < len(manifest["tasks"])):
            parser.error("Valid task index required")
        run_task(args.campaign, None if args.phase == "smoke" else args.task)


if __name__ == "__main__":
    main()
