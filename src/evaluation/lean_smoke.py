"""Small zero-shot formal-proof feasibility test; no RL or reference proofs."""

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time

from src.core.config import data_dir, ensure_scratch_path, model_dir, project_root, scratch_root
from src.training.grpo.train import write_json


HEADER = "import Mathlib\nopen scoped BigOperators\nopen Real Nat Topology\nset_option maxHeartbeats 200000\nset_option maxRecDepth 2048\n"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
FORBIDDEN = re.compile(r"\b(sorry|admit|axiom|constant|opaque|unsafe|partial|native_decide|run_tac|run_elab|elab|macro|syntax|import|namespace|section|end|theorem|lemma|def|instance|attribute|set_option|initialize|IO|System|Lean|dbg_trace|trace|eval)\b|[#`]|/-|--")


def proof_body(raw):
    match = re.search(r"```(?:lean4?|Lean)?\s*\n(.*?)```", raw, re.S)
    proof = (match.group(1) if match else raw).strip()
    if not proof.startswith("by") or not re.match(r"by(?:\s|$)", proof):
        raise ValueError("Expected only a tactic proof starting with by")
    if FORBIDDEN.search(proof) or '"' in proof:
        raise ValueError("Rejected unfinished proof, declaration, metaprogramming, or unsupported syntax")
    return proof


def statement_prefix(statement):
    prefix, separator, tail = statement.rpartition(":=")
    if not separator or tail.strip() not in {"sorry", "by sorry"}:
        raise ValueError("Expected dataset statement with a final sorry placeholder")
    prefix = re.sub(r"^(theorem|lemma)\s+\S+", "theorem candidate_proof", prefix.strip(), count=1)
    if not prefix.startswith("theorem candidate_proof ") and not prefix.startswith("theorem candidate_proof\n") and not prefix.startswith("theorem candidate_proof:"):
        raise ValueError("Unexpected theorem declaration")
    return prefix + " := "


def compile_source(source, tag, timeout=180, inspect_axioms=True):
    checks = scratch_root() / "lean/checks"
    checks.mkdir(parents=True, exist_ok=True)
    path = checks / f"{tag}.lean"
    path.write_text(source)
    command = ["bash", str(project_root() / "scripts/lean/lean.sh"), "lake", "env", "lean", "-M4096", str(path)]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return {"status": "timeout", "verified": False, "seconds": time.monotonic() - started, "output": output[-12000:]}
    result = {"status": "lean_error" if process.returncode else "compiled", "verified": False, "returncode": process.returncode, "seconds": time.monotonic() - started, "output": output[-12000:]}
    if process.returncode or not inspect_axioms:
        return result
    if "sorry" in output.lower():
        result["status"] = "rejected_sorry"
        return result
    match = re.search(r"'candidate_proof' depends on axioms:\s*\[([^\]]*)\]", output)
    if match:
        axioms = {item.strip() for item in match.group(1).split(",") if item.strip()}
    elif "'candidate_proof' does not depend on any axioms" in output:
        axioms = set()
    else:
        result["status"] = "unrecognized_axiom_audit"
        return result
    result.update(axioms=sorted(axioms), verified=axioms <= ALLOWED_AXIOMS, status="verified" if axioms <= ALLOWED_AXIOMS else "rejected_axioms")
    return result


def prepare(root):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    target = data_dir("lean_smoke_v1/pool.json")
    if target.exists():
        print("Reusing frozen validation statement pool")
        return
    revision = HfApi().dataset_info("cat-searcher/minif2f-lean4").sha
    dataset = load_dataset("cat-searcher/minif2f-lean4", split="validation", revision=revision)
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    rows, unsupported = [], []
    for index in indices:
        row = dataset[index]
        try:
            prefix = statement_prefix(row["formal_statement"])
        except ValueError as error:
            unsupported.append({"id": row["id"], "reason": str(error), "original_statement": row["formal_statement"]})
            continue
        rows.append({"source_index": index, "id": row["id"], "statement": prefix, "original_statement": row["formal_statement"]})
    write_json(target, {"source": "cat-searcher/minif2f-lean4", "revision": revision, "split": "validation", "seed": 42, "lean_version": "4.24.0", "mathlib_version": "v4.24.0", "header": HEADER, "rows": rows, "unsupported_source_rows": unsupported, "selection": "Select first20 typechecking statements in seed42-shuffled validation order; compatibility screening only, no model-success selection", "limitations": "Restricted tactic-only zero-shot feasibility test, not a full miniF2F benchmark or certified sandbox"})
    print(f"Prepared {len(rows)} validation statements for host-side compatibility preflight")


def run(root, shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import enable_full_determinism
    from src.evaluation.pilot_exploration import stop_ids

    frozen = json.loads(data_dir("lean_smoke_v1/frozen.json").read_text())
    if not json.loads((root / "checker_preflight.json").read_text())["passed"]:
        raise RuntimeError("Lean checker preflight must pass before inference")
    rows = frozen["rows"][shard * 5:(shard + 1) * 5]
    path = root / f"shard_{shard}.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if [r["id"] for r in existing] != [r["id"] for r in rows[:len(existing)]] or len(existing) > 5:
        raise ValueError("Resume source IDs differ")
    if len(existing) == 5:
        return
    enable_full_determinism(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir("Qwen2.5-Math-1.5B"), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir("Qwen2.5-Math-1.5B"), local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval()
    eos = stop_ids(tokenizer, model.generation_config)
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as stream:
        for row in rows[len(existing):]:
            if time.monotonic() - started > 11000:
                print("Safe pause between problems; rerun shard")
                return
            messages = [{"role": "system", "content": "You are a Lean 4 theorem prover. Return only a complete tactic proof starting with by. Do not use sorry, admit, new declarations, metaprogramming, or native_decide."}, {"role": "user", "content": "Complete this Lean 4 theorem. Mathlib is available. Output only the proof, not the theorem statement.\n\n" + HEADER + row["statement"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            set_seed(42000 + row["source_index"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=8, max_new_tokens=2048, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            attempts = []
            for index, sequence in enumerate(generated):
                tokens = sequence[inputs["input_ids"].shape[1]:].tolist()
                ending = next((i for i, value in enumerate(tokens) if value in eos), None)
                tokens = tokens[:ending + 1] if ending is not None else tokens
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                attempts.append({"index": index, "raw_output": raw, "generated_tokens": len(tokens), "truncated": ending is None and len(tokens) >= 2048})
            stream.write(json.dumps({"id": row["id"], "source_index": row["source_index"], "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "attempts": attempts}) + "\n")
            stream.flush()
            print(f"Generated eight proofs for {row['id']}; host-side Lean checking follows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    root = ensure_scratch_path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(root)
    elif args.shard is not None:
        run(root, args.shard)
    else:
        parser.error("Choose --prepare or --shard")


if __name__ == "__main__":
    main()
