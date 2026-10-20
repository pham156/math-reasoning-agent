"""Host-side orchestration of isolated Lean processes; stdlib-only dependencies."""

import argparse
from collections import Counter
import json
from src.core.config import data_dir, ensure_scratch_path
from src.training.grpo.train import write_json
from src.evaluation.lean_smoke import HEADER, compile_source, proof_body


def preflight(root):
    positive = compile_source(HEADER + "theorem candidate_proof (n : Nat) : n = n := by rfl\n#print axioms candidate_proof\n", "positive_control", timeout=180)
    if not positive["verified"]:
        write_json(root / "checker_preflight.json", {"passed": False, "positive": positive})
        raise RuntimeError("Positive checker control failed; do not launch generation")
    negative = compile_source(HEADER + "theorem candidate_proof : False := by sorry\n#print axioms candidate_proof\n", "negative_control", timeout=180)
    wrong = compile_source(HEADER + "theorem candidate_proof : (0 : Nat) = 1 := by norm_num\n#print axioms candidate_proof\n", "wrong_control", timeout=180)
    report = {"passed": False, "positive": positive, "negative": negative, "wrong": wrong}
    write_json(root / "checker_preflight.json", report)
    if not positive["verified"] or negative["verified"] or wrong["verified"] or wrong["status"] != "lean_error":
        raise RuntimeError("Checker controls failed; inspect checker_preflight.json")
    pool = json.loads(data_dir("lean_smoke_v1/pool.json").read_text())
    frozen_path = data_dir("lean_smoke_v1/frozen.json")
    if frozen_path.exists():
        report["passed"] = True
        write_json(root / "checker_preflight.json", report)
        return
    rows, rejected = [], []
    for row in pool["rows"]:
        check = compile_source(HEADER + row["statement"] + "by sorry\n", f"preflight_{row['source_index']}", timeout=90, inspect_axioms=False)
        if check.get("returncode") != 0:
            rejected.append({"id": row["id"], "check": check})
            continue
        rows.append(row)
        print(f"Preflight: {len(rows)}/20 compatible statements", flush=True)
        if len(rows) == 20:
            break
    if len(rows) != 20:
        raise RuntimeError("Not enough compatible statements; do not launch generation")
    pool.update(rows=rows, compatibility_rejections=rejected)
    write_json(frozen_path, pool)
    report.update(passed=True, selected_ids=[r["id"] for r in rows], compatibility_rejected=len(rejected))
    write_json(root / "checker_preflight.json", report)


def check_shard(root, shard):
    frozen = json.loads(data_dir("lean_smoke_v1/frozen.json").read_text())
    source_rows = {r["id"]: r for r in frozen["rows"]}
    raw = [json.loads(line) for line in (root / f"shard_{shard}.jsonl").read_text().splitlines() if line.strip()]
    expected = [r["id"] for r in frozen["rows"][shard * 5:(shard + 1) * 5]]
    if [r["id"] for r in raw] != expected or any(len(r["attempts"]) != 8 for r in raw):
        raise ValueError("Generation shard incomplete; resume before checking")
    checked = []
    for row in raw:
        for attempt in row["attempts"]:
            try:
                proof = proof_body(attempt["raw_output"])
                source = HEADER + source_rows[row["id"]]["statement"] + proof + "\n#print axioms candidate_proof\n"
                result = compile_source(source, f"model_s{shard}_p{row['source_index']}_a{attempt['index']}")
            except ValueError as error:
                proof, result = None, {"status": "rejected_output", "verified": False, "reason": str(error)}
            attempt.update(proof=proof, check=result)
            print(f"{row['id']} attempt{attempt['index']}: {result['status']}", flush=True)
        row["verified_count"] = sum(a["check"]["verified"] for a in row["attempts"])
        checked.append(row)
        write_json(root / f"shard_{shard}_checked.json", {"rows": checked})


def summarize(root):
    rows = [r for shard in range(4) for r in json.loads((root / f"shard_{shard}_checked.json").read_text())["rows"]]
    if len(rows) != 20 or len({r["id"] for r in rows}) != 20:
        raise ValueError("Need all20 distinct checked problems")
    attempts = [a for row in rows for a in row["attempts"]]
    status = dict(Counter(a["check"]["status"] for a in attempts))
    result = {"problems": 20, "attempts": len(attempts), "verified_problems": sum(r["verified_count"] > 0 for r in rows), "empirical_pass_at_8": sum(r["verified_count"] > 0 for r in rows) / 20, "verified_attempts": sum(a["check"]["verified"] for a in attempts), "statuses": status, "truncated_attempts": sum(a["truncated"] for a in attempts), "mean_tokens": sum(a["generated_tokens"] for a in attempts) / len(attempts), "checker_seconds": sum(a["check"].get("seconds", 0) for a in attempts), "limitations": "Zero-shot base1.5B,20 compatibility-screened miniF2F validation problems, restricted proof syntax,8 samples/problem. No training. Kernel checks formal statement under standard axioms; no claim of full benchmark accuracy or verified natural-language translation."}
    write_json(root / "summary.json", result)
    (root / "summary.md").write_text("# Lean feasibility run\n\n" + json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    root = ensure_scratch_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight(root)
    elif args.summarize:
        summarize(root)
    elif args.shard is not None:
        check_shard(root, args.shard)
    else:
        parser.error("Select an action")


if __name__ == "__main__":
    main()
