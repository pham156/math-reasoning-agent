from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src.config import data_dir, log_dir, model_dir
from src.evaluation_answer import answer_record
from src.evaluation_metrics import branch_stats, retrieval_stats, summarize_results, verifier_stats


MODES = ("baseline", "retrieval", "verifier", "search")


def load_math_subset(dataset_dir: str, split: str, limit: int, category: str | None = None, level: str | None = None) -> list[dict]:
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_dir)
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found. Available splits: {list(dataset.keys())}")

    rows = []
    for i, row in enumerate(dataset[split]):
        row_category = row.get("type", row.get("category", ""))
        row_level = row.get("level", "")
        if category and row_category != category:
            continue
        if level and row_level != level:
            continue
        rows.append(
            {
                "id": f"{split}-{i}",
                "problem": row.get("problem", ""),
                "solution": row.get("solution", ""),
                "type": row_category,
                "level": row_level,
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def solve_baseline(problem: str, model_path: str, max_new_tokens: int, temperature: float) -> dict:
    from src.infer_baseline import generate_solution

    solution = generate_solution(problem, model_path=model_path, max_new_tokens=max_new_tokens, temperature=temperature)
    return {"solution": solution, "retrieved_examples": [], "verification": None, "proof_tree": None, "trace": None}


def solve_retrieval(
    problem: str,
    model_path: str,
    top_k: int,
    candidate_k: int,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    from src.solver_agent import solve_with_retrieval

    result = solve_with_retrieval(
        problem=problem,
        model_path=model_path,
        top_k=top_k,
        candidate_k=candidate_k,
        num_candidates=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return {
        "solution": result["candidates"][0],
        "retrieved_examples": result["examples"],
        "verification": None,
        "proof_tree": None,
        "trace": None,
    }


def solve_verifier(
    problem: str,
    model_path: str,
    top_k: int,
    candidate_k: int,
    max_iterations: int,
    max_new_tokens: int,
    temperature: float,
    expected_answer: str | None,
) -> dict:
    from src.self_critique_loop import run_solver_verifier_loop

    trace = run_solver_verifier_loop(
        problem=problem,
        solver_model_path=model_path,
        top_k=top_k,
        candidate_k=candidate_k,
        max_iterations=max_iterations,
        max_new_tokens=max_new_tokens,
        solver_temperature=temperature,
        expected_answer=expected_answer,
    )
    final_verification = trace.final_verification.to_dict() if trace.final_verification else None
    return {
        "solution": trace.final_solution,
        "retrieved_examples": trace.retrieved_examples,
        "verification": final_verification,
        "proof_tree": None,
        "trace": trace.to_dict(),
    }


def solve_search(
    problem: str,
    model_path: str,
    top_k: int,
    candidate_k: int,
    max_depth: int,
    beam_width: int,
    branch_factor: int,
    max_new_tokens: int,
    use_retrieval: bool,
    use_verifier: bool,
    expected_answer: str | None,
) -> dict:
    from src.proof_search import ProofSearchConfig, build_retrieval_context_for_search, run_proof_search
    from src.proof_tree import best_nodes

    retrieval_context = ""
    examples = []
    if use_retrieval:
        retrieval_context, examples = build_retrieval_context_for_search(problem, top_k=top_k, candidate_k=candidate_k)
    config = ProofSearchConfig(
        model_path=model_path,
        max_depth=max_depth,
        beam_width=beam_width,
        branch_factor=branch_factor,
        num_strategies=max(beam_width, 3),
        max_new_tokens=max_new_tokens,
        use_verifier=use_verifier,
        expected_answer=expected_answer,
    )
    root = run_proof_search(problem, config, retrieval_context=retrieval_context)
    best = best_nodes(root, limit=1)
    solution = best[0].partial_solution if best else ""
    verification = None
    if best and best[0].verifier_score is not None:
        verification = {
            "verdict": "complete" if best[0].status == "complete" else best[0].status,
            "score": best[0].verifier_score,
            "confidence": None,
            "issues": [],
            "critique": best[0].critique,
        }
    return {
        "solution": solution,
        "retrieved_examples": examples,
        "verification": verification,
        "proof_tree": root,
        "trace": None,
    }


def evaluate_example(example: dict, mode: str, args: argparse.Namespace) -> dict:
    start = time.time()
    if mode == "baseline":
        solved = solve_baseline(example["problem"], args.model_path, args.max_new_tokens, args.temperature)
    elif mode == "retrieval":
        solved = solve_retrieval(
            example["problem"], args.model_path, args.top_k, args.candidate_k, args.max_new_tokens, args.temperature
        )
    elif mode == "verifier":
        solved = solve_verifier(
            example["problem"],
            args.model_path,
            args.top_k,
            args.candidate_k,
            args.max_iterations,
            args.max_new_tokens,
            args.temperature,
            expected_answer=None if args.no_reference_to_verifier else example["solution"],
        )
    elif mode == "search":
        solved = solve_search(
            example["problem"],
            args.model_path,
            args.top_k,
            args.candidate_k,
            args.max_depth,
            args.beam_width,
            args.branch_factor,
            args.max_new_tokens,
            args.search_use_retrieval,
            args.search_use_verifier,
            expected_answer=None if args.no_reference_to_verifier else example["solution"],
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if args.verify_all_outputs and solved.get("verification") is None:
        from src.verifier_backends import PromptVerifierBackend

        verifier = PromptVerifierBackend(model_path=args.model_path)
        expected_answer = None if args.no_reference_to_verifier else example["solution"]
        solved["verification"] = verifier.verify(example["problem"], solved["solution"], expected_answer=expected_answer).to_dict()

    answer = answer_record(solved["solution"], example["solution"])
    runtime = time.time() - start
    return {
        "id": example["id"],
        "mode": mode,
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "problem": example["problem"],
        "reference_solution": example["solution"],
        "prediction": solved["solution"],
        **answer,
        "runtime_sec": runtime,
        "retrieval": retrieval_stats(solved.get("retrieved_examples", [])),
        "verifier": verifier_stats(solved.get("verification")),
        "branch": branch_stats(solved.get("proof_tree")),
        "trace": solved.get("trace"),
        "proof_tree": solved["proof_tree"].to_dict() if solved.get("proof_tree") is not None else None,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id",
        "mode",
        "type",
        "level",
        "exact_match",
        "predicted_answer",
        "reference_answer",
        "runtime_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def append_log(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate math reasoning agent modes on MATH subsets.")
    parser.add_argument("--dataset_dir", default=str(data_dir("MATH")))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--category", default=None)
    parser.add_argument("--level", default=None)
    parser.add_argument("--modes", default="baseline,retrieval,verifier,search")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--branch_factor", type=int, default=2)
    parser.add_argument("--search_use_retrieval", action="store_true")
    parser.add_argument("--search_use_verifier", action="store_true")
    parser.add_argument("--verify_all_outputs", action="store_true")
    parser.add_argument("--no_reference_to_verifier", action="store_true")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Available modes: {MODES}")

    experiment_name = args.experiment_name or time.strftime("math_eval_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else log_dir("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"

    examples = load_math_subset(args.dataset_dir, args.split, args.limit, args.category, args.level)
    config = vars(args).copy()
    config["modes"] = modes
    config["experiment_name"] = experiment_name
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(run_log, f"Started experiment={experiment_name} examples={len(examples)} modes={','.join(modes)}")

    rows = []
    for example in examples:
        for mode in modes:
            progress = f"[{len(rows) + 1}] mode={mode} id={example['id']} type={example.get('type')} level={example.get('level')}"
            print(progress)
            append_log(run_log, progress)
            row = evaluate_example(example, mode, args)
            rows.append(row)
            write_jsonl(output_dir / "results.jsonl", rows)
            append_log(
                run_log,
                f"finished mode={mode} id={example['id']} exact={row['exact_match']} runtime={row['runtime_sec']:.2f}s",
            )

    summary = summarize_results(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "results.csv", rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote experiment outputs to {output_dir}")
    append_log(run_log, "Finished experiment")


if __name__ == "__main__":
    main()
