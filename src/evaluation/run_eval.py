from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agents.pipeline import (
    ensure_math_index,
    missing_model_error,
    normalize_retrieved_examples,
    resolve_model_path,
    resolve_output_path,
)
from src.core.config import model_dir, print_storage_summary
from src.core.infer_baseline import (
    adapter_files_present,
    build_generation_inputs,
    build_prompt,
    load_causal_lm,
    model_files_present,
    model_input_device,
    runtime_model_metadata,
)
from src.evaluation.answer import extract_final_answer, exact_match, normalize_answer
from src.evaluation.benchmarks import load_benchmark_examples
from src.retrieval.context import build_compact_solver_prompt, build_solver_prompt
from src.retrieval.retrieve import MathRetriever
from src.verifier.prompts import build_verifier_prompt
from src.verifier.schema import extract_json_object, fallback_verification, normalize_verification


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower().strip()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


class LoadedGenerator:
    def __init__(self, model_path: str | Path, adapter_path: str | Path | None = None) -> None:
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.tokenizer, self.model = load_causal_lm(self.model_path, adapter_path=adapter_path)
        self.device = model_input_device(self.model)
        self.model_metadata = runtime_model_metadata(
            self.model_path,
            tokenizer=self.tokenizer,
            model=self.model,
            adapter_path=adapter_path,
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, dict]:
        import torch

        inputs = build_generation_inputs(self.tokenizer, prompt, self.device)
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        start = time.time()
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": int(generated.shape[-1]),
            "generation_runtime_sec": time.time() - start,
        }


def retrieved_metadata(examples: list[dict]) -> list[dict]:
    keys = (
        "id",
        "type",
        "level",
        "problem",
        "solution",
        "similarity_score",
        "semantic_score",
        "score",
        "rerank_score",
        "faiss_rank",
        "topics",
        "strategies",
        "theorem_patterns",
    )
    return [{key: example.get(key) for key in keys if key in example} for example in examples]


def build_prompt_for_example(
    problem: str,
    retrieval_source: str,
    retriever: MathRetriever | None,
    top_k: int,
    candidate_k: int,
    retrieval_prompt_style: str = "full",
) -> tuple[str, list[dict], dict, float]:
    if retrieval_source == "none":
        return (
            build_prompt(problem),
            [],
            {"requested_source": "none", "actual_source": "none", "retrieval_prompt_style": "none"},
            0.0,
        )

    start = time.time()
    if retriever is None:
        raise RuntimeError("retrieval_source=math requires an initialized MathRetriever.")
    raw_examples = retriever.retrieve(problem, top_k=top_k, candidate_k=candidate_k)
    retrieval_sec = time.time() - start
    examples = normalize_retrieved_examples(raw_examples)
    if retrieval_prompt_style == "compact":
        prompt = build_compact_solver_prompt(problem, examples, max_examples=top_k)
    elif retrieval_prompt_style == "full":
        prompt = build_solver_prompt(problem, examples, max_examples=top_k)
    else:
        raise ValueError(f"Unknown retrieval_prompt_style {retrieval_prompt_style!r}")
    return (
        prompt,
        examples,
        {
            "requested_source": "math",
            "actual_source": "math",
            "num_retrieved": len(examples),
            "retrieval_prompt_style": retrieval_prompt_style,
        },
        retrieval_sec,
    )


def resolve_adapter_path(adapter_path: str | None) -> Path | None:
    if not adapter_path:
        return None
    requested = Path(adapter_path).expanduser()
    if requested.is_absolute():
        return requested
    parts = requested.parts
    if parts and parts[0] == "outputs":
        return resolve_output_path(requested)
    return requested


def verify_solution(generator: LoadedGenerator, problem: str, solution: str, max_new_tokens: int) -> tuple[dict, dict]:
    prompt = build_verifier_prompt(problem, solution, expected_answer=None)
    raw, token_meta = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    try:
        parsed = extract_json_object(raw)
        verification = normalize_verification(parsed, raw=raw)
    except Exception as exc:
        verification = fallback_verification(raw, f"Verifier did not return parseable JSON: {exc}")
    return verification.to_dict(), token_meta


def candidate_record(
    index: int,
    raw_solution: str,
    generation_meta: dict,
    problem: str,
    exact_reference: str,
    generator: LoadedGenerator,
    selection: str,
    use_verifier: bool,
    verifier_max_new_tokens: int,
) -> tuple[dict, float]:
    verifier_result = None
    verifier_token_meta = {}
    verifier_sec = 0.0
    if use_verifier or selection in {"verifier", "consensus_verifier"}:
        verifier_start = time.time()
        verifier_result, verifier_token_meta = verify_solution(
            generator,
            problem,
            raw_solution,
            max_new_tokens=verifier_max_new_tokens,
        )
        verifier_sec = time.time() - verifier_start

    answer = extract_final_answer(raw_solution)
    return (
        {
            "candidate_index": index,
            "raw_solution": raw_solution,
            "extracted_answer": answer,
            "normalized_answer": normalize_answer(answer),
            "exact_match": exact_match(raw_solution, exact_reference),
            "verifier_verdict": verifier_result.get("verdict") if verifier_result else None,
            "verifier_score": verifier_result.get("score") if verifier_result else None,
            "verifier_critique": verifier_result,
            "token_counts": {
                "solver_input_tokens": generation_meta.get("input_tokens"),
                "solver_generated_tokens": generation_meta.get("generated_tokens"),
                "verifier_input_tokens": verifier_token_meta.get("input_tokens"),
                "verifier_generated_tokens": verifier_token_meta.get("generated_tokens"),
            },
            "runtime": {
                "solver_generation_sec": generation_meta.get("generation_runtime_sec", 0.0),
                "verifier_sec": verifier_sec,
            },
        },
        verifier_sec,
    )


def select_candidate(candidates: list[dict], selection: str) -> tuple[dict, dict]:
    if selection == "first" or len(candidates) == 1:
        selected = candidates[0]
        return selected, {
            "selection": "first",
            "selected_candidate_index": selected["candidate_index"],
            "reason": "first_candidate",
        }
    if selection == "verifier":
        selected = max(
            candidates,
            key=lambda candidate: (
                float(candidate.get("verifier_score") or 0.0),
                -int(candidate["candidate_index"]),
            ),
        )
        scores = [float(candidate.get("verifier_score") or 0.0) for candidate in candidates]
        selected_correct = bool(selected.get("exact_match"))
        any_correct = any(bool(candidate.get("exact_match")) for candidate in candidates)
        first_correct = bool(candidates[0].get("exact_match"))
        return selected, {
            "selection": "verifier",
            "selected_candidate_index": selected["candidate_index"],
            "selected_verifier_score": selected.get("verifier_score"),
            "candidate_verifier_scores": scores,
            "any_candidate_correct": any_correct,
            "first_candidate_correct": first_correct,
            "selected_candidate_correct": selected_correct,
            "verifier_selected_correct_over_incorrect": selected_correct
            and any(not bool(candidate.get("exact_match")) for candidate in candidates),
            "verifier_fixed_first_candidate": selected_correct and not first_correct,
            "verifier_missed_available_correct": any_correct and not selected_correct,
        }
    if selection == "consensus_verifier":
        groups: dict[str, list[dict]] = {}
        for candidate in candidates:
            key = candidate.get("normalized_answer") or ""
            groups.setdefault(key, []).append(candidate)

        answer_groups = [
            {
                "normalized_answer": answer,
                "display_answer": group[0].get("extracted_answer"),
                "count": len(group),
                "candidate_indices": [candidate["candidate_index"] for candidate in group],
                "max_verifier_score": max(float(candidate.get("verifier_score") or 0.0) for candidate in group),
                "any_candidate_correct": any(bool(candidate.get("exact_match")) for candidate in group),
            }
            for answer, group in groups.items()
        ]
        answer_groups.sort(
            key=lambda group: (
                group["count"],
                group["max_verifier_score"],
                -min(group["candidate_indices"]),
            ),
            reverse=True,
        )
        majority_threshold = len(candidates) // 2 + 1
        majority_groups = [group for group in answer_groups if group["count"] >= majority_threshold]
        if majority_groups:
            selected_group = majority_groups[0]
            fallback_to_verifier = False
            reason = "majority_answer_group"
            selected_pool = groups[selected_group["normalized_answer"]]
        else:
            verifier_selected, verifier_metadata = select_candidate(candidates, "verifier")
            selected_group = next(
                group
                for group in answer_groups
                if verifier_selected["candidate_index"] in group["candidate_indices"]
            )
            fallback_to_verifier = True
            reason = "no_majority_fallback_to_verifier"
            selected_pool = [verifier_selected]

        selected = max(
            selected_pool,
            key=lambda candidate: (
                float(candidate.get("verifier_score") or 0.0),
                -int(candidate["candidate_index"]),
            ),
        )
        selected_correct = bool(selected.get("exact_match"))
        any_correct = any(bool(candidate.get("exact_match")) for candidate in candidates)
        first_correct = bool(candidates[0].get("exact_match"))
        metadata = {
            "selection": "consensus_verifier",
            "selected_candidate_index": selected["candidate_index"],
            "selected_verifier_score": selected.get("verifier_score"),
            "candidate_verifier_scores": [float(candidate.get("verifier_score") or 0.0) for candidate in candidates],
            "answer_groups": answer_groups,
            "selected_group": selected_group,
            "fallback_to_verifier": fallback_to_verifier,
            "fallback_reason": reason,
            "majority_threshold": majority_threshold,
            "num_answer_groups": len(answer_groups),
            "any_candidate_correct": any_correct,
            "first_candidate_correct": first_correct,
            "selected_candidate_correct": selected_correct,
            "oracle_candidate_available": any_correct,
            "verifier_selected_correct_over_incorrect": selected_correct
            and any(not bool(candidate.get("exact_match")) for candidate in candidates),
            "verifier_fixed_first_candidate": selected_correct and not first_correct,
            "verifier_missed_available_correct": any_correct and not selected_correct,
        }
        if fallback_to_verifier:
            metadata["verifier_fallback_metadata"] = verifier_metadata
        return selected, metadata
    raise ValueError(f"Unknown selection mode {selection!r}")


def evaluate_one(
    example: dict,
    generator: LoadedGenerator,
    retrieval_source: str,
    retriever: MathRetriever | None,
    top_k: int,
    candidate_k: int,
    retrieval_prompt_style: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_verifier: bool,
    verifier_max_new_tokens: int,
    num_candidates: int,
    selection: str,
    benchmark_metadata: dict,
) -> dict:
    total_start = time.time()
    mode = "qwen_with_math_retrieval" if retrieval_source == "math" else "qwen_without_retrieval"
    if use_verifier:
        mode += "_and_verifier"
    prompt, examples, retrieval_meta, retrieval_sec = build_prompt_for_example(
        example["problem"],
        retrieval_source,
        retriever,
        top_k=top_k,
        candidate_k=candidate_k,
        retrieval_prompt_style=retrieval_prompt_style,
    )

    ground_truth_answer = example.get("ground_truth_answer_override") or extract_final_answer(
        example["ground_truth_solution"]
    )
    exact_reference = ground_truth_answer or example["ground_truth_solution"]
    candidates = []
    verifier_sec = 0.0
    for candidate_idx in range(num_candidates):
        raw_solution, generation_meta = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        candidate, candidate_verifier_sec = candidate_record(
            index=candidate_idx,
            raw_solution=raw_solution,
            generation_meta=generation_meta,
            problem=example["problem"],
            exact_reference=exact_reference,
            generator=generator,
            selection=selection,
            use_verifier=use_verifier,
            verifier_max_new_tokens=verifier_max_new_tokens,
        )
        candidates.append(candidate)
        verifier_sec += candidate_verifier_sec

    selected_candidate, selection_metadata = select_candidate(candidates, selection)
    raw_solution = selected_candidate["raw_solution"]
    model_answer = selected_candidate["extracted_answer"]
    verifier_result = selected_candidate.get("verifier_critique")
    solver_input_tokens = sum(
        int(candidate.get("token_counts", {}).get("solver_input_tokens") or 0) for candidate in candidates
    )
    solver_generated_tokens = sum(
        int(candidate.get("token_counts", {}).get("solver_generated_tokens") or 0) for candidate in candidates
    )
    verifier_input_tokens = sum(
        int(candidate.get("token_counts", {}).get("verifier_input_tokens") or 0) for candidate in candidates
    )
    verifier_generated_tokens = sum(
        int(candidate.get("token_counts", {}).get("verifier_generated_tokens") or 0) for candidate in candidates
    )
    solver_generation_sec = sum(
        float(candidate.get("runtime", {}).get("solver_generation_sec") or 0.0) for candidate in candidates
    )
    return {
        "benchmark": example.get("benchmark", benchmark_metadata.get("benchmark")),
        "benchmark_metadata": benchmark_metadata,
        "mode": mode,
        "problem_id": example["problem_id"],
        "problem_index": example["problem_index"],
        "source_id": example.get("source_id", example["problem_id"]),
        "type": example.get("type", ""),
        "category": example.get("category", example.get("type", "")),
        "level": example.get("level", ""),
        "problem": example["problem"],
        "ground_truth_solution": example["ground_truth_solution"],
        "ground_truth_answer": ground_truth_answer,
        "extracted_ground_truth_answer": ground_truth_answer,
        "model_raw_solution": raw_solution,
        "model_answer": model_answer,
        "extracted_model_answer": model_answer,
        "final_extracted_answer": model_answer,
        "normalized_ground_truth_answer": normalize_answer(ground_truth_answer),
        "normalized_model_answer": normalize_answer(model_answer),
        "exact_match": exact_match(raw_solution, exact_reference),
        "exact_match_metadata": {
            "metric": "exact_match_after_answer_extraction",
            "normalization": (
                "boxed/fbox, gsm8k ####, simple final-answer, numeric comma removal, "
                "whitespace/latex-text/list normalization"
            ),
        },
        "retrieved_examples_metadata": retrieved_metadata(examples),
        "retrieval_metadata": retrieval_meta,
        "candidate_solutions": candidates,
        "selected_candidate": selected_candidate,
        "selection_metadata": selection_metadata,
        "verifier_verdict": verifier_result.get("verdict") if verifier_result else None,
        "verifier_score": verifier_result.get("score") if verifier_result else None,
        "verifier_critique": verifier_result,
        "token_counts": {
            "solver_input_tokens": solver_input_tokens,
            "solver_generated_tokens": solver_generated_tokens,
            "selected_solver_input_tokens": selected_candidate.get("token_counts", {}).get("solver_input_tokens"),
            "selected_solver_generated_tokens": selected_candidate.get("token_counts", {}).get("solver_generated_tokens"),
            "verifier_input_tokens": verifier_input_tokens or None,
            "verifier_generated_tokens": verifier_generated_tokens or None,
        },
        "runtime": {
            "retrieval_sec": retrieval_sec,
            "solver_generation_sec": solver_generation_sec,
            "verifier_sec": verifier_sec,
            "total_sec": time.time() - total_start,
        },
        "runtime_sec": time.time() - total_start,
        "benchmark_example_metadata": example.get("benchmark_example_metadata", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark-style math evaluation for real Qwen modes.")
    parser.add_argument(
        "--benchmark",
        choices=["math", "math500", "gsm8k100", "aime2024", "aime2025", "aime"],
        default="math",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Number of examples. Defaults: math=20, math500=500, gsm8k100=100, "
            "aime2024/aime2025/aime=all available."
        ),
    )
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--sampling", choices=["first", "random"], default="first")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--aime_dataset_path", default=None)
    parser.add_argument("--aime_hf_dataset", default=None)
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--retrieval_source", choices=["none", "math"], default="none")
    parser.add_argument("--use_verifier", type=str_to_bool, default=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--retrieval_prompt_style", choices=["full", "compact"], default="full")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--verifier_max_new_tokens", type=int, default=512)
    parser.add_argument("--num_candidates", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument("--selection", choices=["first", "verifier", "consensus_verifier"], default="first")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print_storage_summary()

    effective_model_path, model_metadata = resolve_model_path(args.model_path)
    if not model_files_present(effective_model_path):
        raise missing_model_error(model_metadata)
    effective_adapter_path = resolve_adapter_path(args.adapter_path)
    if effective_adapter_path is not None and not adapter_files_present(effective_adapter_path):
        raise RuntimeError(
            "Adapter evaluation requested, but LoRA adapter files are missing or incomplete.\n"
            f"Adapter path: {effective_adapter_path}\n"
            "Expected adapter_config.json and adapter weights."
        )

    retriever = None
    run_retrieval_metadata = {"requested_source": args.retrieval_source}
    if args.retrieval_source == "math":
        run_retrieval_metadata.update(ensure_math_index(auto_build=True))
        retriever = MathRetriever(
            index_path=run_retrieval_metadata["index_path"],
            metadata_path=run_retrieval_metadata["metadata_path"],
        )

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples, benchmark_metadata = load_benchmark_examples(
        benchmark=args.benchmark,
        split=args.split,
        limit=args.limit,
        sample_seed=args.sample_seed,
        sampling=args.sampling,
        aime_dataset_path=args.aime_dataset_path,
        aime_hf_dataset=args.aime_hf_dataset,
        start_index=args.start_index,
    )
    generator = LoadedGenerator(effective_model_path, adapter_path=effective_adapter_path)

    run_header = {
        "kind": "run_metadata",
        "schema_version": "benchmark_eval_v1",
        "benchmark_metadata": benchmark_metadata,
        "benchmark": args.benchmark,
        "split": args.split,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "sampling": args.sampling,
        "start_index": args.start_index,
        "retrieval_source": args.retrieval_source,
        "use_verifier": args.use_verifier,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "retrieval_prompt_style": args.retrieval_prompt_style if args.retrieval_source == "math" else "none",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_candidates": args.num_candidates,
        "selection": args.selection,
        "model_metadata": {**model_metadata, **generator.model_metadata},
        "adapter_path": str(effective_adapter_path) if effective_adapter_path else None,
        "run_retrieval_metadata": run_retrieval_metadata,
    }

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(run_header, ensure_ascii=False) + "\n")
        for idx, example in enumerate(examples, start=1):
            if not args.quiet:
                print(f"[{idx}/{len(examples)}] {example['problem_id']} {example.get('type')} {example.get('level')}")
            row = evaluate_one(
                example,
                generator,
                retrieval_source=args.retrieval_source,
                retriever=retriever,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                retrieval_prompt_style=args.retrieval_prompt_style,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_verifier=args.use_verifier,
                verifier_max_new_tokens=args.verifier_max_new_tokens,
                num_candidates=args.num_candidates,
                selection=args.selection,
                benchmark_metadata=benchmark_metadata,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    if not args.quiet:
        print(f"Wrote evaluation JSONL to {output_path}")


if __name__ == "__main__":
    main()
