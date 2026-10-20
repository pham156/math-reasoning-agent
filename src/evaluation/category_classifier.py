from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agents.pipeline import missing_model_error, resolve_model_path, resolve_output_path
from src.core.config import model_dir, print_storage_summary
from src.core.infer_baseline import (
    load_causal_lm,
    model_files_present,
    model_input_device,
    move_inputs_to_device,
    runtime_model_metadata,
)
from src.evaluation.benchmarks import load_benchmark_examples


CATEGORIES = (
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Geometry",
    "Number Theory",
    "Counting & Probability",
    "Precalculus",
)


CATEGORY_DESCRIPTIONS = {
    "Algebra": "equations, expressions, factoring, linear/quadratic manipulation, basic functions.",
    "Intermediate Algebra": "polynomials, roots, functional equations, inequalities, advanced algebraic manipulation.",
    "Prealgebra": "arithmetic, ratios, percents, averages, units, simple word problems.",
    "Geometry": "triangles, circles, angles, areas, coordinates in geometric figures.",
    "Number Theory": "integers, divisibility, primes, modular arithmetic, bases, remainders.",
    "Counting & Probability": "counting arrangements, combinations, probability, expected value, combinatorial cases.",
    "Precalculus": "trigonometry, logarithms, exponentials, sequences/series, complex numbers, polar coordinates.",
}


FEW_SHOT_EXAMPLES = [
    (
        "How many positive integers less than 100 are divisible by 3 or 5?",
        "Counting & Probability",
        0.86,
    ),
    (
        "Find all real x such that x^2 - 5x + 6 = 0.",
        "Algebra",
        0.90,
    ),
    (
        "Let P(x) be a monic cubic polynomial with roots 1, 2, and 3. Find P(4).",
        "Intermediate Algebra",
        0.88,
    ),
    (
        "Triangle ABC has AB=13, AC=14, and BC=15. Find its area.",
        "Geometry",
        0.93,
    ),
    (
        "Find the remainder when 7^2024 is divided by 13.",
        "Number Theory",
        0.94,
    ),
    (
        "A shirt costs $20 after a 25 percent discount. What was the original price?",
        "Prealgebra",
        0.84,
    ),
    (
        "Simplify sin^2 x + cos^2 x and evaluate it at x=40 degrees.",
        "Precalculus",
        0.86,
    ),
]


CLASSIFIER_SYSTEM_PROMPT = "You are a MATH benchmark category classifier."


def classifier_user_prompt(problem: str) -> str:
    descriptions = "\n".join(f"- {name}: {description}" for name, description in CATEGORY_DESCRIPTIONS.items())
    examples = "\n".join(
        (
            f"Problem: {example}\n"
            f'JSON: {{"category": "{category}", "confidence": {confidence:.2f}, '
            f'"rationale": "short category cue"}}'
        )
        for example, category, confidence in FEW_SHOT_EXAMPLES
    )
    return (
        "Classify the math problem into exactly one MATH benchmark category. Do not solve it.\n\n"
        f"Categories:\n{descriptions}\n\n"
        f"Examples:\n{examples}\n\n"
        f"Problem: {problem}\n"
        "Category:"
    )


def labeled_examples_text() -> str:
    return "\n".join(f"- {category}: {example}" for example, category, _ in FEW_SHOT_EXAMPLES)


def binary_category_prompt(problem: str, category: str) -> str:
    descriptions = "\n".join(f"- {name}: {description}" for name, description in CATEGORY_DESCRIPTIONS.items())
    return (
        "Decide whether the candidate category is the best MATH benchmark category for the problem. "
        "Do not solve the problem. Answer only Yes or No.\n\n"
        f"Categories:\n{descriptions}\n\n"
        f"Labeled examples:\n{labeled_examples_text()}\n\n"
        f"Problem: {problem}\n\n"
        f"Candidate category: {category}\n"
        f"Candidate description: {CATEGORY_DESCRIPTIONS[category]}\n"
        "Is this the best category? Answer:"
    )


def build_classifier_inputs_from_prompt(tokenizer, user_prompt: str, device) -> dict:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(rendered, "items"):
            return move_inputs_to_device(rendered, device)
        return {"input_ids": rendered.to(device)}
    return move_inputs_to_device(tokenizer(CLASSIFIER_SYSTEM_PROMPT + "\n\n" + user_prompt, return_tensors="pt"), device)


def build_classifier_inputs(tokenizer, problem: str, device) -> dict:
    return build_classifier_inputs_from_prompt(tokenizer, classifier_user_prompt(problem), device)


def normalize_prediction(parsed: dict, raw: str) -> dict:
    category = str(parsed.get("category", "")).strip()
    if category not in CATEGORIES:
        lowered = category.lower()
        matches = [item for item in CATEGORIES if item.lower() == lowered]
        category = matches[0] if matches else "Algebra"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "predicted_category": category,
        "confidence": confidence,
        "rationale": str(parsed.get("rationale", "")).strip(),
        "raw_classifier_output": raw,
    }


class ClassifierGenerator:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.tokenizer, self.model = load_causal_lm(self.model_path)
        self.device = model_input_device(self.model)
        self.model_metadata = runtime_model_metadata(self.model_path, tokenizer=self.tokenizer, model=self.model)

    def continuation_score(self, prompt: str, continuation: str) -> tuple[float, int, int]:
        import torch
        import torch.nn.functional as F

        inputs = build_classifier_inputs_from_prompt(self.tokenizer, prompt, self.device)
        context_ids = inputs["input_ids"][0]
        context_len = int(context_ids.shape[-1])
        label_ids = self.tokenizer(" " + continuation, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(
            context_ids.device
        )

        with torch.no_grad():
            full_ids = torch.cat([context_ids, label_ids], dim=0).unsqueeze(0)
            attention_mask = torch.ones_like(full_ids, device=full_ids.device)
            outputs = self.model(input_ids=full_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, context_len - 1 : -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, label_ids.view(1, -1, 1)).squeeze()
        return float(token_log_probs.mean().item()), int(label_ids.shape[-1]), context_len

    def classify(self, problem: str, max_new_tokens: int) -> tuple[dict, dict]:
        import torch
        import torch.nn.functional as F

        start = time.time()
        category_rows = []
        for category in CATEGORIES:
            prompt = binary_category_prompt(problem, category)
            yes_score, yes_tokens, input_tokens = self.continuation_score(prompt, "Yes")
            no_score, no_tokens, _ = self.continuation_score(prompt, "No")
            margin = yes_score - no_score
            category_rows.append(
                {
                    "category": category,
                    "score": margin,
                    "yes_score": yes_score,
                    "no_score": no_score,
                    "yes_probability": float(torch.sigmoid(torch.tensor(margin)).item()),
                    "yes_tokens": yes_tokens,
                    "no_tokens": no_tokens,
                    "input_tokens": input_tokens,
                }
            )

        score_tensor = torch.tensor([row["score"] for row in category_rows])
        rank_probabilities = F.softmax(score_tensor, dim=0).tolist()
        ranked = sorted(
            (
                {
                    **row,
                    "rank_probability": float(rank_probabilities[idx]),
                }
                for idx, row in enumerate(category_rows)
            ),
            key=lambda row: row["score"],
            reverse=True,
        )
        best = ranked[0]
        raw = json.dumps(
            {
                "method": "binary_yes_no_likelihood",
                "ranked_categories": ranked,
                "max_new_tokens_argument_ignored": max_new_tokens,
            },
            ensure_ascii=False,
        )
        prediction = normalize_prediction(
            {
                "category": best["category"],
                "confidence": best["yes_probability"],
                "rationale": (
                    "highest yes-minus-no likelihood margin from few-shot binary category prompt; "
                    f"runner_up={ranked[1]['category']}:{ranked[1]['yes_probability']:.3f}"
                )
            },
            raw,
        )
        return prediction, {
            "classifier_input_tokens": max(row["input_tokens"] for row in category_rows),
            "classifier_generated_tokens": 0,
            "classifier_runtime_sec": time.time() - start,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify MATH-500 categories with a few-shot LLM classifier.")
    parser.add_argument("--benchmark", default="math500", choices=["math500"])
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--output", default="outputs/retrieval_gating/category_predictions_llm.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print_storage_summary()

    effective_model_path, model_metadata = resolve_model_path(args.model_path)
    if not model_files_present(effective_model_path):
        raise missing_model_error(model_metadata)

    examples, benchmark_metadata = load_benchmark_examples(
        benchmark=args.benchmark,
        split="test",
        limit=args.limit,
        sample_seed=0,
        sampling="first",
    )
    generator = ClassifierGenerator(effective_model_path)
    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "kind": "run_metadata",
        "schema_version": "category_predictions_v1",
        "benchmark": args.benchmark,
        "benchmark_metadata": benchmark_metadata,
        "model_metadata": {**model_metadata, **generator.model_metadata},
        "categories": list(CATEGORIES),
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "max_new_tokens": args.max_new_tokens,
    }
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for idx, example in enumerate(examples, start=1):
            if not args.quiet:
                print(f"[{idx}/{len(examples)}] {example['problem_id']} true={example.get('type')}")
            prediction, token_meta = generator.classify(example["problem"], max_new_tokens=args.max_new_tokens)
            row = {
                "problem_id": example["problem_id"],
                "problem_index": example["problem_index"],
                "source_id": example.get("source_id"),
                "true_category": example.get("type"),
                "level": example.get("level"),
                "problem": example["problem"],
                **prediction,
                "token_counts": token_meta,
                "category_correct": prediction["predicted_category"] == example.get("type"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    if not args.quiet:
        print(f"Wrote category predictions to {output_path}")


if __name__ == "__main__":
    main()
