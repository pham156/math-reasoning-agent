from __future__ import annotations

import argparse

from src.core.config import cache_dir, model_dir


def load_causal_lm(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=str(cache_dir() / "huggingface"), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache_dir() / "huggingface"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def build_prompt(problem: str) -> str:
    return (
        "You are a careful mathematical reasoning assistant. "
        "Solve the problem step by step, justify each claim, and give the final answer clearly.\n\n"
        f"Problem:\n{problem}\n\nSolution:"
    )


def generate_solution(
    problem: str,
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    prompt: str | None = None,
) -> str:
    import torch

    tokenizer, model = load_causal_lm(model_path)
    prompt = prompt or build_prompt(problem)
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        inputs = {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline math inference.")
    parser.add_argument("--model_path", default=str(model_dir("Qwen2.5-Math-7B-Instruct")))
    parser.add_argument("--problem", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    print(generate_solution(args.problem, args.model_path, args.max_new_tokens, args.temperature))


if __name__ == "__main__":
    main()
