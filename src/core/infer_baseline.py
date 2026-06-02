from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import cache_dir, model_dir


def model_files_present(model_path: str | Path) -> bool:
    path = Path(model_path)
    return path.exists() and (path / "config.json").exists() and (
        (path / "tokenizer_config.json").exists() or (path / "tokenizer.json").exists()
    )


def local_model_metadata(model_path: str | Path) -> dict:
    path = Path(model_path)
    files = sorted(item.name for item in path.iterdir()) if path.exists() else []
    return {
        "model_path": str(path),
        "exists": path.exists(),
        "model_files_present": model_files_present(path),
        "files_sample": files[:20],
    }


def load_causal_lm(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_files_present(model_path):
        raise FileNotFoundError(
            f"Model files are missing or incomplete at {model_path}. "
            "Expected at least config.json and tokenizer files. "
            "Download the model into scratch with: bash scripts/download_models.sh"
        )

    cuda_available = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda_available and torch.cuda.is_bf16_supported() else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=str(cache_dir() / "huggingface"), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=str(cache_dir() / "huggingface"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def model_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if device not in {"cpu", "disk", "meta"}:
                return f"cuda:{device}" if isinstance(device, int) else device
    return getattr(model, "device", next(model.parameters()).device)


def runtime_model_metadata(model_path: str, tokenizer=None, model=None) -> dict:
    import torch

    metadata = local_model_metadata(model_path)
    metadata.update(
        {
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch_bfloat16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "requested_torch_dtype": "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "auto",
            "device_map": "auto",
        }
    )
    if tokenizer is not None:
        metadata["tokenizer_class"] = tokenizer.__class__.__name__
        metadata["uses_chat_template"] = bool(getattr(tokenizer, "chat_template", None))
    if model is not None:
        metadata["model_class"] = model.__class__.__name__
        metadata["model_dtype"] = str(getattr(model, "dtype", "unknown"))
        metadata["model_device"] = str(getattr(model, "device", "device_map_auto"))
    return metadata


def move_inputs_to_device(inputs, device) -> dict:
    moved = inputs.to(device)
    return {key: value for key, value in moved.items()}


def build_generation_inputs(tokenizer, prompt: str, device) -> dict:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(rendered, "items"):
            return move_inputs_to_device(rendered, device)
        return {"input_ids": rendered.to(device)}
    return move_inputs_to_device(tokenizer(prompt, return_tensors="pt"), device)


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
    inputs = build_generation_inputs(tokenizer, prompt, model_input_device(model))
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_solution_with_metadata(
    problem: str,
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    prompt: str | None = None,
) -> tuple[str, dict]:
    import torch

    tokenizer, model = load_causal_lm(model_path)
    metadata = runtime_model_metadata(model_path, tokenizer=tokenizer, model=model)
    prompt = prompt or build_prompt(problem)
    inputs = build_generation_inputs(tokenizer, prompt, model_input_device(model))
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    metadata["input_tokens"] = int(inputs["input_ids"].shape[-1])
    metadata["generated_tokens"] = int(generated.shape[-1])
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), metadata


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
