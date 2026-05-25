"""Baseline evaluation entry point.

Query the unedited base model on the custom fact set and save structured
generations for later ES/PS/NS evaluation.
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Any

from editing_utils import load_custom_records, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline evaluation before editing.")
    parser.add_argument("--data", default="data/custom_facts.json", help="Path to the custom fact dataset.")
    parser.add_argument("--output", default="outputs/baseline.json", help="Path to save baseline generations.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name.")
    parser.add_argument("--device", default="auto", help="Device placement: auto, cuda, cpu, or a device id.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype to request from transformers.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0 uses greedy decoding.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke tests.")
    parser.add_argument(
        "--chat-template",
        choices=["auto", "always", "never"],
        default="auto",
        help="Use the tokenizer chat template for instruction-tuned models.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def answer_is_present(generation: str, answer: str) -> bool:
    return normalize_text(answer).casefold() in normalize_text(generation).casefold()


def should_use_chat_template(tokenizer: Any, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(getattr(tokenizer, "chat_template", None))


def build_model_input(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    prompt = prompt.strip()
    if use_chat_template:
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\nAnswer with only the short factual answer.",
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{prompt} "


def dtype_from_name(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(model_name: str, device: str, dtype_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependencies. Install them with: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = dtype_from_name(torch, dtype_name)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    return tokenizer, model


def generate_answer(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    use_chat_template: bool,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    model_input = build_model_input(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)

    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    return normalize_text(tokenizer.decode(new_tokens, skip_special_tokens=True))


def evaluate_record(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    *,
    use_chat_template: bool,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    direct_generation = generate_answer(
        tokenizer,
        model,
        record["prompt"],
        use_chat_template=use_chat_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    rephrase_generation = generate_answer(
        tokenizer,
        model,
        record["rephrase_prompt"],
        use_chat_template=use_chat_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    locality_generation = generate_answer(
        tokenizer,
        model,
        record["locality_prompt"],
        use_chat_template=use_chat_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return {
        "prompt": record["prompt"],
        "target_new": record["target_new"],
        "ground_truth": record["ground_truth"],
        "generation": direct_generation,
        "contains_target_new": answer_is_present(direct_generation, record["target_new"]),
        "contains_ground_truth": answer_is_present(direct_generation, record["ground_truth"]),
        "rephrase_prompt": record["rephrase_prompt"],
        "rephrase_generation": rephrase_generation,
        "rephrase_contains_target_new": answer_is_present(rephrase_generation, record["target_new"]),
        "locality_prompt": record["locality_prompt"],
        "locality_ground_truth": record["locality_ground_truth"],
        "locality_generation": locality_generation,
        "locality_contains_ground_truth": answer_is_present(
            locality_generation,
            record["locality_ground_truth"],
        ),
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    output_path = resolve_path(args.output)
    records = load_custom_records(data_path, args.limit)

    start_time = time.perf_counter()
    tokenizer, model = load_model(args.model, args.device, args.torch_dtype)
    use_chat_template = should_use_chat_template(tokenizer, args.chat_template)

    results = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['prompt']}")
        results.append(
            evaluate_record(
                record,
                tokenizer,
                model,
                use_chat_template=use_chat_template,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )

    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    payload = {
        "metadata": {
            "model": args.model,
            "data": str(data_path),
            "num_records": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "chat_template": use_chat_template,
            "elapsed_seconds": elapsed_seconds,
        },
        "results": results,
        "summary": {
            "target_new_hits": sum(item["contains_target_new"] for item in results),
            "ground_truth_hits": sum(item["contains_ground_truth"] for item in results),
            "rephrase_target_new_hits": sum(item["rephrase_contains_target_new"] for item in results),
            "locality_ground_truth_hits": sum(item["locality_contains_ground_truth"] for item in results),
        },
    }
    save_json(output_path, payload)
    print(
        f"Saved {len(results)} baseline records to {output_path} "
        f"in {elapsed_seconds:.3f} seconds."
    )


if __name__ == "__main__":
    main()
