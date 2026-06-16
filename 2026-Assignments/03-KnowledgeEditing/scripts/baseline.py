#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def contains_answer(pred: str, answer: str) -> bool:
    return normalize_text(answer) in normalize_text(pred)


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.7
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id
    }
    # 清理 None
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    output_ids = model.generate(**inputs, **gen_kwargs)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 尽量裁掉 prompt 前缀
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


def run_baseline(
    model_name_or_path: str,
    data_path: str,
    output_path: str,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.7,
    device_map: str = "auto",
    torch_dtype: str = "float16"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    if device_map and device_map.lower() in {"none", "null"}:
        device_map = None

    print(f"[INFO] Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    if device_map is None:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(data_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    results = []
    start = time.time()

    for i, item in enumerate(data):
        prompt = item["prompt"]
        rephrase_prompt = item.get("rephrase_prompt", "")
        locality_prompt = item.get("locality_prompt", "")

        pred_main = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature
        )
        pred_rephrase = generate_text(
            model, tokenizer, rephrase_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature
        ) if rephrase_prompt else ""

        pred_locality = generate_text(
            model, tokenizer, locality_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature
        ) if locality_prompt else ""

        out = dict(item)
        out.update({
            "pred_main": pred_main,
            "pred_rephrase": pred_rephrase,
            "pred_locality": pred_locality,
            "hit_main_target_new": contains_answer(pred_main, item.get("target_new", "")),
            "hit_rephrase_target_new": contains_answer(pred_rephrase, item.get("target_new", "")) if rephrase_prompt else None,
            "hit_locality_ground_truth": contains_answer(pred_locality, item.get("locality_ground_truth", "")) if locality_prompt else None
        })
        results.append(out)
        print(f"[{i+1}/{len(data)}] done")

    elapsed = time.time() - start
    payload = {
        "meta": {
            "model_name_or_path": model_name_or_path,
            "data_path": data_path,
            "num_samples": len(data),
            "elapsed_sec": elapsed
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved baseline results to: {output_path}")
    print(f"[INFO] Elapsed: {elapsed:.2f}s")


def build_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation without editing.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/custom_10.json")
    parser.add_argument("--output_path", type=str, default="outputs/baseline_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run_baseline(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype
    )