import argparse
import json
import os
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def contains_target(output: str, target: str) -> bool:
    return normalize_text(target) in normalize_text(output)


def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 20) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data_path", type=str, default="data/custom_10.json")
    parser.add_argument("--output_path", type=str, default="outputs/baseline_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    data = load_json(args.data_path)

    results = []

    print("\nBaseline evaluation results:")
    print("-" * 100)

    for idx, item in enumerate(data):
        prompt = item["prompt"]
        target_new = item["target_new"]

        output = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens
        )

        is_correct_before_edit = contains_target(output, target_new)

        result = {
            "id": idx,
            "prompt": prompt,
            "target_new": target_new,
            "ground_truth": item.get("ground_truth", ""),
            "model_output_before_edit": output,
            "correct_before_edit": is_correct_before_edit,
            "rephrase_prompt": item.get("rephrase_prompt", ""),
            "locality_prompt": item.get("locality_prompt", ""),
            "locality_ground_truth": item.get("locality_ground_truth", "")
        }

        results.append(result)

        print(f"[{idx}]")
        print(f"Prompt: {prompt}")
        print(f"Target New: {target_new}")
        print(f"Model Output: {output}")
        print(f"Correct Before Edit: {is_correct_before_edit}")
        print("-" * 100)

    save_json(results, args.output_path)

    total = len(results)
    correct = sum(1 for x in results if x["correct_before_edit"])
    incorrect = total - correct

    print("\nSummary:")
    print(f"Total samples: {total}")
    print(f"Already correct before editing: {correct}")
    print(f"Incorrect / unknown before editing: {incorrect}")
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
