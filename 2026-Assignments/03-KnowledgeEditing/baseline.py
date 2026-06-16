import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = PROJECT_ROOT / "model" / "Qwen2.5-0.5B-Instruct"
DATA_PATH = PROJECT_ROOT / "data" / "custom_edits.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_results.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_answer(prediction: str, answer: str) -> bool:
    return normalize_text(answer) in normalize_text(prediction)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    if not Path(MODEL_NAME).exists():
        raise FileNotFoundError(f"Local model path not found: {MODEL_NAME}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    samples = load_json(DATA_PATH)
    results = []

    for idx, item in enumerate(samples, start=1):
        prompt = item["prompt"]
        target_new = item["target_new"]
        ground_truth = item["ground_truth"]

        print(f"\n[{idx}/{len(samples)}] Prompt: {prompt}")
        answer = generate_answer(model, tokenizer, prompt)
        print(f"Model answer: {answer}")

        result = {
            "id": idx,
            "prompt": prompt,
            "model_answer": answer,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "matches_target_new": contains_answer(answer, target_new),
            "matches_ground_truth": contains_answer(answer, ground_truth),
            "rephrase_prompt": item.get("rephrase_prompt", ""),
            "locality_prompt": item.get("locality_prompt", ""),
            "locality_ground_truth": item.get("locality_ground_truth", ""),
        }
        results.append(result)

    save_json(results, OUTPUT_PATH)

    total = len(results)
    target_new_hits = sum(r["matches_target_new"] for r in results)
    ground_truth_hits = sum(r["matches_ground_truth"] for r in results)

    print("\n===== Baseline Summary =====")
    print(f"Total samples: {total}")
    print(f"Already matches target_new: {target_new_hits}/{total}")
    print(f"Matches ground_truth: {ground_truth_hits}/{total}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
