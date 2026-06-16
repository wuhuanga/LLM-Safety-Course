import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
EASYEDIT_ROOT = PROJECT_ROOT / "EasyEdit"
if str(EASYEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYEDIT_ROOT))

from easyeditor import BaseEditor, ROMEHyperParams


MODEL_NAME = PROJECT_ROOT / "model" / "Qwen2.5-0.5B-Instruct"
DATA_PATH = PROJECT_ROOT / "data" / "crosslingual_edits.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "rome_crosslingual_results.json"
HPARAMS_PATH = PROJECT_ROOT / "EasyEdit" / "hparams" / "ROME" / "qwen2.5-0.5b.yaml"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_answer(prediction: str, *answers: str) -> bool:
    normalized_prediction = normalize_text(prediction)
    for answer in answers:
        if answer and normalize_text(answer) in normalized_prediction:
            return True
    return False


def build_answer_only_zh_prompt(prompt: str) -> str:
    return (
        "请只输出最终答案，不要解释，不要输出选项，不要重复题目。\n"
        f"问题：{prompt}\n"
        "答案："
    )


def looks_like_multiple_choice_output(text: str) -> bool:
    upper_text = text.upper()
    option_markers = ["A.", "B.", "C.", "D.", "A、", "B、", "C、", "D、", "（A", "（B", "（C", "（D"]
    answer_markers = ["答案：A", "答案：B", "答案：C", "答案：D", "ANSWER: A", "ANSWER: B", "ANSWER: C", "ANSWER: D"]
    return any(marker in upper_text for marker in option_markers + answer_markers)


def extract_short_answer(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = lines[0] if lines else text.strip()
    prefixes = ["答案：", "答案:", "答：", "答:", "最终答案：", "最终答案:", "answer:", "Answer:"]
    for prefix in prefixes:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
    candidate = candidate.strip(" ：:，,。；;（）()[]【】\"'“”")
    return candidate


def strict_answer_match(prediction: str, *answers: str) -> bool:
    if not prediction.strip():
        return False
    if looks_like_multiple_choice_output(prediction):
        return False

    candidate = extract_short_answer(prediction)
    normalized_candidate = normalize_text(candidate)
    if not normalized_candidate:
        return False

    for answer in answers:
        if not answer:
            continue
        normalized_answer = normalize_text(answer)
        if (
            normalized_candidate == normalized_answer
            or normalized_candidate.startswith(normalized_answer)
            or normalized_answer.startswith(normalized_candidate)
        ):
            return True
    return False


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 32, answer_only: bool = False) -> str:
    actual_prompt = build_answer_only_zh_prompt(prompt) if answer_only else prompt
    inputs = tokenizer(actual_prompt, return_tensors="pt").to(model.device)

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


def extract_core_metrics(metric: dict) -> dict:
    pre = metric.get("pre", {})
    post = metric.get("post", {})
    locality_post = post.get("locality", {})

    locality_acc = None
    for key, value in locality_post.items():
        if key.endswith("_acc"):
            locality_acc = value
            break

    return {
        "case_id": metric.get("case_id"),
        "time": metric.get("time"),
        "requested_rewrite": metric.get("requested_rewrite"),
        "pre": {
            "rewrite_acc": pre.get("rewrite_acc"),
            "rephrase_acc": pre.get("rephrase_acc"),
            "rewrite_ppl": pre.get("rewrite_ppl"),
            "ood_acc": pre.get("ood_acc"),
        },
        "post": {
            "rewrite_acc": post.get("rewrite_acc"),
            "rephrase_acc": post.get("rephrase_acc"),
            "rewrite_ppl": post.get("rewrite_ppl"),
            "ood_acc": post.get("ood_acc"),
            "locality_acc": locality_acc,
        },
    }


def evaluate_generation(sample: dict, rewrite_answer: str, rephrase_answer: str, zh_rephrase_answer: str, locality_answer: str) -> dict:
    target_new = sample["target_new"]
    zh_target_new = sample.get("zh_target_new", "")
    locality_ground_truth = sample["locality_ground_truth"]

    return {
        "rewrite_hit": contains_answer(rewrite_answer, target_new, zh_target_new),
        "rephrase_hit": contains_answer(rephrase_answer, target_new, zh_target_new),
        "zh_rephrase_hit": strict_answer_match(zh_rephrase_answer, target_new, zh_target_new),
        "locality_hit": contains_answer(locality_answer, locality_ground_truth),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if not HPARAMS_PATH.exists():
        raise FileNotFoundError(f"ROME hparams file not found: {HPARAMS_PATH}")
    if not Path(MODEL_NAME).exists():
        raise FileNotFoundError(f"Local model path not found: {MODEL_NAME}")

    print(f"Loading tokenizer and model from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    base_model.eval()

    samples = load_json(DATA_PATH)
    hparams = ROMEHyperParams.from_hparams(str(HPARAMS_PATH))
    all_results = []

    for idx, item in enumerate(samples, start=1):
        prompt = item["prompt"]
        target_new = item["target_new"]
        ground_truth = item["ground_truth"]
        rephrase_prompt = item["rephrase_prompt"]
        zh_rephrase_prompt = item["zh_rephrase_prompt"]
        locality_prompt = item["locality_prompt"]
        locality_ground_truth = item["locality_ground_truth"]
        subject = item["subject"]

        print(f"\n===== Cross-lingual ROME Edit [{idx}/{len(samples)}] =====")
        print(f"Prompt: {prompt}")
        print(f"Target new: {target_new}")
        print(f"Chinese prompt: {zh_rephrase_prompt}")

        editor = BaseEditor.from_hparams(hparams)
        editor.model = base_model
        editor.tok = tokenizer

        start_time = time.perf_counter()
        metrics, edited_model, _ = editor.edit(
            prompts=prompt,
            target_new=target_new,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompt,
            locality_inputs={
                "neighborhood": {
                    "prompt": locality_prompt,
                    "ground_truth": locality_ground_truth,
                }
            },
            subject=subject,
            sequential_edit=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - start_time

        rewrite_answer = generate_answer(edited_model, tokenizer, prompt)
        rephrase_answer = generate_answer(edited_model, tokenizer, rephrase_prompt)
        zh_rephrase_answer = generate_answer(edited_model, tokenizer, zh_rephrase_prompt, answer_only=True)
        locality_answer = generate_answer(edited_model, tokenizer, locality_prompt)
        generation_eval = evaluate_generation(
            item,
            rewrite_answer=rewrite_answer,
            rephrase_answer=rephrase_answer,
            zh_rephrase_answer=zh_rephrase_answer,
            locality_answer=locality_answer,
        )

        raw_metric = metrics[0]
        result = {
            "id": idx,
            "prompt": prompt,
            "target_new": target_new,
            "zh_target_new": item.get("zh_target_new", ""),
            "ground_truth": ground_truth,
            "rephrase_prompt": rephrase_prompt,
            "zh_rephrase_prompt": zh_rephrase_prompt,
            "zh_answer_only_prompt": build_answer_only_zh_prompt(zh_rephrase_prompt),
            "locality_prompt": locality_prompt,
            "locality_ground_truth": locality_ground_truth,
            "subject": subject,
            "metrics": extract_core_metrics(raw_metric),
            "raw_metrics": raw_metric,
            "generations": {
                "rewrite_answer": rewrite_answer,
                "rephrase_answer": rephrase_answer,
                "zh_rephrase_answer": zh_rephrase_answer,
                "zh_short_answer": extract_short_answer(zh_rephrase_answer),
                "locality_answer": locality_answer,
            },
            "crosslingual_eval": generation_eval,
            "elapsed_seconds": round(elapsed, 4),
        }
        all_results.append(result)
        save_json(all_results, OUTPUT_PATH)

        print("Post-edit generation summary:")
        print(f"  rewrite_answer: {rewrite_answer}")
        print(f"  rephrase_answer: {rephrase_answer}")
        print(f"  zh_rephrase_answer: {zh_rephrase_answer}")
        print(f"  zh_short_answer: {extract_short_answer(zh_rephrase_answer)}")
        print(f"  locality_answer: {locality_answer}")
        print(f"  zh_rephrase_hit: {generation_eval['zh_rephrase_hit']}")

        del editor
        del edited_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n===== Cross-lingual ROME Finished =====")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
