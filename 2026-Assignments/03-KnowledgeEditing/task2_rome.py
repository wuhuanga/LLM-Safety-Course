import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EASYEDIT_ROOT = os.path.join(PROJECT_ROOT, "external", "EasyEdit")
if EASYEDIT_ROOT not in sys.path:
    sys.path.append(EASYEDIT_ROOT)

from easyeditor import BaseEditor, ROMEHyperParams  # noqa: E402
from easyeditor.editors.utils import _prepare_requests  # noqa: E402
from easyeditor.util import nethook  # noqa: E402


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def contains_target(output: str, target: str) -> bool:
    return normalize_text(target) in normalize_text(output)


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    device = get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def build_locality_inputs(item: Dict[str, Any]) -> Optional[Dict[str, Dict[str, List[str]]]]:
    locality_prompt = item.get("locality_prompt")
    locality_ground_truth = item.get("locality_ground_truth")
    if not locality_prompt or not locality_ground_truth:
        return None

    return {
        "neighborhood": {
            "prompt": [locality_prompt],
            "ground_truth": [locality_ground_truth],
        }
    }


def build_request(item: Dict[str, Any]) -> Dict[str, Any]:
    requests = _prepare_requests(
        prompts=[item["prompt"]],
        target_new=[item["target_new"]],
        ground_truth=["<|endoftext|>"],
        rephrase_prompts=[item.get("rephrase_prompt")] if item.get("rephrase_prompt") else None,
        locality_inputs=build_locality_inputs(item),
        portability_inputs=None,
        subject=[item["subject"]],
    )
    return requests[0]


def restore_original_weights(editor: BaseEditor, weights_copy: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, original_value in weights_copy.items():
            param = nethook.get_parameter(editor.model, name)
            param[...] = original_value.to(param.device)


def run_single_edit(
    editor: BaseEditor,
    item: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    request = build_request(item)

    prompt_before = generate_answer(editor.model, editor.tok, item["prompt"], max_new_tokens)
    rephrase_before = None
    locality_before = None

    if item.get("rephrase_prompt"):
        rephrase_before = generate_answer(editor.model, editor.tok, item["rephrase_prompt"], max_new_tokens)
    if item.get("locality_prompt"):
        locality_before = generate_answer(editor.model, editor.tok, item["locality_prompt"], max_new_tokens)

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start_event is not None:
        torch.cuda.synchronize()
        start_event.record()

    edited_model, weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        [request],
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )

    edit_time_ms = None
    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize()
        edit_time_ms = start_event.elapsed_time(end_event)

    prompt_after = generate_answer(edited_model, editor.tok, item["prompt"], max_new_tokens)
    rephrase_after = None
    locality_after = None

    if item.get("rephrase_prompt"):
        rephrase_after = generate_answer(edited_model, editor.tok, item["rephrase_prompt"], max_new_tokens)
    if item.get("locality_prompt"):
        locality_after = generate_answer(edited_model, editor.tok, item["locality_prompt"], max_new_tokens)

    restore_original_weights(editor, weights_copy)

    target_new = item["target_new"]
    locality_ground_truth = item.get("locality_ground_truth", "")

    return {
        "prompt": item["prompt"],
        "subject": item["subject"],
        "target_new": target_new,
        "rephrase_prompt": item.get("rephrase_prompt"),
        "locality_prompt": item.get("locality_prompt"),
        "locality_ground_truth": locality_ground_truth,
        "model_output_before_edit": prompt_before,
        "model_output_after_edit": prompt_after,
        "rewrite_success": contains_target(prompt_after, target_new),
        "rewrite_success_before": contains_target(prompt_before, target_new),
        "rephrase_output_before_edit": rephrase_before,
        "rephrase_output_after_edit": rephrase_after,
        "rephrase_success": contains_target(rephrase_after, target_new) if rephrase_after is not None else None,
        "locality_output_before_edit": locality_before,
        "locality_output_after_edit": locality_after,
        "locality_preserved": contains_target(locality_after, locality_ground_truth) if locality_after is not None else None,
        "edit_time_ms": edit_time_ms,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    rewrite_success = sum(1 for item in results if item["rewrite_success"])
    rephrase_total = sum(1 for item in results if item["rephrase_success"] is not None)
    rephrase_success = sum(1 for item in results if item["rephrase_success"] is True)
    locality_total = sum(1 for item in results if item["locality_preserved"] is not None)
    locality_success = sum(1 for item in results if item["locality_preserved"] is True)

    edit_times = [item["edit_time_ms"] for item in results if item["edit_time_ms"] is not None]

    return {
        "total_samples": total,
        "rewrite_success_count": rewrite_success,
        "rewrite_success_rate": rewrite_success / total if total else 0.0,
        "rephrase_success_count": rephrase_success,
        "rephrase_success_rate": rephrase_success / rephrase_total if rephrase_total else None,
        "locality_preserved_count": locality_success,
        "locality_preserved_rate": locality_success / locality_total if locality_total else None,
        "avg_edit_time_ms": sum(edit_times) / len(edit_times) if edit_times else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/custom_10_with_subject.json")
    parser.add_argument("--hparams_path", type=str, default="configs/qwen2.5_0.5b_rome.yaml")
    parser.add_argument("--output_path", type=str, default="outputs/task2_rome_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_json(args.data_path)
    if args.limit is not None:
        data = data[: args.limit]

    hparams = ROMEHyperParams.from_hparams(args.hparams_path)
    if args.device is not None:
        hparams.device = args.device
    if args.model_name is not None:
        hparams.model_name = args.model_name

    os.makedirs(hparams.stats_dir, exist_ok=True)

    print(f"Loading ROME hparams from: {args.hparams_path}")
    print(f"Using model: {hparams.model_name}")
    print(f"Using device: {hparams.device}")
    print(f"Editing samples: {len(data)}")

    editor = BaseEditor.from_hparams(hparams)
    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(data):
        if "subject" not in item:
            raise ValueError(
                f"Sample {idx} is missing `subject`. "
                "Use data/custom_10_with_subject.json or add a subject field first."
            )

        print("=" * 100)
        print(f"[{idx}] Editing: {item['prompt']} -> {item['target_new']}")
        result = run_single_edit(editor, item, args.max_new_tokens)
        results.append(result)
        print(f"After edit: {result['model_output_after_edit']}")
        print(f"Rewrite success: {result['rewrite_success']}")
        if result["rephrase_success"] is not None:
            print(f"Rephrase success: {result['rephrase_success']}")
        if result["locality_preserved"] is not None:
            print(f"Locality preserved: {result['locality_preserved']}")

    summary = summarize(results)
    save_json({"summary": summary, "results": results}, args.output_path)

    print("=" * 100)
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
