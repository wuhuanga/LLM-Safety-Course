import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EASYEDIT_ROOT = os.path.join(PROJECT_ROOT, "external", "EasyEdit")
if EASYEDIT_ROOT not in sys.path:
    sys.path.append(EASYEDIT_ROOT)

from easyeditor import BaseEditor, MEMITHyperParams  # noqa: E402
from easyeditor.editors.utils import _prepare_requests  # noqa: E402
from easyeditor.util import nethook  # noqa: E402


REMOTE_ROOT_URL = "https://memit.baulab.info/data/dsets"
URL_DICT = {
    "zsre": f"{REMOTE_ROOT_URL}/zsre_mend_eval.json",
    "counterfact": f"{REMOTE_ROOT_URL}/counterfact.json",
}


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
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def ensure_dataset(data_dir: str, dataset_type: str) -> Path:
    path_map = {
        "zsre": Path(data_dir) / "zsre_mend_eval.json",
        "counterfact": Path(data_dir) / "counterfact.json",
    }
    dataset_path = path_map[dataset_type]
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{dataset_path} does not exist. Downloading from {URL_DICT[dataset_type]}")
        torch.hub.download_url_to_file(URL_DICT[dataset_type], dataset_path)
    return dataset_path


def load_batch_records(dataset_path: Path, dataset_type: str, batch_size: int, seed: int) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if dataset_type == "zsre":
        raw = [item for item in raw if item.get("alt", "") != ""]
    elif dataset_type == "counterfact":
        raw = [
            item for item in raw
            if item.get("requested_rewrite", {}).get("target_new", {}).get("str", "") != ""
        ]

    if batch_size > len(raw):
        raise ValueError(f"Requested {batch_size} samples, but filtered dataset only has {len(raw)} valid rows.")

    rng = random.Random(seed)
    sampled = rng.sample(raw, batch_size)
    records: List[Dict[str, Any]] = []

    if dataset_type == "zsre":
        for idx, item in enumerate(sampled):
            records.append(
                {
                    "case_id": idx,
                    "prompt": item["src"],
                    "subject": item["subject"],
                    "target_new": item["alt"],
                    "rephrase_prompt": item["rephrase"],
                    "locality_prompt": item["loc"],
                    "locality_ground_truth": item["loc_ans"],
                }
            )
    elif dataset_type == "counterfact":
        for idx, item in enumerate(sampled):
            rewrite = item["requested_rewrite"]
            records.append(
                {
                    "case_id": idx,
                    "prompt": rewrite["prompt"].format(rewrite["subject"]),
                    "subject": rewrite["subject"],
                    "target_new": rewrite["target_new"]["str"],
                    "rephrase_prompt": item["paraphrase_prompts"][0] if item["paraphrase_prompts"] else None,
                    "locality_prompt": item["neighborhood_prompts"][0] if item["neighborhood_prompts"] else None,
                    "locality_ground_truth": rewrite["target_true"]["str"],
                }
            )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return records


def build_requests(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompts = [item["prompt"] for item in records]
    targets = [item["target_new"] for item in records]
    subjects = [item["subject"] for item in records]
    rephrases = [item["rephrase_prompt"] for item in records]
    locality_prompts = [item["locality_prompt"] for item in records]
    locality_answers = [item["locality_ground_truth"] for item in records]

    requests = _prepare_requests(
        prompts=prompts,
        target_new=targets,
        ground_truth=["<|endoftext|>"] * len(records),
        rephrase_prompts=rephrases,
        locality_inputs={
            "neighborhood": {
                "prompt": locality_prompts,
                "ground_truth": locality_answers,
            }
        },
        portability_inputs=None,
        subject=subjects,
    )

    for request, item in zip(requests, records):
        request["case_id"] = item["case_id"]

    return requests


def evaluate_sample_subset(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    sample_size: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    subset = records[:sample_size]
    details: List[Dict[str, Any]] = []
    rewrite_success = 0

    for item in subset:
        output = generate_answer(model, tokenizer, item["prompt"], max_new_tokens)
        ok = contains_target(output, item["target_new"])
        rewrite_success += int(ok)
        details.append(
            {
                "case_id": item["case_id"],
                "prompt": item["prompt"],
                "target_new": item["target_new"],
                "model_output_after_batch_edit": output,
                "rewrite_success": ok,
            }
        )

    summary = {
        "sample_size": len(subset),
        "rewrite_success_count": rewrite_success,
        "rewrite_success_rate": rewrite_success / len(subset) if subset else 0.0,
    }
    return details, summary


def evaluate_comprehensive_subset(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    sample_size: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    subset = records[:sample_size]
    details: List[Dict[str, Any]] = []
    efficacy_success = 0
    generalization_total = 0
    generalization_success = 0
    locality_total = 0
    locality_success = 0

    for item in subset:
        prompt_output = generate_answer(model, tokenizer, item["prompt"], max_new_tokens)
        rewrite_ok = contains_target(prompt_output, item["target_new"])
        efficacy_success += int(rewrite_ok)

        rephrase_output = None
        rephrase_ok = None
        if item.get("rephrase_prompt"):
            rephrase_output = generate_answer(model, tokenizer, item["rephrase_prompt"], max_new_tokens)
            rephrase_ok = contains_target(rephrase_output, item["target_new"])
            generalization_total += 1
            generalization_success += int(rephrase_ok)

        locality_output = None
        locality_ok = None
        if item.get("locality_prompt") and item.get("locality_ground_truth"):
            locality_output = generate_answer(model, tokenizer, item["locality_prompt"], max_new_tokens)
            locality_ok = contains_target(locality_output, item["locality_ground_truth"])
            locality_total += 1
            locality_success += int(locality_ok)

        details.append(
            {
                "case_id": item["case_id"],
                "prompt": item["prompt"],
                "target_new": item["target_new"],
                "prompt_output_after_batch_edit": prompt_output,
                "rewrite_success": rewrite_ok,
                "rephrase_prompt": item.get("rephrase_prompt"),
                "rephrase_output_after_batch_edit": rephrase_output,
                "rephrase_success": rephrase_ok,
                "locality_prompt": item.get("locality_prompt"),
                "locality_ground_truth": item.get("locality_ground_truth"),
                "locality_output_after_batch_edit": locality_output,
                "locality_preserved": locality_ok,
            }
        )

    summary = {
        "sample_size": len(subset),
        "efficacy_success_count": efficacy_success,
        "efficacy_success_rate": efficacy_success / len(subset) if subset else 0.0,
        "generalization_success_count": generalization_success,
        "generalization_success_rate": (
            generalization_success / generalization_total if generalization_total else None
        ),
        "locality_preserved_count": locality_success,
        "locality_preserved_rate": locality_success / locality_total if locality_total else None,
    }
    return details, summary


def restore_original_weights(editor: BaseEditor, weights_copy: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, original_value in weights_copy.items():
            param = nethook.get_parameter(editor.model, name)
            param[...] = original_value.to(param.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="zsre", choices=["zsre", "counterfact"])
    parser.add_argument("--data_dir", type=str, default="data/batch_edit")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hparams_path", type=str, default="configs/qwen2.5_0.5b_memit.yaml")
    parser.add_argument("--output_path", type=str, default="outputs/task3_memit_results.json")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--eval_sample_size", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    dataset_path = ensure_dataset(args.data_dir, args.dataset_type)
    records = load_batch_records(dataset_path, args.dataset_type, args.batch_size, args.seed)
    requests = build_requests(records)

    hparams = MEMITHyperParams.from_hparams(args.hparams_path)
    if args.device is not None:
        hparams.device = args.device
    if args.model_name is not None:
        hparams.model_name = args.model_name
    hparams.batch_size = args.batch_size

    os.makedirs(hparams.stats_dir, exist_ok=True)

    print(f"Dataset: {args.dataset_type}")
    print(f"Dataset path: {dataset_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using model: {hparams.model_name}")
    print(f"Using device: {hparams.device}")

    editor = BaseEditor.from_hparams(hparams)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    edited_model, weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        requests,
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    peak_memory_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    peak_memory_gb = peak_memory_bytes / (1024 ** 3) if peak_memory_bytes is not None else None

    eval_details, eval_summary = evaluate_sample_subset(
        edited_model,
        editor.tok,
        records,
        min(args.eval_sample_size, len(records)),
        args.max_new_tokens,
    )

    comprehensive_details, comprehensive_summary = evaluate_comprehensive_subset(
        edited_model,
        editor.tok,
        records,
        min(args.eval_sample_size, len(records)),
        args.max_new_tokens,
    )

    restore_original_weights(editor, weights_copy)

    summary = {
        "task": "Task 3 Batch Editing with MEMIT",
        "dataset_type": args.dataset_type,
        "dataset_path": str(dataset_path),
        "batch_size": len(records),
        "model_name": hparams.model_name,
        "device": hparams.device,
        "edit_time_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": peak_memory_gb,
        "sample_eval": eval_summary,
        "task4_metrics": comprehensive_summary,
    }

    save_json(
        {
            "summary": summary,
            "sample_eval_details": eval_details,
            "task4_eval_details": comprehensive_details,
        },
        args.output_path,
    )

    print("=" * 100)
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
