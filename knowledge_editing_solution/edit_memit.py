"""Batch MEMIT editing entry point."""

from __future__ import annotations

import argparse
import time
from typing import Any

from baseline import evaluate_record
from editing_utils import (
    cleanup_cuda,
    get_editor_tokenizer,
    load_benchmark_records,
    load_custom_records,
    maybe_cuda_finish,
    maybe_cuda_start,
    resolve_path,
    save_json,
    to_jsonable,
    write_runtime_hparams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MEMIT batch editing.")
    parser.add_argument(
        "--data",
        default=None,
        help="Optional local JSON data. If omitted, records are downloaded from the benchmark dataset.",
    )
    parser.add_argument("--output", default="outputs/memit_results.json", help="Path to save MEMIT results.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name.")
    parser.add_argument("--config", default="configs/memit.yaml", help="EasyEdit MEMIT hyperparameter YAML.")
    parser.add_argument("--device", default="0", help="CUDA device id for EasyEdit, or cpu if supported.")
    parser.add_argument("--limit", type=int, default=500, help="Number of batch-editing records to use.")
    parser.add_argument("--benchmark-repo", default="zjunlp/KnowEdit", help="Hugging Face dataset repo for MEMIT.")
    parser.add_argument(
        "--benchmark-file",
        default="benchmark/ZsRE/ZsRE-test-all.json",
        help="File inside the Hugging Face dataset repo.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0 uses greedy decoding.")
    return parser.parse_args()


def load_easyedit(config: str, model_name: str, device: str) -> Any:
    try:
        from easyeditor import BaseEditor, MEMITHyperParams
    except ImportError as exc:
        raise SystemExit(
            "Missing EasyEdit. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    runtime_config = write_runtime_hparams(config, model_name=model_name, device=device)
    hparams = MEMITHyperParams.from_hparams(str(runtime_config))
    return BaseEditor.from_hparams(hparams)


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.data:
        return load_custom_records(args.data, args.limit)
    return load_benchmark_records(
        repo_id=args.benchmark_repo,
        repo_file=args.benchmark_file,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )


def run_batch_edit(editor: Any, records: list[dict[str, Any]]) -> Any:
    locality_inputs = {
        "neighborhood": {
            "prompt": [record["locality_prompt"] for record in records],
            "ground_truth": [record["locality_ground_truth"] for record in records],
        }
    }
    return editor.edit(
        prompts=[record["prompt"] for record in records],
        ground_truth=[record["ground_truth"] for record in records],
        target_new=[record["target_new"] for record in records],
        rephrase_prompts=[record["rephrase_prompt"] for record in records],
        locality_inputs=locality_inputs,
        subject=[record.get("subject", "") for record in records],
        keep_original_weight=True,
    )


def main() -> None:
    args = parse_args()
    records = load_records(args)
    start_time = time.perf_counter()
    memory = maybe_cuda_start()
    editor = load_easyedit(args.config, args.model, args.device)
    tokenizer = get_editor_tokenizer(editor)
    use_chat_template = bool(getattr(tokenizer, "chat_template", None))

    print(f"Running one MEMIT batch edit on {len(records)} records.")
    edit_metrics, edited_model, _ = run_batch_edit(editor, records)

    results = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] evaluating edited model: {record['prompt']}")
        generated = evaluate_record(
            record,
            tokenizer,
            edited_model,
            use_chat_template=use_chat_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        generated["subject"] = record.get("subject", "")
        results.append(generated)

    cleanup_cuda()
    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    payload = {
        "metadata": {
            "method": "MEMIT",
            "model": args.model,
            "data": str(resolve_path(args.data)) if args.data else None,
            "benchmark_repo": None if args.data else args.benchmark_repo,
            "benchmark_file": None if args.data else args.benchmark_file,
            "config": str(resolve_path(args.config)),
            "num_records": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "elapsed_seconds": elapsed_seconds,
            "memory": maybe_cuda_finish(memory),
        },
        "easyedit_metrics": to_jsonable(edit_metrics),
        "results": results,
        "summary": {
            "target_new_hits": sum(item["contains_target_new"] for item in results),
            "rephrase_target_new_hits": sum(item["rephrase_contains_target_new"] for item in results),
            "locality_ground_truth_hits": sum(item["locality_contains_ground_truth"] for item in results),
        },
    }
    save_json(args.output, payload)
    print(f"Saved MEMIT results to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
