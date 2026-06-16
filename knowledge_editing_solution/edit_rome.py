"""Single-fact ROME editing entry point."""

from __future__ import annotations

import argparse
import time
from typing import Any

from baseline import answer_is_present, evaluate_record
from editing_utils import (
    cleanup_cuda,
    get_editor_tokenizer,
    load_custom_records,
    maybe_cuda_finish,
    maybe_cuda_start,
    resolve_path,
    save_json,
    to_jsonable,
    write_runtime_hparams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ROME edit per custom fact.")
    parser.add_argument("--data", default="data/custom_facts.json", help="Path to the custom fact dataset.")
    parser.add_argument("--output", default="outputs/rome_results.json", help="Path to save ROME results.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name.")
    parser.add_argument("--config", default="configs/rome.yaml", help="EasyEdit ROME hyperparameter YAML.")
    parser.add_argument("--device", default="0", help="CUDA device id for EasyEdit, or cpu if supported.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke tests.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0 uses greedy decoding.")
    return parser.parse_args()


def load_easyedit(config: str, model_name: str, device: str) -> Any:
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
    except ImportError as exc:
        raise SystemExit(
            "Missing EasyEdit. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    runtime_config = write_runtime_hparams(config, model_name=model_name, device=device)
    hparams = ROMEHyperParams.from_hparams(str(runtime_config))
    return BaseEditor.from_hparams(hparams)


def run_single_edit(editor: Any, record: dict[str, Any]) -> Any:
    locality_inputs = {
        "neighborhood": {
            "prompt": [record["locality_prompt"]],
            "ground_truth": [record["locality_ground_truth"]],
        }
    }
    return editor.edit(
        prompts=[record["prompt"]],
        ground_truth=[record["ground_truth"]],
        target_new=[record["target_new"]],
        rephrase_prompts=[record["rephrase_prompt"]],
        locality_inputs=locality_inputs,
        subject=[record.get("subject", "")],
        keep_original_weight=True,
    )


def metric_payload(metrics: Any) -> dict[str, Any]:
    metrics = to_jsonable(metrics)
    if isinstance(metrics, list) and metrics:
        first = metrics[0]
        return first if isinstance(first, dict) else {"raw": first}
    if isinstance(metrics, dict):
        return metrics
    return {"raw": metrics}


def main() -> None:
    args = parse_args()
    records = load_custom_records(args.data, args.limit)
    start_time = time.perf_counter()
    memory = maybe_cuda_start()
    editor = load_easyedit(args.config, args.model, args.device)
    tokenizer = get_editor_tokenizer(editor)
    use_chat_template = bool(getattr(tokenizer, "chat_template", None))

    results = []
    for index, record in enumerate(records, start=1):
        record = dict(record)
        record.setdefault("subject", "")
        print(f"[{index}/{len(records)}] ROME edit: {record['prompt']} -> {record['target_new']}")
        edit_metrics, edited_model, _ = run_single_edit(editor, record)
        generated = evaluate_record(
            record,
            tokenizer,
            edited_model,
            use_chat_template=use_chat_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        generated.update(
            {
                "subject": record.get("subject", ""),
                "easyedit_metrics": metric_payload(edit_metrics),
                "contains_target_new": answer_is_present(generated["generation"], record["target_new"]),
            }
        )
        results.append(generated)
        cleanup_cuda()

    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    payload = {
        "metadata": {
            "method": "ROME",
            "model": args.model,
            "data": str(resolve_path(args.data)),
            "config": str(resolve_path(args.config)),
            "num_records": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "elapsed_seconds": elapsed_seconds,
            "memory": maybe_cuda_finish(memory),
        },
        "results": results,
        "summary": {
            "target_new_hits": sum(item["contains_target_new"] for item in results),
            "rephrase_target_new_hits": sum(item["rephrase_contains_target_new"] for item in results),
            "locality_ground_truth_hits": sum(item["locality_contains_ground_truth"] for item in results),
        },
    }
    save_json(args.output, payload)
    print(f"Saved ROME results to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
