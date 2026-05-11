import argparse
import logging
import time
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from utils import (
    build_locality_inputs,
    ensure_easyedit_path,
    load_json,
    patch_easyedit_nethook,
    redirect_cuda_to_available_device,
    save_json,
    set_seed,
)


def run_single_edit(record: Dict[str, Any], hparams_path: str) -> Dict[str, Any]:
    ensure_easyedit_path()
    active_device = redirect_cuda_to_available_device()
    if active_device != "cuda":
        print(f"CUDA is unavailable; redirecting EasyEdit CUDA calls to {active_device}.")
    from easyeditor import BaseEditor, ROMEHyperParams
    patch_easyedit_nethook()

    hparams = ROMEHyperParams.from_hparams(hparams_path)
    logging.getLogger("easyeditor.editors.editor").handlers.clear()
    editor = BaseEditor.from_hparams(hparams)

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metrics, _, _ = editor.edit(
        prompts=[record["prompt"]],
        ground_truth=[record["ground_truth"]],
        target_new=[record["target_new"]],
        rephrase_prompts=[record["rephrase_prompt"]],
        locality_inputs=build_locality_inputs([record]),
        subject=[record["subject"]],
        sequential_edit=False,
    )

    elapsed = time.perf_counter() - start
    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "case_id": record.get("case_id"),
        "prompt": record["prompt"],
        "target_new": record["target_new"],
        "elapsed_seconds": elapsed,
        "peak_memory_gb": peak_memory_gb,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROME single-fact editing for each custom case.")
    parser.add_argument("--data", default="data/custom_10.json")
    parser.add_argument("--hparams", default="hparams/ROME/qwen2.5-0.5b.yaml")
    parser.add_argument("--out", default="results/rome_metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    records: List[Dict[str, Any]] = load_json(args.data)
    if args.limit:
        records = records[: args.limit]

    case_results = []
    for record in tqdm(records, desc="ROME edits"):
        case_results.append(run_single_edit(record, args.hparams))

    save_json(
        {
            "algorithm": "ROME",
            "hparams": args.hparams,
            "num_records": len(case_results),
            "case_results": case_results,
        },
        args.out,
    )

    print("\nROME summary")
    for item in case_results:
        print(
            f"{item['case_id']}\t{item['elapsed_seconds']:.2f}s\t"
            f"{item['peak_memory_gb'] if item['peak_memory_gb'] is not None else 'cpu'}"
        )


if __name__ == "__main__":
    main()
