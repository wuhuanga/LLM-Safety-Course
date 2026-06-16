import argparse
import logging
import time
from typing import Any, Dict, List

import torch

from utils import (
    build_locality_inputs,
    ensure_easyedit_path,
    load_json,
    patch_easyedit_local_stats,
    patch_easyedit_nethook,
    redirect_cuda_to_available_device,
    save_json,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MEMIT batch knowledge editing.")
    parser.add_argument("--data", default="data/custom_10.json")
    parser.add_argument("--hparams", default="hparams/MEMIT/qwen2.5-0.5b.yaml")
    parser.add_argument("--out", default="results/memit_metrics.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_easyedit_path()
    active_device = redirect_cuda_to_available_device()
    if active_device != "cuda":
        print(f"CUDA is unavailable; redirecting EasyEdit CUDA calls to {active_device}.")
    from easyeditor import BaseEditor, MEMITHyperParams
    patch_easyedit_nethook()
    patch_easyedit_local_stats()

    set_seed(args.seed)
    records: List[Dict[str, Any]] = load_json(args.data)
    if args.limit:
        records = records[: args.limit]

    hparams = MEMITHyperParams.from_hparams(args.hparams)
    hparams.batch_size = len(records)
    logging.getLogger("easyeditor.editors.editor").handlers.clear()
    editor = BaseEditor.from_hparams(hparams)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    metrics, _, _ = editor.batch_edit(
        prompts=[r["prompt"] for r in records],
        ground_truth=[r["ground_truth"] for r in records],
        target_new=[r["target_new"] for r in records],
        rephrase_prompts=[r["rephrase_prompt"] for r in records],
        locality_inputs=build_locality_inputs(records),
        subject=[r["subject"] for r in records],
        sequential_edit=False,
    )

    elapsed = time.perf_counter() - start
    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    save_json(
        {
            "algorithm": "MEMIT",
            "hparams": args.hparams,
            "num_records": len(records),
            "elapsed_seconds": elapsed,
            "peak_memory_gb": peak_memory_gb,
            "case_ids": [r.get("case_id") for r in records],
            "metrics": metrics,
        },
        args.out,
    )

    print("\nMEMIT summary")
    print(f"records: {len(records)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"peak_memory_gb: {peak_memory_gb if peak_memory_gb is not None else 'cpu'}")


if __name__ == "__main__":
    main()
