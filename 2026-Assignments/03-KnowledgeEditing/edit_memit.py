import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
EASYEDIT_ROOT = PROJECT_ROOT / "EasyEdit"
if str(EASYEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYEDIT_ROOT))

from easyeditor import BaseEditor, MEMITHyperParams


def setup_accelerator() -> str:
    """Detect GPU, enable performance flags, and return device string."""
    if torch.cuda.is_available():
        device = "cuda:0"
        # TF32 gives ~3x speedup on Ampere+ GPUs with negligible precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Auto-tune convolution algorithms for fixed input sizes
        torch.backends.cudnn.benchmark = True
        # PyTorch 2.0+ high-precision matmul shortcut
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("CUDA not available — running on CPU")
    return device


MEMIT_DATA_PATH = Path("data/memit_500.json")
CUSTOM_DATA_PATH = Path("data/custom_edits.json")
OUTPUT_PATH = Path("results/memit_results.json")
HPARAMS_PATH = Path("EasyEdit/hparams/MEMIT/qwen2.5-0.5b.yaml")
MEMIT_SIZE = 500


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_subject(prompt: str) -> str:
    fallback_rules = [
        ("The CEO of ", " is"),
        ("The head coach of ", " is"),
        ("The prime minister of ", " is"),
        ("The monarch of ", " is"),
        ("The president of ", " is"),
        ("The capital of ", " is"),
    ]
    for prefix, suffix in fallback_rules:
        if prompt.startswith(prefix) and prompt.endswith(suffix):
            return prompt[len(prefix):-len(suffix)].strip()

    raise ValueError(f"Could not infer subject from prompt: {prompt}")


def normalize_record(record: dict) -> dict | None:
    if {
        "prompt",
        "target_new",
        "ground_truth",
        "rephrase_prompt",
        "locality_prompt",
        "locality_ground_truth",
    }.issubset(record.keys()):
        normalized = {
            "prompt": record["prompt"],
            "target_new": record["target_new"],
            "ground_truth": record["ground_truth"],
            "rephrase_prompt": record["rephrase_prompt"],
            "locality_prompt": record["locality_prompt"],
            "locality_ground_truth": record["locality_ground_truth"],
        }
        normalized["subject"] = record.get("subject") or build_subject(record["prompt"])
        return normalized

    if {"src", "alt", "answers", "rephrase", "loc", "loc_ans"}.issubset(record.keys()):
        ground_truth = record["answers"]
        while isinstance(ground_truth, list):
            if not ground_truth:
                return None
            ground_truth = ground_truth[0]
        normalized = {
            "prompt": record["src"],
            "target_new": record["alt"],
            "ground_truth": ground_truth,
            "rephrase_prompt": record["rephrase"],
            "locality_prompt": record["loc"],
            "locality_ground_truth": record["loc_ans"],
        }
        normalized["subject"] = record.get("subject") or build_subject(record["src"])
        return normalized

    # CounterFact-style: known_id / subject / attribute / template / prompt
    if {"known_id", "subject", "attribute", "template", "prompt"}.issubset(record.keys()):
        subject = record["subject"]
        attribute = record["attribute"]
        template = record["template"]
        rephrase = f"What is the answer to: {template.replace('{}', subject)}?"
        return {
            "prompt": record["prompt"],
            "target_new": attribute,
            "ground_truth": attribute,
            "rephrase_prompt": rephrase,
            "locality_prompt": "The Eiffel Tower is located in",
            "locality_ground_truth": "Paris",
            "subject": subject,
        }

    return None


def prepare_memit_dataset() -> tuple[list[dict], str]:
    if MEMIT_DATA_PATH.exists():
        raw = load_json(MEMIT_DATA_PATH)
        normalized = []
        for record in raw:
            item = normalize_record(record)
            if item is not None:
                normalized.append(item)
            if len(normalized) >= MEMIT_SIZE:
                break
        if normalized:
            return normalized[:MEMIT_SIZE], f"local cache: {MEMIT_DATA_PATH}"

    dataset_candidates = [
        ("zjunlp/KnowEdit", "zsre", "train"),
        ("zjunlp/KnowEdit", "counterfact", "train"),
    ]

    for dataset_name, config_name, split_name in dataset_candidates:
        try:
            hf_dataset = load_dataset(dataset_name, config_name, split=split_name)
            normalized = []
            for record in hf_dataset:
                item = normalize_record(dict(record))
                if item is not None:
                    normalized.append(item)
                if len(normalized) >= MEMIT_SIZE:
                    break
            if len(normalized) >= MEMIT_SIZE:
                save_json(normalized, MEMIT_DATA_PATH)
                return normalized, f"huggingface: {dataset_name}/{config_name}:{split_name}"
        except Exception:
            continue

    if CUSTOM_DATA_PATH.exists():
        fallback = load_json(CUSTOM_DATA_PATH)
        for item in fallback:
            item.setdefault("subject", build_subject(item["prompt"]))
        return fallback, f"fallback custom dataset: {CUSTOM_DATA_PATH}"

    raise RuntimeError(
        "Could not prepare MEMIT dataset. Please ensure network access to Hugging Face "
        "or manually place a 500-sample file at data/memit_500.json."
    )


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


def build_locality_inputs(samples: list[dict]) -> dict:
    return {
        "neighborhood": {
            "prompt": [item["locality_prompt"] for item in samples],
            "ground_truth": [item["locality_ground_truth"] for item in samples],
        }
    }


def build_results(samples: list[dict], metrics: list[dict], dataset_source: str, elapsed_seconds: float, peak_memory_mb: float) -> dict:
    items = []
    for idx, (item, raw_metric) in enumerate(zip(samples, metrics), start=1):
        items.append(
            {
                "id": idx,
                "prompt": item["prompt"],
                "target_new": item["target_new"],
                "ground_truth": item["ground_truth"],
                "rephrase_prompt": item["rephrase_prompt"],
                "locality_prompt": item["locality_prompt"],
                "locality_ground_truth": item["locality_ground_truth"],
                "subject": item["subject"],
                "metrics": extract_core_metrics(raw_metric),
                "raw_metrics": raw_metric,
            }
        )

    return {
        "meta": {
            "dataset_source": dataset_source,
            "num_samples": len(samples),
            "elapsed_seconds": round(elapsed_seconds, 4),
            "peak_memory_mb": round(peak_memory_mb, 2),
        },
        "results": items,
    }


def print_case_summaries(results: list[dict]):
    print("\n===== Per-case MEMIT Metrics =====")
    for result in results[:10]:
        post_metrics = result["metrics"]["post"]
        print(f"[{result['id']}/{len(results)}] {result['prompt']} -> {result['target_new']}")
        print(f"  rewrite_acc: {post_metrics['rewrite_acc']}")
        print(f"  rephrase_acc: {post_metrics['rephrase_acc']}")
        print(f"  locality_acc: {post_metrics['locality_acc']}")
    if len(results) > 10:
        print(f"... omitted {len(results) - 10} additional samples in console output")


def main():
    if not HPARAMS_PATH.exists():
        raise FileNotFoundError(f"MEMIT hparams file not found: {HPARAMS_PATH}")

    device = setup_accelerator()

    samples, dataset_source = prepare_memit_dataset()
    prompts = [item["prompt"] for item in samples]
    target_new = [item["target_new"] for item in samples]
    ground_truth = [item["ground_truth"] for item in samples]
    rephrase_prompts = [item["rephrase_prompt"] for item in samples]
    subjects = [item.get("subject") or build_subject(item["prompt"]) for item in samples]
    locality_inputs = build_locality_inputs(samples)

    print("===== MEMIT Batch Editing =====")
    print(f"Device: {device}")
    print(f"Samples: {len(samples)}")
    print(f"Dataset source: {dataset_source}")
    print(f"Hparams: {HPARAMS_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    hparams = MEMITHyperParams.from_hparams(str(HPARAMS_PATH))
    # EasyEdit uses hparams.device as an integer index in f'cuda:{hparams.device}'
    if torch.cuda.is_available():
        hparams.device = 0
    editor = BaseEditor.from_hparams(hparams)

    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    metrics, _, _ = editor.batch_edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        subject=subjects,
        sequential_edit=False,
        verbose=False,
    )
    elapsed_seconds = time.perf_counter() - start_time

    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    output = build_results(samples, metrics, dataset_source, elapsed_seconds, peak_memory_mb)
    save_json(output, OUTPUT_PATH)
    print_case_summaries(output["results"])

    print("\n===== MEMIT Batch Editing Finished =====")
    print(f"Elapsed seconds: {elapsed_seconds:.4f}")
    print(f"Peak GPU memory (MB): {peak_memory_mb:.2f}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
