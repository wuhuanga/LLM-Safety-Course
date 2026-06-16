import argparse
import csv
import gc
import json
import logging
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from utils import (
    build_locality_inputs,
    ensure_easyedit_path,
    generate_text,
    load_json,
    patch_easyedit_nethook,
    redirect_cuda_to_available_device,
    set_seed,
)


ZH_PROMPTS = {
    "custom_001": "Twitter（X）的现任 CEO 是",
    "custom_002": "OpenAI 的 CEO 是",
    "custom_003": "微软的 CEO 是",
    "custom_004": "谷歌的 CEO 是",
    "custom_005": "Meta 的 CEO 是",
    "custom_006": "英伟达的 CEO 是",
    "custom_007": "亚马逊的 CEO 是",
    "custom_008": "YouTube 的 CEO 是",
    "custom_009": "IBM 的 CEO 是",
    "custom_010": "迪士尼的 CEO 是",
}


TARGET_ALIASES = {
    "custom_001": ["Linda Yaccarino", "琳达亚卡里诺", "琳达·亚卡里诺"],
    "custom_002": ["Sam Altman", "萨姆奥特曼", "山姆奥特曼", "萨姆·奥特曼", "山姆·奥特曼"],
    "custom_003": ["Satya Nadella", "萨提亚纳德拉", "萨蒂亚纳德拉", "纳德拉"],
    "custom_004": ["Sundar Pichai", "桑达尔皮查伊", "桑德尔皮查伊", "皮查伊"],
    "custom_005": ["Mark Zuckerberg", "马克扎克伯格", "扎克伯格"],
    "custom_006": ["Jensen Huang", "Jen-Hsun Huang", "黄仁勋"],
    "custom_007": ["Andy Jassy", "安迪贾西", "安迪·贾西"],
    "custom_008": ["Neal Mohan", "尼尔莫汉", "尼尔·莫汉"],
    "custom_009": ["Arvind Krishna", "阿尔温德克里希纳", "阿文德克里希纳", "克里希纳"],
    "custom_010": ["Bob Iger", "鲍勃艾格", "鲍勃·艾格"],
}


def normalize_for_match(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    kept = []
    for char in normalized:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        kept.append(char)
    return "".join(kept)


def contains_any_answer(output: str, aliases: List[str]) -> bool:
    normalized_output = normalize_for_match(output)
    return any(normalize_for_match(alias) in normalized_output for alias in aliases if alias)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def write_json(payload: Dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "subject",
        "zh_prompt",
        "target_new",
        "baseline_hit",
        "cross_lingual_hit",
        "edit_time_seconds",
        "total_time_seconds",
        "peak_memory_gb",
        "baseline_output",
        "edited_output",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def load_editor(hparams_path: str, easyedit_root: str):
    ensure_easyedit_path(easyedit_root)
    active_device = redirect_cuda_to_available_device()
    if active_device != "cuda":
        print(f"CUDA is unavailable; redirecting EasyEdit CUDA calls to {active_device}.")

    from easyeditor import BaseEditor, ROMEHyperParams

    patch_easyedit_nethook()
    hparams = ROMEHyperParams.from_hparams(hparams_path)
    logging.getLogger("easyeditor.editors.editor").handlers.clear()
    return BaseEditor.from_hparams(hparams)


def evaluate_cross_lingual_case(
    record: Dict[str, Any],
    hparams_path: str,
    easyedit_root: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    case_id = record["case_id"]
    zh_prompt = ZH_PROMPTS.get(case_id, f"{record['subject']} 的答案是")
    aliases = TARGET_ALIASES.get(case_id, [record["target_new"]])
    if record["target_new"] not in aliases:
        aliases = [record["target_new"], *aliases]

    editor = load_editor(hparams_path, easyedit_root)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()

    baseline_output = generate_text(editor.model, editor.tok, zh_prompt, max_new_tokens=max_new_tokens)
    baseline_hit = contains_any_answer(baseline_output, aliases)

    edit_start = time.perf_counter()
    metrics, edited_model, _ = editor.edit(
        prompts=[record["prompt"]],
        ground_truth=[record["ground_truth"]],
        target_new=[record["target_new"]],
        rephrase_prompts=[zh_prompt],
        locality_inputs=build_locality_inputs([record]),
        subject=[record["subject"]],
        sequential_edit=True,
        verbose=False,
    )
    edit_time = time.perf_counter() - edit_start

    edited_model.eval()
    edited_output = generate_text(edited_model, editor.tok, zh_prompt, max_new_tokens=max_new_tokens)
    cross_lingual_hit = contains_any_answer(edited_output, aliases)

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "case_id": case_id,
        "subject": record["subject"],
        "english_edit_prompt": record["prompt"],
        "zh_prompt": zh_prompt,
        "target_new": record["target_new"],
        "target_aliases": aliases,
        "baseline_output": baseline_output,
        "edited_output": edited_output,
        "baseline_hit": baseline_hit,
        "cross_lingual_hit": cross_lingual_hit,
        "edit_time_seconds": edit_time,
        "total_time_seconds": time.perf_counter() - total_start,
        "peak_memory_gb": peak_memory_gb,
        "metrics": metrics,
    }

    del editor
    del edited_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bonus: edit English facts and test Chinese cross-lingual generalization."
    )
    parser.add_argument("--data", default="data/custom_10.json")
    parser.add_argument("--hparams", default="hparams/ROME/qwen2.5-0.5b.yaml")
    parser.add_argument("--out", default="results/bonus_cross_lingual.json")
    parser.add_argument("--csv-out", default="results/bonus_cross_lingual.csv")
    parser.add_argument("--easyedit-root", default="external/EasyEdit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    records: List[Dict[str, Any]] = load_json(args.data)
    if args.limit:
        records = records[: args.limit]

    rows = []
    for record in tqdm(records, desc="cross-lingual ROME"):
        rows.append(
            evaluate_cross_lingual_case(
                record,
                args.hparams,
                args.easyedit_root,
                args.max_new_tokens,
            )
        )

    baseline_hits = sum(1 for row in rows if row["baseline_hit"])
    cross_lingual_hits = sum(1 for row in rows if row["cross_lingual_hit"])
    improved_hits = sum(1 for row in rows if row["cross_lingual_hit"] and not row["baseline_hit"])

    summary = {
        "algorithm": "ROME",
        "bonus_task": "cross_lingual_generalization",
        "description": "English fact editing, Chinese prompt testing.",
        "hparams": args.hparams,
        "num_records": len(rows),
        "baseline_accuracy": baseline_hits / len(rows) if rows else None,
        "cross_lingual_accuracy": cross_lingual_hits / len(rows) if rows else None,
        "improved_accuracy": improved_hits / len(rows) if rows else None,
        "results": rows,
    }

    write_json(summary, args.out)
    write_csv(rows, args.csv_out)

    print("\nCross-lingual summary")
    print(f"records: {len(rows)}")
    print(f"baseline_accuracy: {summary['baseline_accuracy']:.4f}" if rows else "baseline_accuracy: n/a")
    print(
        f"cross_lingual_accuracy: {summary['cross_lingual_accuracy']:.4f}"
        if rows
        else "cross_lingual_accuracy: n/a"
    )
    print(f"improved_accuracy: {summary['improved_accuracy']:.4f}" if rows else "improved_accuracy: n/a")
    print(f"json: {args.out}")
    print(f"csv: {args.csv_out}")


if __name__ == "__main__":
    main()
