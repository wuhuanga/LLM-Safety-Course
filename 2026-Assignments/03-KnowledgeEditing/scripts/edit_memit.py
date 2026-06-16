#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Text / metric utils
# -------------------------
def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def first_sentence(s: str) -> str:
    if not s:
        return ""
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", s.strip(), maxsplit=1)
    return parts[0].strip()


def build_candidates(target_new: str, aliases: Optional[List[str]] = None) -> List[str]:
    cands = [target_new, target_new.strip()]
    if aliases:
        cands.extend(aliases)
    out, seen = [], set()
    for c in cands:
        nc = norm_text(c)
        if nc and nc not in seen:
            seen.add(nc)
            out.append(c)
    return out


def contains_any(text: str, candidates: List[str]) -> bool:
    t = norm_text(text)
    return any(norm_text(c) in t for c in candidates if c)


def hit_main(pred: str, candidates: List[str]) -> bool:
    # 首句优先 + 全文兜底
    return contains_any(first_sentence(pred), candidates) or contains_any(pred, candidates)


def summarize_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {"ES": 0.0, "PS": 0.0, "NS": 0.0}

    es = sum(1 for x in results if x["hit_main_target_new"]) / n * 100.0

    ps_total = sum(1 for x in results if x.get("rephrase_prompt"))
    ps_hit = sum(1 for x in results if x.get("rephrase_prompt") and x["hit_rephrase_target_new"])
    ps = (ps_hit / ps_total * 100.0) if ps_total else 0.0

    ns_total = sum(1 for x in results if x.get("locality_prompt"))
    ns_hit = sum(1 for x in results if x.get("locality_prompt") and x["hit_locality_ground_truth"])
    ns = (ns_hit / ns_total * 100.0) if ns_total else 0.0

    return {"ES": es, "PS": ps, "NS": ns}


def _mean_val(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, list):
        if not val:
            return None
        return float(sum(val) / len(val))
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def summarize_easyedit_metrics(editor_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not editor_metrics:
        return {"ES": 0.0, "PS": 0.0, "NS": 0.0}

    es_vals: List[float] = []
    ps_vals: List[float] = []
    ns_vals: List[float] = []

    for metric in editor_metrics:
        post = metric.get("post", {})
        es = _mean_val(post.get("rewrite_acc"))
        if es is not None:
            es_vals.append(es)

        ps = _mean_val(post.get("rephrase_acc"))
        if ps is not None:
            ps_vals.append(ps)

        locality = post.get("locality", {})
        if isinstance(locality, dict):
            for key, val in locality.items():
                if key.endswith("acc"):
                    ns = _mean_val(val)
                    if ns is not None:
                        ns_vals.append(ns)

    es = (sum(es_vals) / len(es_vals) * 100.0) if es_vals else 0.0
    ps = (sum(ps_vals) / len(ps_vals) * 100.0) if ps_vals else 0.0
    ns = (sum(ns_vals) / len(ns_vals) * 100.0) if ns_vals else 0.0

    return {"ES": es, "PS": ps, "NS": ns}


# -------------------------
# Generation
# -------------------------
@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 16, do_sample: bool = False, temperature: float = 0.7) -> str:
    if not prompt:
        return ""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    out_ids = model.generate(**inputs, **kwargs)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


def get_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }.get(name, torch.float16)


# -------------------------
# EasyEdit MEMIT
# -------------------------
def load_memit_editor(hparams_path: str):
    # 关键：先加载为 MEMITHyperParams 对象
    from easyeditor import BaseEditor, MEMITHyperParams
    hp = MEMITHyperParams.from_hparams(hparams_path)
    editor = BaseEditor.from_hparams(hp)
    return editor


def run_memit(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 截断到500（或用户指定）
    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"[INFO] Loaded samples: {len(data)}")

    # 模型加载（用于后续生成评测）
    dtype = get_dtype(args.torch_dtype)
    device_map = args.device_map
    if device_map and device_map.lower() in {"none", "null"}:
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    _base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if device_map is None:
        _base_model = _base_model.to("cuda" if torch.cuda.is_available() else "cpu")
    _base_model.eval()

    # 构造编辑字段
    prompts = [x["prompt"] for x in data]
    targets = [x["target_new"] for x in data]
    subjects = [x.get("subject", "") for x in data]

    rephrase_prompts = [x.get("rephrase_prompt", "") for x in data]
    has_rephrase = any(bool(x) for x in rephrase_prompts)

    locality_prompts = [x.get("locality_prompt", "") for x in data]
    locality_truths = [x.get("locality_ground_truth", "") for x in data]
    has_locality = all(bool(p) and bool(g) for p, g in zip(locality_prompts, locality_truths))

    locality_inputs = None
    if has_locality:
        locality_inputs = {
            "locality": {
                "prompt": locality_prompts,
                "ground_truth": locality_truths
            }
        }

    # 编辑
    editor = load_memit_editor(args.hparams)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    try:
        kwargs = {
            "prompts": prompts,
            "target_new": targets,
            "subject": subjects,
            "keep_original_weight": False
        }
        if has_rephrase:
            kwargs["rephrase_prompts"] = rephrase_prompts
        if locality_inputs is not None:
            kwargs["locality_inputs"] = locality_inputs

        metrics, edited_model, extra = editor.edit(**kwargs)
    except TypeError:
        kwargs = {
            "prompts": prompts,
            "target_new": targets
        }
        if locality_inputs is not None:
            kwargs["locality_inputs"] = locality_inputs
        metrics, edited_model, extra = editor.edit(**kwargs)

    edit_time_sec = time.time() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    print(f"[INFO] MEMIT batch edit done. time={edit_time_sec:.2f}s peak_mem={peak_mem_mb}MB")

    # 编辑后逐条评测
    results = []
    for i, rec in enumerate(data, start=1):
        pred_main = generate_text(
            edited_model, tokenizer, rec["prompt"],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature
        )
        pred_rephrase = generate_text(
            edited_model, tokenizer, rec.get("rephrase_prompt", ""),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature
        ) if rec.get("rephrase_prompt") else ""
        pred_locality = generate_text(
            edited_model, tokenizer, rec.get("locality_prompt", ""),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature
        ) if rec.get("locality_prompt") else ""

        cands = build_candidates(rec["target_new"], rec.get("target_aliases", []))
        es_hit = hit_main(pred_main, cands)
        ps_hit = hit_main(pred_rephrase, cands) if rec.get("rephrase_prompt") else None
        ns_hit = contains_any(pred_locality, [rec.get("locality_ground_truth", "")]) if rec.get("locality_prompt") else None

        row = dict(rec)
        row.update({
            "target_new_stripped": rec["target_new"].strip(),
            "target_candidates": cands,
            "pred_main": pred_main,
            "pred_rephrase": pred_rephrase,
            "pred_locality": pred_locality,
            "hit_main_target_new": es_hit,
            "hit_rephrase_target_new": ps_hit,
            "hit_locality_ground_truth": ns_hit
        })
        results.append(row)

        if i % 50 == 0 or i == len(data):
            print(f"[EVAL] {i}/{len(data)}")

    summary_gen = summarize_metrics(results)
    summary = summarize_easyedit_metrics(metrics) if metrics else summary_gen

    payload = {
        "meta": {
            "method": "MEMIT",
            "model_name_or_path": args.model_name_or_path,
            "hparams_path": args.hparams,
            "data_path": args.data,
            "num_samples": len(results),
            "edit_time_sec": edit_time_sec,
            "peak_mem_mb": peak_mem_mb
        },
        "summary": summary,
        "summary_gen": summary_gen,
        "editor_metrics": metrics,
        "results": results
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved: {args.out}")
    print(f"[SUMMARY] ES={summary['ES']:.2f}% | PS={summary['PS']:.2f}% | NS={summary['NS']:.2f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Batch Knowledge Editing with MEMIT")
    p.add_argument("--model_name_or_path", type=str, default="hf_models/Qwen2.5-0.5B")
    p.add_argument("--data", type=str, default="/home/dx/LLM-Safety-Course/2026-Assignments/03-KnowledgeEditing/data/counterfact_500.json")
    p.add_argument("--hparams", type=str, required=True, help="e.g. hparams/memit_qwen2.5-0.5b.yaml")
    p.add_argument("--out", type=str, default="results/memit_results_counterfact500.json")

    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)

    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_memit(args)