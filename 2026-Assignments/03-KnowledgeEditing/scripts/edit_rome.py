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
    # 去重
    uniq = []
    seen = set()
    for c in cands:
        nc = norm_text(c)
        if nc and nc not in seen:
            seen.add(nc)
            uniq.append(c)
    return uniq


def contains_any(text: str, candidates: List[str]) -> bool:
    t = norm_text(text)
    for c in candidates:
        if norm_text(c) in t:
            return True
    return False


def strict_hit_main(pred: str, candidates: List[str]) -> bool:
    """
    更稳健的主判定：
    1) 先看首句命中
    2) 再看全文命中（兜底）
    """
    return contains_any(first_sentence(pred), candidates) or contains_any(pred, candidates)


def has_ascii_alnum(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text or ""))


def align_target_with_prompt(prompt: str, target: str) -> str:
    if not target:
        return target
    if prompt and prompt[-1].isspace():
        return target.lstrip()
    if has_ascii_alnum(prompt) and not target.startswith(" "):
        return " " + target
    return target


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    if not prompt:
        return ""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    out_ids = model.generate(**inputs, **kwargs)
    txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if txt.startswith(prompt):
        return txt[len(prompt):].strip()
    return txt.strip()


@torch.no_grad()
def target_seq_prob(model, tokenizer, prompt: str, target: str) -> float:
    if not prompt or not target:
        return 0.0
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    inp = prompt_ids
    prob = 1.0
    for i in range(target_ids.shape[1]):
        logits = model(input_ids=inp).logits[:, -1, :]
        tid = target_ids[:, i]
        p = torch.softmax(logits, dim=-1).gather(-1, tid.unsqueeze(-1)).squeeze(-1)
        prob *= float(p.item())
        inp = torch.cat([inp, tid.unsqueeze(0)], dim=1)
    return prob


def get_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }.get(name, torch.float16)


def load_editor_from_hparams(hparams_path: str):
    # 关键：BaseEditor.from_hparams 要吃 HyperParams 对象
    from easyeditor import BaseEditor, ROMEHyperParams
    hp = ROMEHyperParams.from_hparams(hparams_path)
    return BaseEditor.from_hparams(hp)


def call_edit_single(editor, rec: Dict[str, Any]):
    prompt = rec["prompt"]
    target_new = rec["target_new"]
    subject = rec.get("subject", None)
    rephrase = rec.get("rephrase_prompt", None)

    locality_inputs = None
    if rec.get("locality_prompt") and rec.get("locality_ground_truth"):
        locality_inputs = {
            "locality": {
                "prompt": rec["locality_prompt"],
                "ground_truth": rec["locality_ground_truth"]
            }
        }

    # 常见签名
    try:
        kwargs = {
            "prompts": [prompt],
            "target_new": [target_new],
            "keep_original_weight": False
        }
        if subject:
            kwargs["subject"] = [subject]
        if rephrase:
            kwargs["rephrase_prompts"] = [rephrase]
        if locality_inputs is not None:
            kwargs["locality_inputs"] = locality_inputs
        return editor.edit(sequential_edit=True, **kwargs)
    except TypeError:
        # 退化签名
        kwargs = {
            "prompts": [prompt],
            "target_new": [target_new]
        }
        if subject:
            kwargs["subject"] = [subject]
        if locality_inputs is not None:
            kwargs["locality_inputs"] = locality_inputs
        return editor.edit(sequential_edit=True, **kwargs)


def summarize_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(results)
    es = sum(1 for x in results if x["hit_main_target_new"]) / n * 100 if n else 0.0
    ps_total = sum(1 for x in results if x.get("rephrase_prompt"))
    ps_hit = sum(1 for x in results if x.get("rephrase_prompt") and x["hit_rephrase_target_new"])
    ps = ps_hit / ps_total * 100 if ps_total else 0.0

    ns_total = sum(1 for x in results if x.get("locality_prompt"))
    ns_hit = sum(1 for x in results if x.get("locality_prompt") and x["hit_locality_ground_truth"])
    ns = ns_hit / ns_total * 100 if ns_total else 0.0

    return {"ES": es, "PS": ps, "NS": ns}


def run(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    dtype = get_dtype(args.torch_dtype)
    device_map = args.device_map
    if device_map and device_map.lower() in {"none", "null"}:
        device_map = None
    all_results = []
    t_all = time.time()

    for idx, rec in enumerate(data, start=1):
        print(f"\n[INFO] ROME editing sample {idx}/{len(data)}")

        target_new_eval = align_target_with_prompt(rec["prompt"], rec["target_new"])

        # 每条重置
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
        if device_map is None:
            base_model = base_model.to("cuda" if torch.cuda.is_available() else "cpu")
        base_model.eval()

        # PRE
        pre_main = generate_text(base_model, tokenizer, rec["prompt"], args.max_new_tokens, args.do_sample, args.temperature)
        pre_reph = generate_text(base_model, tokenizer, rec.get("rephrase_prompt", ""), args.max_new_tokens, args.do_sample, args.temperature) if rec.get("rephrase_prompt") else ""
        pre_loca = generate_text(base_model, tokenizer, rec.get("locality_prompt", ""), args.max_new_tokens, args.do_sample, args.temperature) if rec.get("locality_prompt") else ""
        pre_prob = target_seq_prob(base_model, tokenizer, rec["prompt"], target_new_eval)

        print(f"[PRE] prompt      : {rec['prompt']}")
        print(f"[PRE] target      : {rec['target_new']!r}")
        if target_new_eval != rec["target_new"]:
            print(f"[PRE] target_used : {target_new_eval!r}")
        print(f"[PRE] pred_main   : {pre_main!r}")
        print(f"[PRE] target_prob : {pre_prob:.8f}")

        # EDIT
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        editor = load_editor_from_hparams(args.hparams)
        t_edit = time.time()
        metrics, edited_model, extra = call_edit_single(editor, rec)
        edit_time = time.time() - t_edit

        # POST
        post_main = generate_text(edited_model, tokenizer, rec["prompt"], args.max_new_tokens, args.do_sample, args.temperature)
        post_reph = generate_text(edited_model, tokenizer, rec.get("rephrase_prompt", ""), args.max_new_tokens, args.do_sample, args.temperature) if rec.get("rephrase_prompt") else ""
        post_loca = generate_text(edited_model, tokenizer, rec.get("locality_prompt", ""), args.max_new_tokens, args.do_sample, args.temperature) if rec.get("locality_prompt") else ""
        post_prob = target_seq_prob(edited_model, tokenizer, rec["prompt"], target_new_eval)

        target_candidates = build_candidates(rec["target_new"], rec.get("target_aliases", []))

        es_hit = strict_hit_main(post_main, target_candidates)
        ps_hit = strict_hit_main(post_reph, target_candidates) if rec.get("rephrase_prompt") else None
        ns_hit = contains_any(post_loca, [rec.get("locality_ground_truth", "")]) if rec.get("locality_prompt") else None

        print(f"[POST] pred_main   : {post_main!r}")
        print(f"[POST] target_prob : {post_prob:.8f}")
        print(f"[POST] hit(ES/PS/NS): {es_hit}/{ps_hit}/{ns_hit}")

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

        row = dict(rec)
        row.update({
            "target_new_stripped": rec["target_new"].strip(),
            "target_new_used": target_new_eval,
            "target_candidates": target_candidates,

            "pre_pred_main": pre_main,
            "pre_pred_rephrase": pre_reph,
            "pre_pred_locality": pre_loca,
            "pre_target_prob": pre_prob,

            "pred_main": post_main,
            "pred_rephrase": post_reph,
            "pred_locality": post_loca,
            "post_target_prob": post_prob,

            "hit_main_target_new": es_hit,
            "hit_rephrase_target_new": ps_hit,
            "hit_locality_ground_truth": ns_hit,

            "edit_time_sec": edit_time,
            "peak_mem_mb": peak_mem,
            "editor_metrics": metrics
        })
        all_results.append(row)

        del edited_model, base_model, editor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_metrics(all_results)
    payload = {
        "meta": {
            "method": "ROME",
            "model_name_or_path": args.model_name_or_path,
            "hparams_path": args.hparams,
            "data_path": args.data,
            "num_samples": len(all_results),
            "total_time_sec": time.time() - t_all
        },
        "summary": summary,
        "results": all_results
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Saved:", args.out)
    print(f"[SUMMARY] ES={summary['ES']:.2f}% | PS={summary['PS']:.2f}% | NS={summary['NS']:.2f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="hf_models/Qwen2.5-0.5B")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--hparams", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--max_new_tokens", type=int, default=16)  # 默认更短，减少跑偏
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)

    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)