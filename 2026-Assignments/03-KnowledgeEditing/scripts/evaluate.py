#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Dict, Any, List, Optional


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'“”‘’]", "", s)
    return s


def contains_answer(pred: str, answer: str) -> bool:
    if answer is None or answer == "":
        return False
    return normalize_text(answer) in normalize_text(pred or "")


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


def compute_easyedit_metrics(editor_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not editor_metrics:
        return {"num_samples": 0, "ES": 0.0, "PS": 0.0, "NS": 0.0}

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

    ES = 100.0 * (sum(es_vals) / len(es_vals)) if es_vals else 0.0
    PS = 100.0 * (sum(ps_vals) / len(ps_vals)) if ps_vals else 0.0
    NS = 100.0 * (sum(ns_vals) / len(ns_vals)) if ns_vals else 0.0

    return {
        "num_samples": len(editor_metrics),
        "ES": ES,
        "PS": PS,
        "NS": NS,
        "details": {
            "es_total": len(es_vals),
            "ps_total": len(ps_vals),
            "ns_total": len(ns_vals)
        }
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"num_samples": 0, "ES": 0.0, "PS": 0.0, "NS": 0.0}

    es_hits = 0
    ps_hits = 0
    ns_hits = 0

    ps_total = 0
    ns_total = 0

    for x in results:
        pred_main = x.get("pred_main", "")
        pred_rephrase = x.get("pred_rephrase", "")
        pred_locality = x.get("pred_locality", "")

        target_new = x.get("target_new", "")
        locality_gt = x.get("locality_ground_truth", "")

        if contains_answer(pred_main, target_new):
            es_hits += 1

        if x.get("rephrase_prompt", ""):
            ps_total += 1
            if contains_answer(pred_rephrase, target_new):
                ps_hits += 1

        if x.get("locality_prompt", "") and locality_gt:
            ns_total += 1
            if contains_answer(pred_locality, locality_gt):
                ns_hits += 1

    ES = 100.0 * es_hits / n
    PS = 100.0 * ps_hits / ps_total if ps_total > 0 else 0.0
    NS = 100.0 * ns_hits / ns_total if ns_total > 0 else 0.0

    return {
        "num_samples": n,
        "ES": ES,
        "PS": PS,
        "NS": NS,
        "details": {
            "es_hits": es_hits,
            "es_total": n,
            "ps_hits": ps_hits,
            "ps_total": ps_total,
            "ns_hits": ns_hits,
            "ns_total": ns_total
        }
    }


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        # 兼容纯list格式
        return {"meta": {}, "results": payload}
    if "results" not in payload:
        raise ValueError(f"`results` field not found in {path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Compute ES/PS/NS from result json.")
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more result json paths, e.g. outputs/rome_results.json outputs/memit_results.json"
    )
    parser.add_argument("--save_path", type=str, default="outputs/metrics_summary.json")
    args = parser.parse_args()

    all_summary = {}

    for p in args.input_paths:
        payload = load_results(p)
        if "editor_metrics" in payload:
            metrics = compute_easyedit_metrics(payload["editor_metrics"])
        else:
            metrics = compute_metrics(payload["results"])
        meta = payload.get("meta", {})
        all_summary[p] = {"meta": meta, "metrics": metrics}
        print(f"\n== {p} ==")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved summary to {args.save_path}")


if __name__ == "__main__":
    main()