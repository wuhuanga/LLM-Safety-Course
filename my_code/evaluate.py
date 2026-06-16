"""Task 4: 综合评估 — 计算ES(编辑成功率)、PS(泛化性)、NS(局部性)三个核心指标。"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time

DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams
from easyeditor import ZsreDataset, CounterFactDataset

EVAL_MODE = "ROME"  # 切换为 "MEMIT" 即可评估批量编辑场景

# 从JSON文件加载自定义数据（ROME模式下使用）
ROME_FACTS = json.load(open(os.path.join(ROOT, "data", "custom_facts.json"), "r", encoding="utf-8"))


def evaluate_rome():
    """ROME逐条编辑并评估10条自定义事实。"""
    print("=" * 70)
    print("  Comprehensive Evaluation - ROME (10 custom facts)")
    print("=" * 70)

    hparams = ROMEHyperParams.from_hparams(os.path.join(ROOT, "hparams", "ROME", "qwen2.5-0.5b"))
    hparams.model_name = os.path.join(ROOT, "hugging_cache", "Qwen2.5-0.5B")
    editor = BaseEditor.from_hparams(hparams)
    tok = editor.tok

    es_list, ps_list, ns_list = [], [], []

    for i, fact in enumerate(ROME_FACTS):
        metrics, _, _ = editor.edit(
            prompts=fact["prompt"],
            target_new=fact["target_new"],
            ground_truth=fact["ground_truth"],
            subject=fact["subject"],
            rephrase_prompts=fact["rephrase_prompt"],
            locality_inputs={
                "neighborhood": {
                    "prompt": fact["locality_prompt"],
                    "ground_truth": fact["locality_ground_truth"],
                }
            },
            keep_original_weight=True,
            verbose=False,
        )

        post = metrics[0]["post"]
        es = post["rewrite_acc"][0] if post.get("rewrite_acc") else 0
        ps = post["rephrase_acc"][0] if post.get("rephrase_acc") else 0
        ns_dict = post.get("locality", {})
        ns = list(ns_dict.values())[0][0] if ns_dict else 0

        es_list.append(es if es is not None else 0)
        ps_list.append(ps if ps is not None else 0)
        ns_list.append(ns if ns is not None else 0)

    # 逐条明细
    print()
    for i, fact in enumerate(ROME_FACTS):
        es_str = "YES" if es_list[i] else "NO"
        ps_str = "YES" if ps_list[i] else "NO"
        ns_str = "YES" if ns_list[i] else "NO"
        print(f"[{i+1}] 问题: {fact['prompt']}")
        print(f"    旧答案: {fact['ground_truth']}  新答案: {fact['target_new']}")
        print(f"    ES: {es_str}  PS: {ps_str}  NS: {ns_str}")
        print()

    # 汇总
    avg_es = sum(es_list) / len(es_list)
    avg_ps = sum(ps_list) / len(ps_list)
    avg_ns = sum(ns_list) / len(ns_list)

    print("=" * 60)
    print(f"  ES (Efficacy)     : {avg_es:.0%}")
    print(f"  PS (Paraphrase)   : {avg_ps:.0%}")
    print(f"  NS (Neighborhood) : {avg_ns:.0%}")
    print(f"  ROME Summary: ES={avg_es:.0%}, PS={avg_ps:.0%}, NS={avg_ns:.0%}")


def evaluate_memit():
    """MEMIT批量编辑并评估500条数据（来自ZsRE或CounterFact）。"""
    print("=" * 70)
    print("  Comprehensive Evaluation - MEMIT (500 records)")
    print("=" * 70)

    N = 500
    ZSRE_PATH = os.path.join(ROOT, "data", "zsre_mend_eval.json")
    CF_PATH = os.path.join(ROOT, "data", "counterfact.json")

    if os.path.exists(ZSRE_PATH):
        ds = ZsreDataset(ZSRE_PATH, size=N)
        prompts = [d["prompt"] for d in ds]
        target_new = [d["target_new"] for d in ds]
        ground_truth = [d["ground_truth"] for d in ds]
        rephrase_prompts = [d["rephrase_prompt"] for d in ds]
        locality_prompts = [d["locality_prompt"] for d in ds]
        locality_answers = [d["locality_ground_truth"] for d in ds]
    elif os.path.exists(CF_PATH):
        ds = CounterFactDataset(CF_PATH, size=N)
        # CounterFact 嵌套格式: {'requested_rewrite': {'prompt': ..., 'subject': ..., 'target_new': {'str': ...}, ...}}
        prompts = [d["requested_rewrite"]["prompt"].format(d["requested_rewrite"]["subject"]) for d in ds]
        target_new = [d["requested_rewrite"]["target_new"]["str"] if isinstance(d["requested_rewrite"]["target_new"], dict) else d["requested_rewrite"]["target_new"] for d in ds]
        ground_truth = [d["requested_rewrite"]["target_true"]["str"] if isinstance(d["requested_rewrite"]["target_true"], dict) else d["requested_rewrite"]["target_true"] for d in ds]
        rephrase_prompts = [d["requested_rewrite"]["rephrase_prompt"].format(d["requested_rewrite"]["subject"]) if d["requested_rewrite"].get("rephrase_prompt") else "" for d in ds]
        locality_prompts = [d["requested_rewrite"]["locality_prompt"].format(d["requested_rewrite"]["subject"]) if d["requested_rewrite"].get("locality_prompt") else "" for d in ds]
        locality_answers = [d["requested_rewrite"]["locality_ground_truth"]["str"] if isinstance(d["requested_rewrite"].get("locality_ground_truth"), dict) else d["requested_rewrite"].get("locality_ground_truth", "") for d in ds]
    else:
        print("ERROR: Neither ZsRE nor CounterFact dataset found in ./data/")
        return

    hparams = MEMITHyperParams.from_hparams(os.path.join(ROOT, "hparams", "MEMIT", "qwen2.5-0.5b"))
    hparams.model_name = os.path.join(ROOT, "hugging_cache", "Qwen2.5-0.5B")
    editor = BaseEditor.from_hparams(hparams)

    t0 = time.time()

    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs={
            "neighborhood": {
                "prompt": locality_prompts,
                "ground_truth": locality_answers,
            }
        },
        keep_original_weight=True,
    )

    elapsed = time.time() - t0
    if DEVICE == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_mem = 0

    # 提取指标 — batch_edit 返回的 metrics 是 list[dict]
    rewrite_accs = [m["post"]["rewrite_acc"][0] for m in metrics if m["post"].get("rewrite_acc")]
    rephrase_accs = [m["post"]["rephrase_acc"][0] for m in metrics if m["post"].get("rephrase_acc")]
    loc_dicts = [m["post"].get("locality", {}) for m in metrics]
    loc_accs = []
    for ld in loc_dicts:
        for v in ld.values():
            if v:
                loc_accs.append(v[0])

    avg_es = sum(rewrite_accs) / len(rewrite_accs) if rewrite_accs else 0
    avg_ps = sum(rephrase_accs) / len(rephrase_accs) if rephrase_accs else 0
    avg_ns = sum(loc_accs) / len(loc_accs) if loc_accs else 0

    print(f"\n  Records evaluated: {len(metrics)}")
    print(f"  Device            : {DEVICE.upper()}")
    print(f"  ES (Efficacy)     : {avg_es:.2%}")
    print(f"  PS (Paraphrase)   : {avg_ps:.2%}")
    print(f"  NS (Neighborhood) : {avg_ns:.2%}")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    if DEVICE == "cuda":
        print(f"  Peak GPU memory   : {peak_mem:.2f} GiB")
    else:
        print(f"  Peak GPU memory   : N/A (CPU mode)")


# 根据EVAL_MODE选择评估场景
if __name__ == "__main__":
    print(f"Running comprehensive evaluation - Mode: {EVAL_MODE}\n")
    if EVAL_MODE == "ROME":
        evaluate_rome()
    elif EVAL_MODE == "MEMIT":
        evaluate_memit()
    else:
        print(f"Unknown mode: {EVAL_MODE}. Please set EVAL_MODE to 'ROME' or 'MEMIT'.")
