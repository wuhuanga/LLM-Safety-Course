"""Task 2: ROME单条编辑 — 逐条编辑10个事实，每次编辑后重置模型权重。"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

from easyeditor import BaseEditor, ROMEHyperParams

# 从JSON文件加载自定义测试数据集
FACTS = json.load(open(os.path.join(ROOT, "data", "custom_facts.json"), "r", encoding="utf-8"))

# 加载一次模型，循环中复用，keep_original_weight确保每次编辑前重置权重
print("Loading Qwen2.5-0.5B model for ROME editing ...")
hparams = ROMEHyperParams.from_hparams(os.path.join(ROOT, "hparams", "ROME", "qwen2.5-0.5b"))
hparams.model_name = os.path.join(ROOT, "hugging_cache", "Qwen2.5-0.5B")
editor = BaseEditor.from_hparams(hparams)

# 逐条编辑，收集ES/PS/NS指标
print(f"\n{'ID':<3} {'Prompt':<42} {'ES(rewrite)':<12} {'PS(rephrase)':<14} {'NS(locality)':<16}")
print("-" * 100)

all_metrics = []

for i, fact in enumerate(FACTS):
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
        keep_original_weight=True,  # 编辑后立即还原权重，保证各条编辑相互独立
        verbose=False,
    )

    # metrics 是 list[dict]: [{'pre': {...}, 'post': {'rewrite_acc': [...], 'rephrase_acc': [...], 'locality': {...}}}]
    post = metrics[0]["post"]
    es = post["rewrite_acc"][0] if post.get("rewrite_acc") else 0
    ps = post["rephrase_acc"][0] if post.get("rephrase_acc") else 0
    ns_dict = post.get("locality", {})
    ns = list(ns_dict.values())[0][0] if ns_dict else 0

    all_metrics.append({"es": es, "ps": ps, "ns": ns})
    print(f"{i+1:<3} {fact['prompt'][:40]:<42} {str(es):<12} {str(ps):<14} {str(ns):<16}")

# 汇总平均值
avg_es = sum(m["es"] for m in all_metrics if m["es"] is not None) / len(all_metrics)
avg_ps = sum(m["ps"] for m in all_metrics if m["ps"] is not None) / len(all_metrics)
avg_ns = sum(m["ns"] for m in all_metrics if m["ns"] is not None) / len(all_metrics)

print("-" * 100)
print(f"Average  ES = {avg_es:.2%}  |  PS = {avg_ps:.2%}  |  NS = {avg_ns:.2%}")
print("\nNote: ES=编辑成功率, PS=泛化性(同义改写), NS=局部性(无关事实不受影响)")
