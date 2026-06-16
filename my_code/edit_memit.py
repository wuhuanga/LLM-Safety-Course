"""Task 3: MEMIT批量编辑 — 一次性注入500条知识，记录显存和耗时。"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time

# GPU可用则用GPU，否则降级为CPU patch
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

from easyeditor import BaseEditor, MEMITHyperParams
from easyeditor import ZsreDataset, CounterFactDataset

# 参数配置：编辑500条，数据源选zsre或counterfact
BATCH_SIZE = 500
DATA_SOURCE = "counterfact"
ZSRE_PATH = os.path.join(ROOT, "data", "zsre_mend_eval.json")
CF_PATH = os.path.join(ROOT, "data", "counterfact.json")

# 加载数据集，按数据源选择加载方式
print(f"Loading {BATCH_SIZE} records from {DATA_SOURCE.upper()} dataset ...")
if DATA_SOURCE == "zsre":
    if not os.path.exists(ZSRE_PATH):
        print(f"ERROR: ZsRE data not found at '{ZSRE_PATH}'.")
        print("Please download zsre_mend_eval.json to ./data/ (see EasyEdit repo).")
        sys.exit(1)
    ds = ZsreDataset(ZSRE_PATH, size=BATCH_SIZE)
    prompts = [d["prompt"] for d in ds]
    target_new = [d["target_new"] for d in ds]
    ground_truth = [d["ground_truth"] for d in ds]
    rephrase_prompts = [d["rephrase_prompt"] for d in ds]
    locality_prompts = [d["locality_prompt"] for d in ds]
    locality_answers = [d["locality_ground_truth"] for d in ds]
    subjects = None
else:
    if not os.path.exists(CF_PATH):
        print(f"ERROR: CounterFact data not found at '{CF_PATH}'.")
        print("Please download counterfact.json to ./data/ (see EasyEdit repo).")
        sys.exit(1)
    ds = CounterFactDataset(CF_PATH, size=BATCH_SIZE)
    prompts = [d["requested_rewrite"]["prompt"].format(d["requested_rewrite"]["subject"]) for d in ds]
    target_new = [d["requested_rewrite"]["target_new"]["str"] if isinstance(d["requested_rewrite"]["target_new"], dict) else d["requested_rewrite"]["target_new"] for d in ds]
    ground_truth = [d["requested_rewrite"]["target_true"]["str"] if isinstance(d["requested_rewrite"]["target_true"], dict) else d["requested_rewrite"]["target_true"] for d in ds]
    # CounterFact: 取第一条 paraphrase 和 neighborhood 作为评估数据
    rephrase_prompts = [d["paraphrase_prompts"][0] if d.get("paraphrase_prompts") else "" for d in ds]
    locality_prompts = [d["neighborhood_prompts"][0] if d.get("neighborhood_prompts") else "" for d in ds]
    locality_answers = [""] * len(ds)  # CounterFact 无显式 locality_ground_truth，框架用编辑前输出
    subjects = [d["requested_rewrite"]["subject"] for d in ds]

print(f"Loaded {len(prompts)} records.")

# 加载MEMIT编辑器
print("Loading Qwen2.5-0.5B model for MEMIT editing ...")
hparams = MEMITHyperParams.from_hparams(os.path.join(ROOT, "hparams", "MEMIT", "qwen2.5-0.5b"))
hparams.model_name = os.path.join(ROOT, "hugging_cache", "Qwen2.5-0.5B")
editor = BaseEditor.from_hparams(hparams)

# 批量编辑，记录耗时和峰值显存
print(f"\nStarting MEMIT batch edit on {BATCH_SIZE} facts ...")

t0 = time.time()

metrics, edited_model, _ = editor.batch_edit(
    prompts=prompts,
    target_new=target_new,
    ground_truth=ground_truth,
    subject=subjects,
    rephrase_prompts=rephrase_prompts,
    locality_inputs={
        "neighborhood": {
            "prompt": locality_prompts,
            "ground_truth": locality_answers,
        }
    },
    keep_original_weight=True,
    verbose=True,
)

elapsed = time.time() - t0
if DEVICE == "cuda":
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
else:
    peak_mem_gb = 0

# 提取指标
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

# 输出资源和效果统计
print(f"\n{'='*60}")
print(f"  MEMIT Batch Edit Results")
print(f"{'='*60}")
print(f"  Device          : {DEVICE.upper()}")
print(f"  Facts edited    : {len(metrics)}")
print(f"  Total time      : {elapsed:.1f}s")
print(f"  Time per fact   : {elapsed / len(metrics) * 1000:.1f}ms")
if DEVICE == "cuda":
    print(f"  Peak GPU memory : {peak_mem_gb:.2f} GiB")
else:
    print(f"  Peak GPU memory : N/A (CPU mode)")
print(f"  {'─'*56}")
print(f"  ES (Efficacy)     : {avg_es:.2%}")
print(f"  PS (Paraphrase)   : {avg_ps:.2%}")
print(f"  NS (Neighborhood) : {avg_ns:.2%}")
print(f"{'='*60}")
