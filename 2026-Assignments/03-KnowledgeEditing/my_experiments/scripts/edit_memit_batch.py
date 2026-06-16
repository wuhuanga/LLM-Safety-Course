"""
Task 3: MEMIT Batch Knowledge Editing
对 500 条 ZsRE/CounterFact 数据使用 MEMIT 进行批量知识注入
记录显存占用和耗时
"""
import json
import sys
import os
import time
import torch
import numpy as np

# 添加 EasyEdit 路径
easyedit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../EasyEdit'))
if os.path.exists(easyedit_path):
    sys.path.insert(0, easyedit_path)

from easyeditor import BaseEditor, MEMITHyperParams


def load_zsre_data(data_path, max_samples=500):
    """
    加载 ZsRE 数据集，适配 EasyEdit 的格式。
    支持 zsre_mend_eval.json 和 zsre_mend_eval_portability_gpt4.json 两种格式。
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    data = []
    for i, record in enumerate(raw):
        if len(data) >= max_samples:
            break

        prompt = record.get("src", "")
        target_new = record.get("alt", "")
        ground_truth = record.get("answers", [""])[0] if "answers" in record else ""
        rephrase = record.get("rephrase", prompt)
        loc_prompt = record.get("loc", "")
        loc_ans = record.get("loc_ans", "")

        # subject: portability 版本有 'subject' 字段，基本版本可能没有
        # 尝试从 src 中提取 {} 中的内容，或使用 record['subject']
        subject = record.get("subject", None)

        if subject is None or subject.strip() == "":
            # 回退策略：如果 prompt 中有 {}，提取 {} 前后的关键词作为 subject
            # 这里使用一个简单的启发式方法
            if "{}" in prompt:
                # prompt 已经是模板格式，跳过
                subject = prompt.replace("{}", "").strip().split()[-1] if prompt.replace("{}", "").strip() else "it"
            else:
                # 取 prompt 的后半部分作为 subject（启发式）
                words = prompt.strip().split()
                subject = words[-1] if words else prompt.strip()

        # 清理 locality prompt（ZsRE 格式以 "nq question: " 开头）
        if loc_prompt.startswith("nq question: "):
            loc_prompt = loc_prompt[len("nq question: "):]

        # 过滤空 target_new（ZsRE 数据集中有些条目的 alt 为空）
        if not target_new or not target_new.strip():
            continue

        data.append({
            "prompt": prompt,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "rephrase_prompt": rephrase,
            "locality_prompt": loc_prompt,
            "locality_ground_truth": loc_ans,
            "subject": subject,
        })

    print(f"Loaded {len(data)} samples from {os.path.basename(data_path)}")
    return data


def get_gpu_memory_usage():
    """返回当前 GPU 显存使用量 (MB)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return allocated, reserved
    return 0, 0


def main():
    # ==================== 配置 ====================
    # ZsRE 数据集路径（请确保已下载到此处）
    # 优先使用 portability 版本（有 subject 字段），否则使用基本版本
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../EasyEdit/data/zsre'))
    data_paths = [
        os.path.join(data_dir, 'zsre_mend_eval_portability_gpt4.json'),
        os.path.join(data_dir, 'zsre_mend_eval.json'),
    ]
    data_path = None
    for p in data_paths:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        print(f"ERROR: No ZsRE data found. Looked in:")
        for p in data_paths:
            print(f"  {p}")
        print("\nPlease download the ZsRE dataset from:")
        print("  https://drive.google.com/file/d/1YtQvv4WvTa4rJyDYQR2J-uK8rnrt0kTA/view?usp=sharing")
        print("And place the JSON files in EasyEdit/data/zsre/")
        return

    N_SAMPLES = 500  # Task 3 要求 500 条

    # ==================== 加载数据 ====================
    print("=" * 80)
    print("Task 3: MEMIT Batch Knowledge Editing")
    print("=" * 80)

    data = load_zsre_data(data_path, max_samples=N_SAMPLES)

    # 提取各字段
    prompts = [d["prompt"] for d in data]
    targets = [d["target_new"] for d in data]
    subjects = [d["subject"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    rephrases = [d["rephrase_prompt"] for d in data]
    loc_prompts = [d["locality_prompt"] for d in data]
    loc_answers = [d["locality_ground_truth"] for d in data]

    # ==================== 加载配置和模型 ====================
    yaml_path = os.path.join(os.path.dirname(__file__), '../config/memit_qwen0.5b.yaml')
    hparams = MEMITHyperParams.from_hparams(yaml_path)

    print(f"\nConfiguration:")
    print(f"  Model:     {hparams.model_name}")
    print(f"  Algorithm: MEMIT")
    print(f"  Layers:    {hparams.layers}")
    print(f"  Batch size:{hparams.batch_size}")
    print(f"  Samples:   {N_SAMPLES}")
    print(f"  Data:      {os.path.basename(data_path)}")

    # ==================== 记录编辑前显存 ====================
    torch.cuda.empty_cache()
    mem_before_alloc, mem_before_reserved = get_gpu_memory_usage()
    print(f"\nGPU Memory before loading model: {mem_before_alloc:.1f} MB allocated, {mem_before_reserved:.1f} MB reserved")

    print("\nLoading model...")
    t0 = time.time()

    editor = BaseEditor.from_hparams(hparams)

    model_load_time = time.time() - t0
    mem_model_alloc, mem_model_reserved = get_gpu_memory_usage()
    print(f"Model loaded in {model_load_time:.1f}s")
    print(f"GPU Memory after loading: {mem_model_alloc:.1f} MB allocated, {mem_model_reserved:.1f} MB reserved")
    print(f"Model memory footprint: ~{mem_model_alloc - mem_before_alloc:.1f} MB")

    # ==================== 构建 locality_inputs ====================
    locality_inputs = {
        'neighborhood': {
            'prompt': loc_prompts,
            'ground_truth': loc_answers
        }
    }

    # ==================== 执行批量编辑 ====================
    print("\n" + "=" * 60)
    print("Starting MEMIT batch edit...")
    print("=" * 60)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before_edit, _ = get_gpu_memory_usage()

    t_edit_start = time.time()

    try:
        # 使用 batch_edit 进行真正的批量编辑
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts,
            target_new=targets,
            ground_truth=ground_truths,
            rephrase_prompts=rephrases,
            locality_inputs=locality_inputs,
            subject=subjects,
            sequential_edit=False,
            verbose=True,
        )
    except Exception as e:
        print(f"\nERROR during batch edit: {e}")
        import traceback
        traceback.print_exc()
        return

    edit_time = time.time() - t_edit_start

    mem_after_edit, _ = get_gpu_memory_usage()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ==================== 汇总结果 ====================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Task 3: MEMIT Batch Editing)")
    print("=" * 80)

    # 时间和显存统计
    print(f"\n--- Resource Usage ---")
    print(f"  Model load time:       {model_load_time:.1f}s")
    print(f"  Batch edit time:       {edit_time:.1f}s ({edit_time/60:.1f} min)")
    print(f"  Time per sample:       {edit_time/N_SAMPLES*1000:.0f} ms")
    print(f"  Peak GPU memory:       {peak_memory:.1f} MB")
    print(f"  GPU mem after edit:    {mem_after_edit:.1f} MB allocated")

    # 编辑质量统计
    es_vals = []  # Edit Success
    ps_vals = []  # Paraphrase/Generalization Success
    ns_vals = []  # Neighborhood/Locality Success

    for m in metrics:
        post = m.get('post', {})

        es = post.get('rewrite_acc', None)
        if isinstance(es, (list, np.ndarray)):
            es = float(np.mean(es))
        elif es is not None:
            es = float(es)

        ps = post.get('rephrase_acc', None)
        if isinstance(ps, (list, np.ndarray)):
            ps = float(np.mean(ps))
        elif ps is not None:
            ps = float(ps)

        ns = 1.0
        if 'locality' in post and post['locality']:
            for lkey in post['locality']:
                if lkey.endswith('_acc'):
                    val = post['locality'][lkey]
                    if isinstance(val, (list, np.ndarray)):
                        ns = float(np.mean(val))
                    else:
                        ns = float(val)
                    break

        if es is not None:
            es_vals.append(es)
        if ps is not None:
            ps_vals.append(ps)
        ns_vals.append(ns)

    es_mean = np.mean(es_vals) * 100 if es_vals else 0
    ps_mean = np.mean(ps_vals) * 100 if ps_vals else 0
    ns_mean = np.mean(ns_vals) * 100 if ns_vals else 0

    print(f"\n--- Editing Quality ---")
    print(f"  Edit Success (ES):        {es_mean:.1f}%  ({len(es_vals)} valid)")
    print(f"  Generalization (PS):      {ps_mean:.1f}%  ({len(ps_vals)} valid)")
    print(f"  Locality (NS):            {ns_mean:.1f}%  ({len(ns_vals)} valid)")
    print(f"  Samples edited:           {len(metrics)}")

    # ==================== 保存结果 ====================
    output_dir = os.path.join(os.path.dirname(__file__), '../outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 保存汇总
    summary = {
        "task": "Task 3 - MEMIT Batch Editing",
        "model": hparams.model_name,
        "algorithm": "MEMIT",
        "layers": hparams.layers,
        "batch_size": hparams.batch_size,
        "num_samples": N_SAMPLES,
        "data_source": os.path.basename(data_path),
        "timing": {
            "model_load_seconds": round(model_load_time, 1),
            "batch_edit_seconds": round(edit_time, 1),
            "batch_edit_minutes": round(edit_time / 60, 1),
            "time_per_sample_ms": round(edit_time / N_SAMPLES * 1000, 0),
        },
        "memory": {
            "peak_gpu_memory_mb": round(peak_memory, 1),
            "model_memory_mb": round(mem_model_alloc - mem_before_alloc, 1),
        },
        "metrics": {
            "ES_percent": round(es_mean, 1),
            "PS_percent": round(ps_mean, 1),
            "NS_percent": round(ns_mean, 1),
            "valid_es_count": len(es_vals),
            "valid_ps_count": len(ps_vals),
            "valid_ns_count": len(ns_vals),
            "total_edited": len(metrics),
        }
    }

    summary_path = os.path.join(output_dir, 'memit_batch_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    # 保存逐条结果（精简版，去掉 request 等不可序列化内容）
    clean_metrics = []
    for m in metrics:
        clean = {
            'case_id': m.get('case_id', -1),
            'rewrite_acc': None,
            'rephrase_acc': None,
            'locality_acc': None,
        }
        post = m.get('post', {})
        es = post.get('rewrite_acc')
        if isinstance(es, (list, np.ndarray)):
            clean['rewrite_acc'] = [float(x) for x in es]
        elif es is not None:
            clean['rewrite_acc'] = float(es)

        ps = post.get('rephrase_acc')
        if isinstance(ps, (list, np.ndarray)):
            clean['rephrase_acc'] = [float(x) for x in ps]
        elif ps is not None:
            clean['rephrase_acc'] = float(ps)

        if 'locality' in post and post['locality']:
            for lkey in post['locality']:
                if lkey.endswith('_acc'):
                    val = post['locality'][lkey]
                    if isinstance(val, (list, np.ndarray)):
                        clean['locality_acc'] = [float(x) for x in val]
                    else:
                        clean['locality_acc'] = float(val)
                    break

        clean_metrics.append(clean)

    detail_path = os.path.join(output_dir, 'memit_batch_details.json')
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)
    print(f"Details saved to: {detail_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
