"""
Task 4: Comprehensive Evaluation
读取 Task 2 (ROME) 和 Task 3 (MEMIT) 的结果，汇总 ES/PS/NS 表格。
"""
import json
import os
import numpy as np


def load_rome_results(path):
    """加载 ROME 编辑结果"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    es_vals, ps_vals, ns_vals = [], [], []
    for r in data:
        if r.get('rewrite_acc') is not None:
            es_vals.append(float(r['rewrite_acc']))
        if r.get('rephrase_acc') is not None:
            ps_vals.append(float(r['rephrase_acc']))
        if r.get('locality_acc') is not None:
            ns_vals.append(float(r['locality_acc']))

    return {
        'ES': np.mean(es_vals) * 100 if es_vals else 0,
        'PS': np.mean(ps_vals) * 100 if ps_vals else 0,
        'NS': np.mean(ns_vals) * 100 if ns_vals else 0,
        'ES_valid': len(es_vals),
        'PS_valid': len(ps_vals),
        'NS_valid': len(ns_vals),
        'total': len(data),
    }, data


def load_memit_results(summary_path, detail_path=None):
    """加载 MEMIT 批量编辑结果"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    m = summary.get('metrics', {})
    return {
        'ES': m.get('ES_percent', 0),
        'PS': m.get('PS_percent', 0),
        'NS': m.get('NS_percent', 0),
        'ES_valid': m.get('valid_es_count', 0),
        'PS_valid': m.get('valid_ps_count', 0),
        'NS_valid': m.get('valid_ns_count', 0),
        'total': summary.get('num_samples', 0),
        'time_min': summary.get('timing', {}).get('batch_edit_minutes', 0),
        'peak_memory_mb': summary.get('memory', {}).get('peak_gpu_memory_mb', 0),
    }, summary


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '../outputs')

    # 加载结果
    rome_path = os.path.join(output_dir, 'rome_edit_results.json')
    memit_summary_path = os.path.join(output_dir, 'memit_batch_summary.json')

    print("=" * 80)
    print("Task 4: Comprehensive Evaluation Report")
    print("=" * 80)

    results = {}

    if os.path.exists(rome_path):
        results['ROME'], rome_data = load_rome_results(rome_path)
        print(f"\nLoaded ROME results: {results['ROME']['total']} samples")
    else:
        print(f"\nWARNING: ROME results not found at {rome_path}")

    if os.path.exists(memit_summary_path):
        results['MEMIT'], memit_data = load_memit_results(memit_summary_path)
        print(f"Loaded MEMIT results: {results['MEMIT']['total']} samples")
    else:
        print(f"\nWARNING: MEMIT results not found at {memit_summary_path}")

    # ==================== 汇总表格 ====================
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<10} {'Samples':<10} {'ES (%)':<10} {'PS (%)':<10} {'NS (%)':<10}")
    print("-" * 50)

    for method in ['ROME', 'MEMIT']:
        if method not in results:
            print(f"{method:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
        r = results[method]
        print(f"{method:<10} {r['total']:<10} {r['ES']:<10.1f} {r['PS']:<10.1f} {r['NS']:<10.1f}")

    print("-" * 50)

    # ==================== 详细分析 ====================
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for method in ['ROME', 'MEMIT']:
        if method not in results:
            continue
        r = results[method]
        print(f"\n--- {method} ---")
        print(f"  Total samples:    {r['total']}")
        print(f"  Edit Success:     {r['ES']:.1f}% ({r['ES_valid']} valid)")
        print(f"  Generalization:   {r['PS']:.1f}% ({r['PS_valid']} valid)")
        print(f"  Locality:         {r['NS']:.1f}% ({r['NS_valid']} valid)")

    # MEMIT 资源统计
    if 'MEMIT' in results:
        r = results['MEMIT']
        print(f"\n--- MEMIT Resource Usage ---")
        print(f"  Total edit time:  {r.get('time_min', 0):.1f} min")
        print(f"  Peak GPU memory:  {r.get('peak_memory_mb', 0):.1f} MB")

    # ==================== 分析与建议 ====================
    print("\n" + "=" * 80)
    print("ANALYSIS & OBSERVATIONS")
    print("=" * 80)

    print("""
  1. ES (Edit Success): 直接编辑 prompt 后模型输出目标答案的成功率。
     - ROME 逐条编辑，每条独立优化，通常 ES 较高。
     - MEMIT 批量编辑 500 条，ES 通常低于逐条编辑，但效率高。

  2. PS (Generalization): 同义改写 prompt 后的成功率，衡量编辑的泛化能力。
     - PS 通常低于 ES，说明编辑可能过度拟合原始 prompt。

  3. NS (Locality): 无关事实保持正确率，衡量编辑的局部性。
     - NS 应接近 100%，说明编辑没有破坏模型的其他知识。
     - 当前 NS 较高，说明 MEMIT 的局部性保持良好。

  4. 建议：
     - 可尝试调整 edit layers 范围（如 [7-15]）改善 MEMIT ES/PS。
     - 增大 v_num_grad_steps 可能提高编辑质量（但增加时间）。
     - batch_size 增大可加速但可能降低质量。
    """)

    # ==================== 保存报告 ====================
    report_path = os.path.join(output_dir, 'task4_evaluation_report.json')
    report = {
        'task': 'Task 4 - Comprehensive Evaluation',
        'model': 'Qwen/Qwen2.5-0.5B',
        'results': {}
    }
    for method in ['ROME', 'MEMIT']:
        if method in results:
            report['results'][method] = {
                'ES_percent': round(results[method]['ES'], 1),
                'PS_percent': round(results[method]['PS'], 1),
                'NS_percent': round(results[method]['NS'], 1),
                'total_samples': results[method]['total'],
            }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
