import json
import os
import pandas as pd
import matplotlib
# 强制使用非交互式后端，防止容器内无 GUI 报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 未找到 {file_path}，当前维度可能缺失数据。")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_metrics(metrics_data):
    if not metrics_data:
        return 0.0, 0.0, 0.0
    
    total_es, total_ps, total_ns = 0.0, 0.0, 0.0
    valid_count = len(metrics_data)
    
    for res in metrics_data:
        total_es += res.get('post', {}).get('rewrite_acc', [0.0])[0]
        total_ps += res.get('post', {}).get('rephrase_acc', [0.0])[0]
        total_ns += res.get('post', {}).get('locality', {}).get('neighborhood_acc', [0.0])[0]
        
    return (total_es / valid_count * 100), (total_ps / valid_count * 100), (total_ns / valid_count * 100)

def extract_failures(memit_data, output_path="./failures.json"):
    failures = []
    for item in memit_data:
        # 筛选条件：编辑成功率 (rewrite_acc) 为 0
        es = item.get('post', {}).get('rewrite_acc', [0.0])[0]
        if es == 0.0:
            req = item.get('requested_rewrite', {})
            failures.append({
                "subject": req.get('subject', 'Unknown'),
                "prompt": req.get('prompt', 'Unknown'),
                "target_new": req.get('target_new', 'Unknown'),
                "ground_truth": req.get('ground_truth', 'Unknown'),
                "rephrase_prompt": req.get('rephrase_prompt', 'Unknown'),
                "post_edit_metrics": {
                    "ES (编辑成功率)": es,
                    "PS (泛化成功率)": item.get('post', {}).get('rephrase_acc', [0.0])[0],
                    "NS (局部无关性)": item.get('post', {}).get('locality', {}).get('neighborhood_acc', [0.0])[0]
                }
            })
    
    # 将 indent 改为 4，让导出的 JSON 排版更宽敞易读
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=4)
    print(f"✅ 成功捕获 {len(failures)} 条 MEMIT 失败案例，已导出至 {output_path}")

def main():
    print("\n" + "="*65)
    print(" 📊 启动 Task 4: 综合评估与可视化生成")
    print("="*65)

    # 1. 加载数据 (Baseline假定全错以作对比，实际若有baseline.json也可加载)
    rome_data = load_json("./rome_results.json")
    memit_data = load_json("./memit_results.json")

    # 2. 计算指标
    rome_es, rome_ps, rome_ns = calculate_metrics(rome_data)
    memit_es, memit_ps, memit_ns = calculate_metrics(memit_data)

    # 3. 生成三线对比表
    df = pd.DataFrame({
        "评估维度": ["编辑成功率 (ES)", "知识泛化性 (PS)", "局部特异性 (NS)"],
        "Baseline (原始)": ["0.00%", "0.00%", "100.00%"],
        "ROME (单条编辑)": [f"{rome_es:.2f}%", f"{rome_ps:.2f}%", f"{rome_ns:.2f}%"],
        "MEMIT (500条批量)": [f"{memit_es:.2f}%", f"{memit_ps:.2f}%", f"{memit_ns:.2f}%"]
    })

    print("\n📈 算法多维对比表：\n")
    print(df.to_markdown(index=False))
    
    df.to_csv("./evaluation_table.csv", index=False)
    print("\n💾 表格已保存至 ./evaluation_table.csv")

    # 4. 绘制直方图对比 (离线渲染)
    labels = ['Efficacy (ES)', 'Generalization (PS)', 'Locality (NS)']
    rome_scores = [rome_es, rome_ps, rome_ns]
    memit_scores = [memit_es, memit_ps, memit_ns]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width/2 for i in x], rome_scores, width, label='ROME (Single)', color='#4C72B0')
    ax.bar([i + width/2 for i in x], memit_scores, width, label='MEMIT (Batch 500)', color='#55A868')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Knowledge Editing Performance: ROME vs MEMIT')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    chart_path = "./evaluation_chart.png"
    plt.savefig(chart_path, dpi=300)
    print(f"📊 对比柱状图已保存至 {chart_path}")

    # 5. 提取并保存失败案例
    extract_failures(memit_data)
    print("\n" + "="*65)
    print(" 🎉 Task 4 评估流程全部执行完毕！")
    print("="*65)

if __name__ == "__main__":
    main()