import json
import os
import time
import torch
import traceback
from easyeditor import BaseEditor, MEMITHyperParams

def run_memit_batch_edit():
    data_path = "./batch_data_500.json"
    map_path = "./subject_map.json"
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0" 
    hparams_path = "./hparams/MEMIT/tinyllama.yaml"    

    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"❌ 找不到 MEMIT 配置文件: {hparams_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"❌ 找不到映射表: {map_path}")

    print(f"📦 正在加载 MEMIT 配置与模型: {hparams_path}")
    hparams = MEMITHyperParams.from_hparams(hparams_path)
    
    hparams.model_name = model_path
    editor = BaseEditor.from_hparams(hparams)

    # 直接将原始读取的数据赋给 dataset
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(map_path, "r", encoding="utf-8") as f:
        subject_map = json.load(f)

    prompts = [item["prompt"] for item in dataset]
    targets_new = [item["target_new"] for item in dataset]
    ground_truths = [item["ground_truth"] for item in dataset]
    rephrase_prompts = [item["rephrase_prompt"] for item in dataset]
    subjects = [subject_map[p] for p in prompts]
    
    locality_inputs = {
        'neighborhood': {
            'prompt': [item["locality_prompt"] for item in dataset],
            'ground_truth': [item["locality_ground_truth"] for item in dataset]
        }
    }

    print("\n" + "="*65)
    print(f" 🚀 开始 Task 3: MEMIT 批量注入 {len(dataset)} 条事实")
    print("="*65)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        metrics, _, _ = editor.edit(
            prompts=prompts, 
            ground_truth=ground_truths, 
            target_new=targets_new,
            subject=subjects, 
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            keep_original_weight=True 
        )
    except Exception as e:
        print(f"\n❌ 底层框架执行报错: {e}")
        traceback.print_exc()
        return

    end_time = time.time()
    time_cost = end_time - start_time
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\n" + "="*65)
    print(" 📊 Task 3 性能评估报告")
    print("="*65)
    print(f"⏱️ 批量编辑耗时: {time_cost:.2f} 秒")
    print(f"💾 峰值显存占用: {max_memory_allocated:.2f} MB")
    
    if metrics:
        total_es, total_ps, total_ns = 0.0, 0.0, 0.0
        valid_count = len(metrics)
        
        for res in metrics:
            total_es += res.get('post', {}).get('rewrite_acc', [0.0])[0]
            total_ps += res.get('post', {}).get('rephrase_acc', [0.0])[0]
            total_ns += res.get('post', {}).get('locality', {}).get('neighborhood_acc', [0.0])[0]
            
        avg_es = (total_es / valid_count) * 100
        avg_ps = (total_ps / valid_count) * 100
        avg_ns = (total_ns / valid_count) * 100

        print(f"\n🎯 平均编辑成功率 (Efficacy, ES): {avg_es:.2f}%")
        print(f"🔄 平均泛化性 (Generalization, PS): {avg_ps:.2f}%")
        print(f"🛡️ 平均局部性 (Locality, NS): {avg_ns:.2f}%\n")

    
    # 新增：结果落盘，供 evaluate.py 读取
    if metrics:
        with open("./memit_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("💾 MEMIT 批量评估指标已保存至 ./memit_results.json")
    
    else:
        print("⚠️ 未能成功提取评估指标。")

if __name__ == "__main__":
    run_memit_batch_edit()
