import json
import os
import torch
from easyeditor import BaseEditor, ROMEHyperParams

SUBJECT_MAPPING = {
    "The current CEO of Twitter (X) is": "Twitter (X)",
    "The capital of France is": "France",
    "The author of Harry Potter is": "Harry Potter",
    "The highest mountain on Earth is": "Earth",
    "The main ingredient in guacamole is": "guacamole",
    "The planet closest to the Sun is": "Sun",
    "The founder of Microsoft is": "Microsoft",
    "The Eiffel Tower is located in the city of": "Eiffel Tower",
    "The lead singer of the rock band Queen was": "Queen",
    "The chemical symbol for water is": "water"
}

def run_rome_single_edit():
    data_path = "./data.json"
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0" 
    hparams_path = "./hparams/ROME/tinyllama.yaml"    

    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"找不到配置文件: {hparams_path}")

    print(f"正在加载 ROME 配置: {hparams_path}")
    hparams = ROMEHyperParams.from_hparams(hparams_path)
    hparams.model_name = model_path
    editor = BaseEditor.from_hparams(hparams)

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("\n" + "="*65)
    print(" 🚀 开始 Task 2: 循环编辑 (严格重置权重)")
    print("="*65)
    all_metrics = [] 
    # ==========================================
    # 阶段一：严格遵守指导书的循环测试（每次重置权重）
    # ==========================================
    for i, item in enumerate(dataset):
        prompts = [item["prompt"]]
        target_new = [item["target_new"]]
        ground_truth = [item["ground_truth"]]
        rephrase_prompts = [item["rephrase_prompt"]]
        current_subject = SUBJECT_MAPPING.get(item["prompt"], "")
        
        locality_inputs = {
            'neighborhood': {
                'prompt': [item["locality_prompt"]],
                'ground_truth': [item["locality_ground_truth"]]
            }
        }

        print(f"\n📍 正在编辑第 [{i+1}/{len(dataset)}] 条...")
        
        metrics, _, _ = editor.edit(
            prompts=prompts, ground_truth=ground_truth, target_new=target_new,
            subject=[current_subject], rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            keep_original_weight=True  # 保持合规
        )
        all_metrics.extend(metrics) 

        if len(metrics) > 0:
            res = metrics[0]
            es = res.get('post', {}).get('rewrite_acc', [0.0])[0]
            ps = res.get('post', {}).get('rephrase_acc', [0.0])[0]
            ns = res.get('post', {}).get('locality', {}).get('neighborhood_acc', [0.0])[0]
            print(f"🎯 ES: {es * 100:.2f}% | 🔄 PS: {ps * 100:.2f}% | 🛡️ NS: {ns * 100:.2f}%")

    # ==========================================
    # 阶段二：官方硬性验证要求（独立注入与生成测试）
    # ==========================================
    print("\n" + "="*65)
    print("  最终验证：针对推特CEO事实的文本生成测试")
    print("="*65)
    
    target_item = dataset[0] # 取出推特CEO这条数据
    
    # 最后一次专属注入，这次设为 False 以便提取模型生成结果
    _, edited_model, _ = editor.edit(
        prompts=[target_item["prompt"]],
        ground_truth=[target_item["ground_truth"]],
        target_new=[target_item["target_new"]],
        subject=[SUBJECT_MAPPING.get(target_item["prompt"], "")],
        keep_original_weight=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = editor.tok(target_item["prompt"], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = edited_model.generate(
            **inputs, max_new_tokens=10,
            pad_token_id=editor.tok.eos_token_id, do_sample=False
        )
    
    generated_text = editor.tok.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🔥 验证环节模型真实输出: {generated_text}\n")
    print("✅ Task 2: ROME 单条编辑与生成验证全部完成！")

    # 新增：结果落盘，供 evaluate.py 读取
    if all_metrics: 
        with open("./rome_results.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2) 
        print("💾 ROME 评估指标已保存至 ./rome_results.json")


if __name__ == "__main__":
    run_rome_single_edit()
