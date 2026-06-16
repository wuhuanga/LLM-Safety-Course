import json
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加EasyEdit路径（如果脚本不在EasyEdit目录内）
# 根据你的实际目录结构调整
easyedit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../EasyEdit'))
if os.path.exists(easyedit_path):
    sys.path.insert(0, easyedit_path)

def load_model(model_name="Qwen/Qwen2.5-0.5B"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉输入prompt部分，只保留生成内容
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

def main():
    # 数据集路径（根据实际位置调整）
    data_path = os.path.join(os.path.dirname(__file__), '../data/custom_edit.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model, tokenizer = load_model()
    
    print("\n" + "="*80)
    print("Task 1: Baseline Evaluation (Before Editing)")
    print("="*80 + "\n")
    
    results = []
    for idx, item in enumerate(data):
        prompt = item['prompt']
        target_new = item['target_new']
        ground_truth = item.get('ground_truth', 'N/A')
        
        print(f"Sample {idx+1}:")
        print(f"Prompt: {prompt}")
        print(f"Expected target (edit target): {target_new}")
        print(f"Ground truth (original answer): {ground_truth}")
        
        response = generate_response(model, tokenizer, prompt)
        print(f"Model response: {response}")
        
        # 简单判断模型回答是否包含目标答案（大小写不敏感）
        contains_target = target_new.lower() in response.lower()
        contains_ground = ground_truth.lower() in response.lower() if ground_truth != 'N/A' else False
        
        print(f"Contains target_new? {contains_target}")
        print(f"Contains ground_truth? {contains_ground}")
        print("-"*60)
        
        results.append({
            "prompt": prompt,
            "response": response,
            "contains_target": contains_target,
            "contains_ground": contains_ground
        })
    
    # 统计基线准确率
    total = len(results)
    correct_ground = sum(1 for r in results if r['contains_ground'])
    print(f"\nSummary: Out of {total} prompts, model originally answered ground_truth correctly in {correct_ground} cases ({correct_ground/total*100:.1f}%).")
    print("This demonstrates the need for knowledge editing to correct or update facts.")
    
    # 可选：保存结果到文件
    output_path = os.path.join(os.path.dirname(__file__), '../outputs/baseline_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()