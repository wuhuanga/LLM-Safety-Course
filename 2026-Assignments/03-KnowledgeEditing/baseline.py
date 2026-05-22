import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def run_baseline():
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0"
    data_path = "./data.json"
    # 1. 加载本地被隔离的模型与分词器
    print(f"正在加载基座模型: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    # 2. 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print("\n" + "="*60)
    print(" 🚀 开始基线测试 (Baseline Evaluation)")
    print("="*60)
    # 3. 遍历数据进行推理推理
    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        ground_truth = item["ground_truth"]
        target_new = item["target_new"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)   
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  
            )   
        # 解码并提取模型生成的原始文本
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):].strip()       
        print(f"📍 数据 [{i+1}/{len(dataset)}]")
        print(f"Prompt (输入):          {prompt}")
        print(f"Ground Truth (原知识):  {ground_truth}")
        print(f"Target New (待编辑为):  {target_new}")
        print(f"Model Output (实际输出): {generated_text}")
        print("-" * 60)      
    print("✅ 基线测试完毕。请截图保存终端输出，作为 Task 1 的实验报告素材。")
if __name__ == "__main__":
    run_baseline()