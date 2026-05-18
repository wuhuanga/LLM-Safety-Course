import os
import sys
import io
import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" 

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
transformers.logging.set_verbosity_error()

def main():
    dataset_path = "./dataset.json" 
    output_path = "baseline_results.json"

    if not os.path.exists(dataset_path):
        print(f"❌ 错误：在当前目录下找不到 {dataset_path}，请检查文件名！")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    dataset = raw_data[:10] 

    # 4. 加载原始 GPT-2 Medium
    model_name = "gpt2-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    saved_results = []

    for item in dataset:
        prompt_text = item.get("prompt", item.get("src", ""))
        gt_text = item.get("ground_truth", item.get("target_true", item.get("target_new", item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else "")))
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        model_output = model_output.split('\n')[0].strip()
        
        print(f"[Q]: {prompt_text}")
        print(f"[A] (Model): {model_output}")
        print(f"[Target]: {gt_text}")
        print()

  
        saved_results.append({
            "prompt": prompt_text,
            "model_output": model_output,
            "target": gt_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_results, f, ensure_ascii=False, indent=4)

    print("---------------------------------------------------------------------------")
    print(f"Baseline Evaluation Completed! Results saved to {output_path}")

if __name__ == "__main__":
    main()