import sys
import io
import os
import json
import time
import torch
import transformers

# 🚨 强制终端使用 UTF-8，防报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" 

transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("easyeditor").setLevel(logging.ERROR)

sys.path.insert(0, os.path.abspath("./EasyEdit"))
from easyeditor import BaseEditor, MEMITHyperParams

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def pad_str(text, length):
    text = str(text)
    display_len = sum(2 if ord(c) > 127 else 1 for c in text)
    return text + ' ' * max(0, length - display_len)

def load_and_format_dataset(path, num_samples=500):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if num_samples is not None:
        data = data[:num_samples]
        
    prompts, targets, subjects = [], [], []
    rephrase_prompts = []
    loc_prompts, loc_ans = [], []

    for item in data:
        # 精准抓取你发来的 JSON 字段格式
        prompts.append(item.get("src", ""))
        ans = item.get("answers", [""])
        targets.append(ans[0] if isinstance(ans, list) else ans)
        subjects.append(item.get("subject", ""))
        
        rephrase_prompts.append(item.get("rephrase", ""))
        loc_prompts.append(item.get("loc", ""))
        loc_ans.append(item.get("loc_ans", ""))

    locality_inputs = {}
    if any(loc_prompts): 
        locality_inputs = {
            'neighborhood': {
                'prompt': loc_prompts,
                'ground_truth': loc_ans
            }
        }
    else:
        locality_inputs = None 

    portability_inputs = {}
    if any(rephrase_prompts):
        portability_inputs = {
            'rephrase': {
                'prompt': rephrase_prompts,
                'ground_truth': targets  
            }
        }
    else:
        portability_inputs = None

    return prompts, targets, subjects, portability_inputs, locality_inputs

def safe_extract_metric(metric_dict):
    if not metric_dict: return "N/A"
    try:
        values = list(metric_dict.values())
        if not values: return "N/A"
        first_val = values[0]
        if isinstance(first_val, list) and len(first_val) > 0:
            return f"{float(first_val[0]):.4f}"
        return f"{float(first_val):.4f}"
    except:
        return "N/A"

def main():
    print("\n" + "=" * 120)
    print("🚀 Task 4: Comprehensive Evaluation (Generalization & Locality)")
    print("=" * 120)
    
    dataset_path = "./data/zsre_mend_train.json" 
    num_test_samples = 500
    
    print(f"⏳ Loading Dataset...")
    prompts, targets, subjects, portability_inputs, locality_inputs = load_and_format_dataset(dataset_path, num_samples=num_test_samples)
    print(f"✅ Loaded {len(prompts)} samples for deep evaluation.")

    print("⏳ Initializing MEMIT Editor with GPT-2...")
    with HiddenPrints():
        hparams = MEMITHyperParams.from_hparams("./gpt-medium-memit.yaml")
        editor = BaseEditor.from_hparams(hparams)
        
    if editor.tok.pad_token is None:
        editor.tok.pad_token = editor.tok.eos_token
    
    print("⚡ Starting Deep Knowledge Injection and Multi-dimensional Evaluation...")
    print("   (This involves evaluating extra paraphrase and neighborhood prompts, please wait...)")
    
    start_time = time.time()
    with HiddenPrints():
        # 🚨 正确传入 portability_inputs 字典，唤醒底层的泛化计算！
        metrics, _, _ = editor.edit(
            prompts=prompts, 
            target_new=targets, 
            subject=subjects, 
            portability_inputs=portability_inputs,
            locality_inputs=locality_inputs,
            keep_original_weight=False 
        )
    end_time = time.time()

    print("\n" + "=" * 120)
    print("📊 Task 4 Advanced Metrics Report")
    print("=" * 120)
    print(f"{pad_str('ID', 4)} | {pad_str('Subject', 22)} | {pad_str('Efficacy (Rewrite)', 20)} | {pad_str('Generalization (Port)', 23)} | {pad_str('Locality (No Bleed)', 20)}")
    print("-" * 120)

    total_rewrite, total_port, total_loc = 0.0, 0.0, 0.0
    valid_port_count, valid_loc_count = 0, 0

    for i, metric in enumerate(metrics):
        subject = subjects[i][:19] + "..." if len(subjects[i]) > 22 else subjects[i]
        post = metric.get('post', {})
        
        # 1. ES (编辑成功率)
        rewrite_acc = post.get('rewrite_acc', [0.0])
        rewrite_val = rewrite_acc[0] if isinstance(rewrite_acc, list) else rewrite_acc
        total_rewrite += float(rewrite_val)
        
        # 2. PS (泛化能力)
        port_val_str = safe_extract_metric(post.get('portability', {}))
        if port_val_str != "N/A":
            total_port += float(port_val_str)
            valid_port_count += 1
            
        # 3. NS (局部稳定性)
        loc_val_str = safe_extract_metric(post.get('locality', {}))
        if loc_val_str != "N/A":
            total_loc += float(loc_val_str)
            valid_loc_count += 1

        print(f"{pad_str(i+471, 4)} | {pad_str(subject, 22)} | {pad_str(f'{float(rewrite_val):.4f}', 20)} | {pad_str(port_val_str, 23)} | {pad_str(loc_val_str, 20)}")

    print("-" * 120)
    
    avg_rewrite = (total_rewrite / len(prompts)) * 100
    avg_port = (total_port / valid_port_count * 100) if valid_port_count > 0 else 0.0
    avg_loc = (total_loc / valid_loc_count * 100) if valid_loc_count > 0 else 0.0
    
    

    print(f"⏱️ Total Evaluation Time: {(end_time - start_time):.2f} seconds")
    print(f"🎯 Average Efficacy (Rewrite Success)  : {avg_rewrite:.2f}%")
    print(f"🌍 Average Portability (Generalization): {avg_port:.2f}%")
    print(f"🛡️ Average Locality (Knowledge Shield) : {avg_loc:.2f}%")
    print("=" * 120)

if __name__ == "__main__":
    main()
