import os
import io
# 🚨 开启显存碎片整理，6G显存护身符
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" 

import json, time, torch, sys, csv, datetime
import transformers
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("easyeditor").setLevel(logging.ERROR)

sys.path.insert(0, os.path.abspath("./EasyEdit"))
from easyeditor import BaseEditor, MEMITHyperParams

def pad_str(text, length):
    text = str(text)
    display_len = sum(2 if ord(c) > 127 else 1 for c in text)
    return text + ' ' * max(0, length - display_len)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# 智能兼容加载器（支持 CounterFact 和 zsRE）
def load_dataset_intelligently(path, num_samples=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_samples is not None: data = data[:num_samples]
    requests = []
    for item in data:
        if "requested_rewrite" in item:
            requests.append({
                "prompt": item["requested_rewrite"]["prompt"],
                "target_new": item["requested_rewrite"]["target_new"]["str"],
                "subject": item["requested_rewrite"]["subject"]
            })
        else:
            requests.append({
                "prompt": item.get("prompt", item.get("src", "")),
                "target_new": item.get("target_new", item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else ""),
                "subject": item.get("subject", "")
            })
    return requests

def main():
    os.makedirs("outputs", exist_ok=True)
    print("\n" + "=" * 120)
    print("🚀 Task 3: MEMIT Batch Knowledge Editing (GPT-2 Medium)")
    print("=" * 120)
    
    dataset_path = "./data/zsre_mend_train.json" 
    print("⏳ Loading Dataset...")
    requests = load_dataset_intelligently(dataset_path, num_samples=500)
    print(f"✅ Loaded {len(requests)} editing samples")
    
    print("⏳ Initializing MEMIT Editor with GPT-2...")
    with HiddenPrints():
        hparams = MEMITHyperParams.from_hparams("./gpt-medium-memit.yaml")
        editor = BaseEditor.from_hparams(hparams)
        
    if editor.tok.pad_token is None:
        editor.tok.pad_token = editor.tok.eos_token
    
    prompts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    subjects = [r["subject"] for r in requests]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    print(f"⚡ Starting MEMIT Batch Injection for {len(requests)} items... (Please wait)")
    with HiddenPrints():
        # 🚨 keep_original_weight=False，确保知识留在模型里进行验证
        metrics, edited_model, orig_weights = editor.edit(
            prompts=prompts, target_new=targets, subject=subjects, keep_original_weight=False 
        )

    end_time = time.time()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print("\n" + "=" * 120)
    print("📊 Batch Editing Statistics")
    print("=" * 120)
    print(f"Editing Time    : {(end_time-start_time):.2f} seconds")
    print(f"Peak GPU Memory : {peak_memory:.2f} GB")

    print("\n" + "=" * 120)
    print("🔍 Task 3 Verification (Academic Logit Evaluation - With Prompts)")
    print("=" * 120)

    success_count = 0
    results_list = []
    
    print(f"{pad_str('ID', 4)} | {pad_str('Subject', 22)} | {pad_str('Prompt (Truncated)', 45)} | {pad_str('Hit', 5)} | {pad_str('Target New', 20)}")
    print("-" * 120)

    for i, metric in enumerate(metrics):
        prompt = prompts[i]
        target = targets[i]
        subject = subjects[i]
        
        # 截断以保持排版整洁
        prompt_trunc = prompt[:42] + "..." if len(prompt) > 45 else prompt
        subject_trunc = subject[:19] + "..." if len(subject) > 22 else subject
        target_trunc = target[:17] + "..." if len(target) > 20 else target
        
        post_acc = metric['post']['rewrite_acc']
        is_success = (post_acc[0] == 1.0) if isinstance(post_acc, list) else (post_acc == 1.0)

        if is_success: 
            success_count += 1
            
        print(f"{pad_str(i+1, 4)} | {pad_str(subject_trunc, 22)} | {pad_str(prompt_trunc, 45)} | {pad_str(str(is_success), 5)} | {pad_str(target_trunc, 20)}")
        
        results_list.append({
            "id": i + 1, "subject": subject, "prompt": prompt, "target_hit": str(is_success), "target_new": target
        })

    print("-" * 120)
    accuracy = (success_count / len(requests)) * 100
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join("outputs", f"task3_memit_results_{timestamp}.json")
    csv_path = os.path.join("outputs", f"task3_memit_results_{timestamp}.csv")
    
    print(f"🎯 Final MEMIT Success Rate (Internal Logits): {success_count}/{len(requests)} ({accuracy:.1f}%)")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)
        
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject", "prompt", "target_hit", "target_new"])
        writer.writeheader()
        writer.writerows(results_list)

if __name__ == "__main__":
    main()