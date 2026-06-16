"""Task 1: 基线测试 — 在不做任何编辑的情况下，测试原始模型对10条过时知识的回答。"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

from easyeditor import BaseEditor, ROMEHyperParams
from easyeditor.evaluate.evaluate_utils import exact_match_score

# 从JSON文件加载自定义测试数据集
FACTS = json.load(open(os.path.join(ROOT, "data", "custom_facts.json"), "r", encoding="utf-8"))

# 加载模型（不做任何编辑，只用于推理）
print("Loading Qwen2.5-0.5B model ...")
hparams = ROMEHyperParams.from_hparams(os.path.join(ROOT, "hparams", "ROME", "qwen2.5-0.5b"))
hparams.model_name = os.path.join(ROOT, "hugging_cache", "Qwen2.5-0.5B")
editor = BaseEditor.from_hparams(hparams)
model, tok = editor.model, editor.tok
model.eval()

# 逐条推理，统计旧知识和新知识的命中数
print()

old_hits, new_hits = 0, 0

# 约束 base model 输出格式，避免生成选择题等无关续写，不引入额外知识
FEWSHOT_PREFIX = "问题：苹果公司的CEO是\n答：蒂姆·库克\n\n问题：Google的母公司是\n答：Alphabet\n\n问题：法国现任总统是\n答：埃马纽埃尔·马克龙\n\n"

for i, fact in enumerate(FACTS):
    prompt_text = FEWSHOT_PREFIX + f"问题：{fact['prompt']}\n答："
    inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=1.2,
        )
    gen_text = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    hit_old = exact_match_score(gen_text, fact["ground_truth"])
    hit_new = exact_match_score(gen_text, fact["target_new"])
    old_hits += hit_old
    new_hits += hit_new

    print(f"[{i+1}] 问题: {fact['prompt']}")
    print(f"    输出: {gen_text if gen_text else '(空)'}")
    print(f"    旧答案: {fact['ground_truth']}  新答案: {fact['target_new']}")
    print(f"    匹配旧知识: {'是' if hit_old else '否'}    匹配新知识: {'是' if hit_new else '否'}")
    print()

print("=" * 60)
print(f"旧知识命中: {old_hits}/10    新知识命中: {new_hits}/10")
if old_hits > new_hits:
    print("结论: 模型依赖旧知识 — 基线成立，可以进行编辑实验。")
else:
    print("注意: 模型未稳定持有旧知识，但 MEMIT 编辑仍可验证效果。")
