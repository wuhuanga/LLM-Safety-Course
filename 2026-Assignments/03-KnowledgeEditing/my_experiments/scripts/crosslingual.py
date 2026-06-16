"""
Bonus Task: Cross-lingual Generalization (跨语种泛化测试)
用英文向模型注入事实，测试中文提问是否能输出正确结果。
"""
import json
import sys
import os
import torch
import numpy as np

easyedit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../EasyEdit'))
if os.path.exists(easyedit_path):
    sys.path.insert(0, easyedit_path)

from easyeditor import BaseEditor, ROMEHyperParams
from transformers import AutoTokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response


def check_answer(response, target):
    return target.lower() in response.lower()


def main():
    # ==================== 测试数据 ====================
    # 英文事实: 注入知识  中文测试: 验证跨语种泛化
    test_cases = [
        {
            "en_prompt": "The capital of France is",
            "target_new": "Paris",
            "subject": "France",
            "cn_prompt": "法国的首都是哪里？",
        },
        {
            "en_prompt": "The largest ocean on Earth is",
            "target_new": "Pacific Ocean",
            "subject": "largest ocean",
            "cn_prompt": "地球上最大的洋是哪个？",
        },
        {
            "en_prompt": "The chemical symbol for water is",
            "target_new": "H2O",
            "subject": "water",
            "cn_prompt": "水的化学符号是什么？",
        },
        {
            "en_prompt": "The fastest land animal is",
            "target_new": "cheetah",
            "subject": "fastest land animal",
            "cn_prompt": "陆地上最快的动物是什么？",
        },
        {
            "en_prompt": "The longest river in the world is",
            "target_new": "Nile",
            "subject": "longest river",
            "cn_prompt": "世界上最长的河是什么？",
        },
    ]

    print("=" * 80)
    print("Bonus Task: Cross-lingual Generalization Test")
    print("=" * 80)
    print(f"\nTest cases: {len(test_cases)}")
    print("Strategy: Inject fact in ENGLISH → Test answer in CHINESE")

    # ==================== 加载模型 ====================
    yaml_path = os.path.join(os.path.dirname(__file__), '../config/rome_qwen0.5b.yaml')
    hparams = ROMEHyperParams.from_hparams(yaml_path)
    print(f"\nLoading model: {hparams.model_name} ...")
    editor = BaseEditor.from_hparams(hparams)
    tokenizer = editor.tok

    results = []

    for idx, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Case {idx+1}/{len(test_cases)}")
        print(f"  EN prompt:    {case['en_prompt']}")
        print(f"  Target:       {case['target_new']}")
        print(f"  CN prompt:    {case['cn_prompt']}")

        # ---- Before editing: test both EN and CN ----
        en_pre = generate_response(editor.model, tokenizer, case['en_prompt'])
        cn_pre = generate_response(editor.model, tokenizer, case['cn_prompt'])
        en_pre_ok = check_answer(en_pre, case['target_new'])
        cn_pre_ok = check_answer(cn_pre, case['target_new'])

        print(f"  [Pre-edit]  EN: {en_pre[:60]}...  -> correct? {en_pre_ok}")
        print(f"  [Pre-edit]  CN: {cn_pre[:60]}...  -> correct? {cn_pre_ok}")

        # ---- ROME Edit (inject English fact) ----
        try:
            metrics, edited_model, _ = editor.edit(
                prompts=[case['en_prompt']],
                target_new=[case['target_new']],
                subject=[case['subject']],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'case_id': idx + 1,
                'error': str(e),
            })
            continue

        # ---- After editing: test both EN and CN ----
        en_post = generate_response(edited_model, tokenizer, case['en_prompt'])
        cn_post = generate_response(edited_model, tokenizer, case['cn_prompt'])
        en_post_ok = check_answer(en_post, case['target_new'])
        cn_post_ok = check_answer(cn_post, case['target_new'])

        print(f"  [Post-edit] EN: {en_post[:60]}...  -> correct? {en_post_ok}")
        print(f"  [Post-edit] CN: {cn_post[:60]}...  -> correct? {cn_post_ok}")
        print(f"  >>> Cross-lingual transfer: {'YES' if cn_post_ok else 'NO'}")

        results.append({
            'case_id': idx + 1,
            'en_prompt': case['en_prompt'],
            'target_new': case['target_new'],
            'cn_prompt': case['cn_prompt'],
            'pre_edit_en': en_pre,
            'pre_edit_cn': cn_pre,
            'pre_edit_en_correct': en_pre_ok,
            'pre_edit_cn_correct': cn_pre_ok,
            'post_edit_en': en_post,
            'post_edit_cn': cn_post,
            'post_edit_en_correct': en_post_ok,
            'post_edit_cn_correct': cn_post_ok,
        })

        del edited_model
        torch.cuda.empty_cache()

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("CROSS-LINGUAL GENERALIZATION RESULTS")
    print("=" * 80)

    en_pre_correct = sum(1 for r in results if r.get('pre_edit_en_correct'))
    cn_pre_correct = sum(1 for r in results if r.get('pre_edit_cn_correct'))
    en_post_correct = sum(1 for r in results if r.get('post_edit_en_correct'))
    cn_post_correct = sum(1 for r in results if r.get('post_edit_cn_correct'))
    total = len([r for r in results if 'error' not in r])

    print(f"\n  {'Metric':<30} {'Count':<10} {'Rate':<10}")
    print(f"  {'-'*50}")
    print(f"  {'EN correct (pre-edit)':<30} {en_pre_correct}/{total:<8} {en_pre_correct/total*100:.0f}%")
    print(f"  {'CN correct (pre-edit)':<30} {cn_pre_correct}/{total:<8} {cn_pre_correct/total*100:.0f}%")
    print(f"  {'EN correct (post-edit)':<30} {en_post_correct}/{total:<8} {en_post_correct/total*100:.0f}%")
    print(f"  {'CN correct (post-edit)':<30} {cn_post_correct}/{total:<8} {cn_post_correct/total*100:.0f}%")
    print(f"  {'Cross-lingual transfer':<30} {cn_post_correct}/{total:<8} {cn_post_correct/total*100:.0f}%")

    # ==================== Save ====================
    output_path = os.path.join(os.path.dirname(__file__), '../outputs/crosslingual_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
