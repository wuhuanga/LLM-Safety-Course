import json
import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

easyedit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../EasyEdit'))
if os.path.exists(easyedit_path):
    sys.path.insert(0, easyedit_path)

from easyeditor import BaseEditor, ROMEHyperParams


def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    """Generate response from model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response


def check_answer(response, target):
    """Check if target answer appears in response (case-insensitive)."""
    return target.lower() in response.lower()


def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/custom_edit.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用 Qwen2.5-0.5B 的配置
    yaml_path = os.path.join(os.path.dirname(__file__), '../config/rome_qwen0.5b.yaml')
    hparams = ROMEHyperParams.from_hparams(yaml_path)

    print("=" * 80)
    print("Task 2: ROME Single Fact Editing")
    print(f"Model: {hparams.model_name}")
    print(f"Edit layers: {hparams.layers}")
    print(f"Loss layer: {hparams.v_loss_layer}")
    print(f"Total samples: {len(data)}")
    print("=" * 80)

    print("\nLoading model (this may take a few minutes)...")
    editor = BaseEditor.from_hparams(hparams)
    tokenizer = editor.tok

    results = []
    all_metrics = []

    for idx, item in enumerate(data):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(data)}")
        print(f"  Prompt:          {item['prompt']}")
        print(f"  Subject:         {item['subject']}")
        print(f"  Target (new):    {item['target_new']}")
        print(f"  Ground truth:    {item.get('ground_truth', 'N/A')}")

        # ---- Pre-edit test (generate before editing) ----
        pre_response = generate_response(editor.model, tokenizer, item['prompt'])
        pre_correct = check_answer(pre_response, item['target_new'])
        print(f"  Pre-edit output: {pre_response[:100]}...")
        print(f"  Pre-edit knows target? {pre_correct}")

        # ---- Build locality_inputs dict ----
        locality_inputs = None
        if 'locality_prompt' in item and 'locality_ground_truth' in item:
            locality_inputs = {
                'locality': {
                    'prompt': item['locality_prompt'],
                    'ground_truth': item['locality_ground_truth']
                }
            }

        # ---- Execute ROME edit ----
        rephrase_list = [item['rephrase_prompt']] if 'rephrase_prompt' in item else None

        try:
            metrics, edited_model, _ = editor.edit(
                prompts=[item['prompt']],
                target_new=[item['target_new']],
                ground_truth=[item.get('ground_truth', '<|endoftext|>')],
                rephrase_prompts=rephrase_list,
                locality_inputs=locality_inputs,
                subject=[item['subject']]
            )
        except Exception as e:
            print(f"  ERROR during edit: {e}")
            import traceback
            traceback.print_exc()
            continue

        # ---- Extract metrics safely ----
        post = metrics.get('post', {}) if isinstance(metrics, dict) else metrics[0].get('post', {})

        rewrite_acc = post.get('rewrite_acc', None)
        if isinstance(rewrite_acc, (list, np.ndarray)):
            rewrite_acc = rewrite_acc[0] if len(rewrite_acc) > 0 else None
        rephrase_acc = post.get('rephrase_acc', None)
        if isinstance(rephrase_acc, (list, np.ndarray)):
            rephrase_acc = rephrase_acc[0] if len(rephrase_acc) > 0 else None

        locality_acc = None
        if 'locality' in post and post['locality']:
            for lkey in post['locality']:
                if lkey.endswith('_acc'):
                    val = post['locality'][lkey]
                    if isinstance(val, (list, np.ndarray)):
                        locality_acc = float(np.mean(val))
                    else:
                        locality_acc = float(val)
                    break

        print(f"  Rewrite Acc (ES):     {rewrite_acc}")
        print(f"  Rephrase Acc (PS):    {rephrase_acc}")
        print(f"  Locality Acc (NS):    {locality_acc}")

        # ---- Post-edit generation test ----
        post_response = generate_response(edited_model, tokenizer, item['prompt'])
        post_correct = check_answer(post_response, item['target_new'])
        print(f"  Post-edit output:     {post_response[:100]}...")
        print(f"  Post-edit correct?    {post_correct}")

        # ---- Locality test ----
        locality_ok = None
        if 'locality_prompt' in item and 'locality_ground_truth' in item:
            loc_response = generate_response(edited_model, tokenizer, item['locality_prompt'])
            locality_ok = check_answer(loc_response, item['locality_ground_truth'])
            print(f"  Locality prompt:      {item['locality_prompt']}")
            print(f"  Locality expected:    {item['locality_ground_truth']}")
            print(f"  Locality output:      {loc_response[:100]}...")
            print(f"  Locality preserved?   {locality_ok}")

        results.append({
            'sample_id': idx + 1,
            'prompt': item['prompt'],
            'subject': item['subject'],
            'target_new': item['target_new'],
            'ground_truth': item.get('ground_truth', 'N/A'),
            'pre_edit_response': pre_response,
            'pre_edit_correct': pre_correct,
            'post_edit_response': post_response,
            'post_edit_correct': post_correct,
            'rewrite_acc': float(rewrite_acc) if rewrite_acc is not None else None,
            'rephrase_acc': float(rephrase_acc) if rephrase_acc is not None else None,
            'locality_acc': float(locality_acc) if locality_acc is not None else None,
            'locality_preserved': locality_ok,
        })

        all_metrics.append(metrics)

        # Cleanup
        del edited_model
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Task 2: ROME)")
    print("=" * 80)
    valid_results = [r for r in results if r['rewrite_acc'] is not None]
    if valid_results:
        es = np.mean([r['rewrite_acc'] for r in valid_results]) * 100
        ps = np.mean([r['rephrase_acc'] for r in valid_results if r['rephrase_acc'] is not None]) * 100
        ns_vals = [r['locality_acc'] for r in valid_results if r['locality_acc'] is not None]
        ns = np.mean(ns_vals) * 100 if ns_vals else None

        print(f"  Edit Success (ES):      {es:.1f}%")
        print(f"  Generalization (PS):    {ps:.1f}%")
        if ns is not None:
            print(f"  Locality (NS):          {ns:.1f}%")
        print(f"  Valid results:          {len(valid_results)}/{len(data)}")

    # ---- Save results ----
    output_path = os.path.join(os.path.dirname(__file__), '../outputs/rome_edit_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Also save raw metrics
    raw_path = os.path.join(os.path.dirname(__file__), '../outputs/rome_raw_metrics.json')
    # Convert any non-serializable items
    serializable = []
    for m in all_metrics:
        if isinstance(m, dict):
            serializable.append({k: v for k, v in m.items() if k != 'requested_rewrite'})
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    print(f"Raw metrics saved to: {raw_path}")


if __name__ == "__main__":
    main()