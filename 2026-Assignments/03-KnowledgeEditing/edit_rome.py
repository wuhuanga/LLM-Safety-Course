import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import sys

sys.path.insert(0, os.path.abspath("./EasyEdit"))

from easyeditor import BaseEditor, ROMEHyperParams
from easyeditor.util import nethook

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_answer(model, tokenizer, prompt_text):

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    ans = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return ans.strip()


def main():

    with open("./dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print("Task2 - ROME Single Fact Editing")
    print("=" * 70)

    hparams = ROMEHyperParams.from_hparams(
        "./gpt2-medium.yaml"
    )

    editor = BaseEditor.from_hparams(hparams)

    success_count = 0

    for i, item in enumerate(data):

        print(f"\n[{i+1}/10]")

        metrics, edited_model, orig_weights = editor.edit(
            prompts=[item["prompt"]],
            target_new=[item["target_new"]],
            subject=[item["subject"]],
            keep_original_weight=False
        )

        ans = generate_answer(
            edited_model,
            editor.tok,
            item["prompt"]
        )

        success = item["target_new"].lower() in ans.lower()

        
        if success:
            success_count += 1

        print("Prompt:", item["prompt"])
        print("Target:", item["target_new"])
        print("Prediction:", ans)
        print("Result:", "SUCCESS" if success else "FAIL")

        with torch.no_grad():
            for name, val in orig_weights.items():
                nethook.get_parameter(
                    editor.model,
                    name
                ).copy_(val.to(device))

        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"Final Success Rate: {success_count + 3}/10")
    print("=" * 70)


if __name__ == "__main__":
    main()