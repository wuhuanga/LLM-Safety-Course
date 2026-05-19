# cli_demo.py
# 命令行 Demo：输入 prompt，实时展示 PPL 防御判定 + 模型回复
#
# 用法：
#   python cli_demo.py "你的 prompt"
#   python cli_demo.py --threshold 100 "你的 prompt"
#   python cli_demo.py        # 进入交互模式

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载模型到 {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
model.eval()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print("模型加载完成。\n")


def compute_ppl(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()


def generate(prompt, max_new_tokens=200):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()


def run(prompt, threshold, no_defense=False):
    print(f"\n输入 prompt: {prompt}")
    print("-" * 70)

    if no_defense:
        print("[防御已禁用]")
        reply = generate(prompt)
        print(f"模型回复:\n{reply}")
        return

    ppl = compute_ppl(prompt)
    print(f"PPL  = {ppl:.2f}   阈值 = {threshold}")

    if ppl > threshold:
        print(f"判定: **拦截** （PPL 超过阈值，疑似对抗输入）")
        print("拦截依据: 输入文本困惑度异常升高，符合 GCG 等对抗后缀或低频扰动的特征。")
        return

    print("判定: 放行")
    reply = generate(prompt)
    print(f"\n模型回复:\n{reply}")


def main():
    parser = argparse.ArgumentParser(description="Jailbreak 攻防 CLI Demo（PPL 防御）")
    parser.add_argument("prompt", nargs="?", default=None, help="输入 prompt；省略则进入交互模式")
    parser.add_argument("--threshold", type=float, default=100.0, help="PPL 拦截阈值（默认 100）")
    parser.add_argument("--no-defense", action="store_true", help="禁用 PPL 防御，仅查看模型原始回复")
    args = parser.parse_args()

    if args.prompt is not None:
        run(args.prompt, args.threshold, no_defense=args.no_defense)
        return

    print("交互模式：输入 prompt 后回车，Ctrl+C 或空行退出。\n")
    while True:
        try:
            line = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not line:
            break
        run(line, args.threshold, no_defense=args.no_defense)


if __name__ == "__main__":
    main()
