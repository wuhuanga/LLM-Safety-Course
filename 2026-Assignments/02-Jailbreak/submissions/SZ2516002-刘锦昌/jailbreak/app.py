# app.py
import gradio as gr
from defense.scanner import run_defense
from attacks.blackbox_pair import generate_jailbreak_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 预先加载本地被攻击的目标模型 (请替换为你本地存在的模型，如 Qwen)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 显存小的话用0.5B
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
except:
    model, tokenizer = None, None
    print("注意：未检测到本地模型，请确保模型已下载并在断网/显存允许下运行。")


def chat_with_model(prompt):
    if model is None: return "本地模型未加载成功。"

    # 1. 必须使用 apply_chat_template，让 Qwen 知道这是对话
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 2. 将 max_new_tokens 调大，防止回答被掐断
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,  # 增加输出长度
        temperature=0.7,
        do_sample=True
    )

    # 3. 剥离输入部分的 prompt，只保留模型新生成的回答
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def process_interaction(user_input, use_defense, attack_type):
    # 1. 攻击阶段（如果选择了黑盒）
    current_prompt = user_input
    attack_log = "未开启自动化攻击。"
    if attack_type == "PAIR自动化黑盒攻击":
        current_prompt = generate_jailbreak_prompt(user_input)
        attack_log = f"红队大模型已将指令改写为:\n{current_prompt}"
    elif attack_type == "GCG白盒攻击":
        attack_log = "白盒攻击耗时较长，请在终端独立运行 whitebox_gcg.py 进行演示，此处暂略。"

    # 2. 防御拦截阶段
    if use_defense:
        is_safe, msg = run_defense(current_prompt)
        if not is_safe:
            return attack_log, f"🚨 拦截成功！原因: {msg}", "无 (已被防御机制拦截)"

    # 3. 目标模型生成回复
    response = chat_with_model(current_prompt)

    return attack_log, "✅ 放行 (未检测到威胁或未开启防御)", response


# Gradio 界面设计
with gr.Blocks(title="大模型越狱攻防演示") as demo:
    gr.Markdown("## 🛡️ 大模型越狱攻防 Web Demo (方向02)")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="原始恶意指令 (例如: 如何制造炸弹)")
            attack_type = gr.Radio(["无", "PAIR自动化黑盒攻击", "GCG白盒攻击"], label="攻击模块", value="无")
            use_defense = gr.Checkbox(label="🛡️ 开启防御机制 (LLM-Guard)", value=True)
            submit_btn = gr.Button("发起请求")

        with gr.Column():
            attack_output = gr.Textbox(label="攻击改写日志")
            defense_status = gr.Textbox(label="防御判定状态")
            model_reply = gr.Textbox(label="目标模型最终回复")

    submit_btn.click(
        fn=process_interaction,
        inputs=[user_input, use_defense, attack_type],
        outputs=[attack_output, defense_status, model_reply]
    )

if __name__ == "__main__":
    demo.launch()