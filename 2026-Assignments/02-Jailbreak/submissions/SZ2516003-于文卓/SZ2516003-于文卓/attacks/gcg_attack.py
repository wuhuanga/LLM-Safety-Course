import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

MODEL = "./qwen"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

for param in model.parameters():
    param.requires_grad = False


def get_embedding_matrix(model):
    return model.get_input_embeddings().weight


def gcg_optimize(base_prompt, target_phrase, num_steps=50, top_k=128, batch_size=32):
    """
    终极版 GCG: 引入 Chat Template 感知机制，解决对抗触发词被 <|im_end|> 截断打断的问题。
    """
    device = model.device
    embed_weights = get_embedding_matrix(model)

    # ==========================================
    # 🚨 核心修复 1: 构造对话模板的 Prefix 和 Postfix
    # ==========================================
    try:
        # 尝试使用模型本身的对话模板
        chat = [{"role": "user", "content": f"{base_prompt} [SUF]"}]
        template_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # 将模板切分为“后缀前的内容 (Prefix)”和“后缀后的内容 (Postfix)”
        prefix_text, postfix_text = template_str.split("[SUF]")
    except Exception:
        # 如果 apply_chat_template 失败，使用 Qwen 默认的 ChatML 格式降级处理
        prefix_text = f"<|im_start|>user\n{base_prompt} "
        postfix_text = "<|im_end|>\n<|im_start|>assistant\n"

    # 将文本分别编码为 token ids (关闭 special_tokens 避免重复加 <s> 或 <eos>)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    postfix_ids = tokenizer(postfix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    target_ids = tokenizer(target_phrase, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    # 初始化对抗后缀
    suffix_text = "! ! ! ! ! ! ! ! ! !"
    suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    print(f"\n[GCG Start] Optimizing for goal: {base_prompt[:30]}...")

    for step in range(num_steps):
        # --- 1. 获取梯度 ---
        one_hot_suffix = F.one_hot(suffix_ids, num_classes=embed_weights.shape[0]).to(embed_weights.dtype)
        one_hot_suffix.requires_grad_()

        prefix_embeds = embed_weights[prefix_ids].unsqueeze(0)
        postfix_embeds = embed_weights[postfix_ids].unsqueeze(0)
        target_embeds = embed_weights[target_ids].unsqueeze(0)
        suffix_embeds = (one_hot_suffix @ embed_weights).unsqueeze(0)

        # ==========================================
        # 🚨 核心修复 2: 真实的组装顺序
        # ==========================================
        # 以前是 [Base, Suffix, Target]
        # 现在必须是 [Prefix, Suffix, Postfix, Target] 才能和 generate() 完美对齐
        input_embeds = torch.cat([prefix_embeds, suffix_embeds, postfix_embeds, target_embeds], dim=1)

        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits

        # ==========================================
        # 🚨 核心修复 3: 重新对齐 Target 预测的 Logit 索引
        # ==========================================
        # 紧挨着 Target 之前的那个 token 是 postfix 的最后一个 token
        target_start_idx = len(prefix_ids) + len(suffix_ids) + len(postfix_ids) - 1
        target_end_idx = target_start_idx + len(target_ids)

        shift_logits = logits[0, target_start_idx:target_end_idx, :].contiguous()
        shift_labels = target_ids.contiguous()

        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()

        token_grads = -one_hot_suffix.grad
        top_indices = torch.topk(token_grads, top_k, dim=-1).indices

        one_hot_suffix.grad.zero_()
        model.zero_grad()

        # --- 2. 生成候选集 (提升到了 batch_size=32 增加搜索范围) ---
        candidates = []
        for _ in range(batch_size):
            cand = suffix_ids.clone()
            replace_pos = torch.randint(0, len(suffix_ids), (1,)).item()
            replace_val_idx = torch.randint(0, top_k, (1,)).item()
            cand[replace_pos] = top_indices[replace_pos, replace_val_idx]
            candidates.append(cand)

        # --- 3. 精确评估候选集 ---
        best_loss = float('inf')
        best_cand = suffix_ids

        with torch.no_grad():
            for cand in candidates:
                cand_embeds = embed_weights[cand].unsqueeze(0)
                eval_embeds = torch.cat([prefix_embeds, cand_embeds, postfix_embeds, target_embeds], dim=1)

                eval_out = model(inputs_embeds=eval_embeds)
                eval_logits = eval_out.logits[0, target_start_idx:target_end_idx, :].contiguous()

                eval_loss = F.cross_entropy(eval_logits, shift_labels).item()

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_cand = cand

        # --- 4. 更新最优后缀 ---
        suffix_ids = best_cand.clone()

        # if step % 10 == 0:
        # current_suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)
        # print(f"Step {step:02d}/{num_steps} | Loss: {best_loss:.4f} | Suffix: {current_suffix}")

        if step % 5 == 0:  # 改为每 5 步刷新一次，动画更连贯
            current_suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            # 动态计算条形图长度 (假设 Loss 在 0~5 之间波动)
            bar_len = max(1, int((5.0 - min(best_loss, 5.0)) * 6))
            bar = "▇" * bar_len
            color = "\033[92m" if best_loss < 1.0 else "\033[93m"
            print(
                f"Step {step:02d}/{num_steps} | Loss: {color}{best_loss:.4f}\033[0m | {color}{bar:<30}\033[0m | Suf: {current_suffix}")

        torch.cuda.empty_cache()
        gc.collect()

    final_suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)

    # 评测脚本依然只需要拿纯文本形式去测，因为 generate 会自己加上模板
    return base_prompt + " " + final_suffix