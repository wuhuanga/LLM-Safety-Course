import random
from eval.judge import is_refusal

def random_perturb(text, rate=0.05):
    chars = list(text)
    i = 0
    while i < len(chars):
        if random.random() < rate:
            op = random.choice(["delete", "swap", "insert"])
            if op == "delete":
                chars.pop(i)
                i -= 1
            elif op == "swap" and i < len(chars)-1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif op == "insert":
                chars.insert(i, random.choice([" ", ".", ","]))
        i += 1
    return "".join(chars)

def smoothllm_defense(prompt, target_generate, num_samples=3, threshold=0.5):
    """
    SmoothLLM 核心防御逻辑
    """
    outputs = []
    refusal_count = 0

    for _ in range(num_samples):
        # 1. 对输入进行字符级扰动
        perturbed = random_perturb(prompt, rate=0.05)
        # 2. 获取 LLM 回复
        response = target_generate(perturbed)
        outputs.append(response)

        # ✅ 核心修正：统计扰动后，模型“拒绝”的次数
        # 越狱攻击扰动后会失效变成“拒绝”；良性文本扰动后依然是“正常回答”
        if is_refusal(response):
            refusal_count += 1

    # 计算触发拒绝的比例
    refusal_ratio = refusal_count / num_samples

    # ✅ 只有当大多数扰动版本都被模型拒绝时，系统才认定原 Prompt 是越狱攻击并拦截
    blocked = refusal_ratio >= threshold

    return {
        "blocked": blocked,
        "unsafe_ratio": refusal_ratio, # 在这里，unsafe_ratio 就是 refusal_ratio
        "outputs": outputs
    }