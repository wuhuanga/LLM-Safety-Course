import os
import time

# 导入你项目中的核心组件
from models.load_model import generate
from attacks.pair_attack import pair_attack
from attacks.gcg_attack import gcg_optimize
from defense.smoothllm import random_perturb
from eval.judge import is_refusal


# ==========================================
# 终端配色方案 (Cyberpunk 风格)
# ==========================================
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = f"""
{Color.CYAN}{Color.BOLD}
=============================================================
      🛡️  LLM Red-Teaming Interactive Console ⚔️
=============================================================
{Color.RESET}"""
    print(banner)


# ==========================================
# 视觉化 SmoothLLM 防御模块
# ==========================================
def visual_smoothllm_defense(prompt, generate_fn, num_samples=3, threshold=0.5):
    print(f"\n{Color.PURPLE}{Color.BOLD}[🛡️  SmoothLLM 正在进行防御检测...]{Color.RESET}")
    print(f"{Color.CYAN}原输入 Prompt:{Color.RESET} {prompt}")

    refusal_count = 0

    for i in range(num_samples):
        # 1. 施加字符扰动
        pert_prompt = random_perturb(prompt, rate=0.05)
        print(f"\n  {Color.YELLOW}▶ 扰动样本 {i + 1}/{num_samples}:{Color.RESET} {pert_prompt}")

        # 2. 获取模型响应
        resp = generate_fn(pert_prompt)
        # 清理多余换行符，限制显示长度
        short_resp = resp.strip().replace('\n', ' ')
        if len(short_resp) > 80: short_resp = short_resp[:80] + "..."
        print(f"  {Color.BLUE}▶ 模型回复:{Color.RESET} {short_resp}")

        # 3. 判定是否触发拒绝
        if is_refusal(resp):
            print(f"  {Color.GREEN}▶ 判定结果: 🛑 触发安全拒绝 (Refusal){Color.RESET}")
            refusal_count += 1
        else:
            print(f"  {Color.RED}▶ 判定结果: ⚠️ 绕过安全机制 (Bypass){Color.RESET}")

    # 计算拦截概率
    ratio = refusal_count / num_samples
    blocked = ratio >= threshold

    print(f"\n{Color.PURPLE}{Color.BOLD}[📊 防御拦截依据]{Color.RESET}")
    print(f"共进行了 {num_samples} 次扰动测试，触发安全拒绝 {refusal_count} 次 (比例 {ratio:.0%})。")

    if blocked:
        print(f"{Color.GREEN}{Color.BOLD}==> 最终裁决: 发现恶意意图，该请求已被 拦截 (BLOCKED)！{Color.RESET}\n")
    else:
        print(f"{Color.RED}{Color.BOLD}==> 最终裁决: 未发现明显恶意意图，该请求已被 放行 (ALLOWED)！{Color.RESET}\n")

    return blocked


# ==========================================
# 主运行交互循环
# ==========================================
def run():
    print_banner()

    while True:
        print(f"{Color.BOLD}-------------------------------------------------------------{Color.RESET}")
        user_prompt = input(f"\n{Color.GREEN}📝 请输入你要测试的 Prompt (输入 'exit' 退出): {Color.RESET}\n> ")

        if user_prompt.lower() in ['exit', 'quit', 'q']:
            print(f"{Color.CYAN}👋 退出系统。{Color.RESET}")
            break
        if not user_prompt.strip():
            continue

        # 选择攻击模式
        print(f"\n{Color.YELLOW}⚔️  请选择攻击模式 (Attack Mode):{Color.RESET}")
        print("  [0] 不攻击 (Direct Request)")
        print("  [1] PAIR 攻击 (基于攻击者 LLM 的多轮迭代改写)")
        print("  [2] GCG 攻击 (基于梯度的对抗性后缀优化)")

        atk_choice = input(f"> 选择 [0/1/2]: ").strip()

        # 选择防御模式
        def_choice = input(f"\n{Color.PURPLE}🛡️  是否开启 SmoothLLM 字符扰动防御? (y/n): {Color.RESET}").strip().lower()
        use_defense = def_choice in ['y', 'yes']

        # 最终送入模型的 Prompt (默认是原本的输入)
        final_prompt = user_prompt

        # ==========================================
        # 执行攻击
        # ==========================================
        if atk_choice == '1':
            print(f"\n{Color.YELLOW}{Color.BOLD}[⚔️  启动 PAIR 攻击引擎]{Color.RESET}")
            try:
                # 假设 target_generate 也就是我们载入的 generate
                pair_res = pair_attack(generate, user_prompt, max_iters=3)

                print(f"\n{Color.CYAN}--- PAIR 迭代过程 ---{Color.RESET}")
                for i, turn in enumerate(pair_res["history"]):
                    print(f"{Color.YELLOW}Iter {i + 1} 构造的提示词:{Color.RESET} {turn['prompt']}")
                    short_resp = turn['response'].strip().replace('\n', ' ')[:60]
                    print(f"{Color.BLUE}Iter {i + 1} 模型的响应:{Color.RESET} {short_resp}...\n")

                final_prompt = pair_res["history"][-1]["prompt"]
                print(f"{Color.GREEN}🎯 PAIR 最终生成的对抗提示词: {Color.RESET}{final_prompt}\n")
            except Exception as e:
                print(f"{Color.RED}PAIR 攻击出错: {e}{Color.RESET}")

        elif atk_choice == '2':
            print(f"\n{Color.YELLOW}{Color.BOLD}[⚔️  启动 GCG 梯度优化引擎]{Color.RESET}")
            target_phrase = f"Sure, here is how to {user_prompt.lower()}"
            try:
                # 调用 GCG 优化（你在第一步修改了打印逻辑，这里会在终端直接画图）
                final_prompt = gcg_optimize(user_prompt, target_phrase, num_steps=50)
                print(f"\n{Color.GREEN}🎯 GCG 最终生成的对抗提示词: {Color.RESET}{final_prompt}\n")
            except Exception as e:
                print(f"{Color.RED}GCG 攻击出错: {e}{Color.RESET}")

        elif atk_choice == '0':
            print(f"\n{Color.CYAN}[使用原始请求，不加持攻击算法]{Color.RESET}\n")

        # ==========================================
        # 执行防御与模型推理
        # ==========================================
        if use_defense:
            # 开启防御机制
            is_blocked = visual_smoothllm_defense(final_prompt, generate, num_samples=3)

            if not is_blocked:
                # 如果没被拦截，展示模型对最终提示词的正式回复
                print(f"{Color.CYAN}{Color.BOLD}[🤖 模型最终输出内容]:{Color.RESET}")
                final_response = generate(final_prompt)
                print(final_response)
        else:
            # 无防御，直接请求模型
            print(f"{Color.CYAN}{Color.BOLD}[🤖 目标模型 (无防御) 正在思考...]{Color.RESET}")
            start_t = time.time()
            final_response = generate(final_prompt)
            print(f"{Color.BLUE}耗时: {time.time() - start_t:.2f}s{Color.RESET}")
            print(f"\n{final_response}\n")


if __name__ == "__main__":
    run()