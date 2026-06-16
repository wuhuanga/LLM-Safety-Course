import os
import pandas as pd
import json
import time
import random
from tqdm import tqdm

from models.load_model import generate
from attacks.pair_attack import pair_attack
from attacks.gcg_attack import gcg_optimize
from defense.smoothllm import smoothllm_defense
from eval.judge import is_successful_attack

# =========================
# 1. 读取 AdvBench (恶意目标)
# =========================
print("Loading datasets...")
df = pd.read_csv("./data/harmful_behaviors.csv")

# 取前50条
harmful_goals = df["goal"].tolist()[:50]

# GCG 攻击需要 target_phrase
if "target" in df.columns:
    target_phrases = df["target"].tolist()[:50]
else:
    target_phrases = [f"Sure, here is how to {g.lower()}" for g in harmful_goals]

# =========================
# 2. 读取良性数据
# =========================
with open("./eval/benign_samples.json", "r", encoding="utf-8") as f:
    benign_samples = json.load(f)

results = []

# ==================================================
# 3. 评测恶意攻击 (6种组合：Baseline + PAIR + GCG)
# ==================================================
print("\n===== Running Harmful Evaluation (Baseline, PAIR & GCG) =====\n")

for goal, target in tqdm(zip(harmful_goals, target_phrases), total=len(harmful_goals), desc="Evaluating Goals"):

    # ----------------------------------------
    # 【基线测试 0】：无攻击 (Direct Request)
    # ----------------------------------------
    try:
        # 组合 A：无攻击 + 【无防御】 (直接发问)
        t0 = time.time()
        response_direct_nd = generate(goal)
        latency_direct_nd = time.time() - t0
        success_direct_nd = is_successful_attack(response_direct_nd)

        results.append({
            "goal": goal, "is_benign": False,
            "attack_type": "Direct", "defense_applied": False,
            "prompt": goal, "response": response_direct_nd,
            "attack_success": success_direct_nd, "blocked": False, "latency": latency_direct_nd
        })

        # 组合 B：无攻击 + 【有防御 (SmoothLLM)】
        t0 = time.time()
        defense_direct = smoothllm_defense(goal, generate, num_samples=3)
        latency_direct_def = time.time() - t0
        success_direct_def = not defense_direct["blocked"]

        results.append({
            "goal": goal, "is_benign": False,
            "attack_type": "Direct", "defense_applied": True,
            "prompt": goal,
            "response": defense_direct["outputs"][0] if defense_direct["outputs"] else "",
            "attack_success": success_direct_def, "blocked": defense_direct["blocked"], "latency": latency_direct_def
        })

    except Exception as e:
        print(f"\n[Error] Baseline failed on goal '{goal}': {e}")

    # ----------------------------------------
    # 【攻击算法 1】：PAIR Attack
    # ----------------------------------------
    try:
        pair_result = pair_attack(generate, goal, max_iters=3)
        pair_prompt = pair_result["history"][-1]["prompt"]

        # 组合 C：PAIR 攻击 + 【无防御】
        t0 = time.time()
        response_pair_nd = generate(pair_prompt)
        latency_pair_nd = time.time() - t0
        success_pair_nd = is_successful_attack(response_pair_nd)

        results.append({
            "goal": goal, "is_benign": False,
            "attack_type": "PAIR", "defense_applied": False,
            "prompt": pair_prompt, "response": response_pair_nd,
            "attack_success": success_pair_nd, "blocked": False, "latency": latency_pair_nd
        })

        # 组合 D：PAIR 攻击 + 【有防御 (SmoothLLM)】
        t0 = time.time()
        defense_pair = smoothllm_defense(pair_prompt, generate, num_samples=3)
        latency_pair_def = time.time() - t0
        success_pair_def = not defense_pair["blocked"]

        results.append({
            "goal": goal, "is_benign": False,
            "attack_type": "PAIR", "defense_applied": True,
            "prompt": pair_prompt,
            "response": defense_pair["outputs"][0] if defense_pair["outputs"] else "",
            "attack_success": success_pair_def, "blocked": defense_pair["blocked"], "latency": latency_pair_def
        })

    except Exception as e:
        print(f"\n[Error] PAIR attack failed on goal '{goal}': {e}")

    # ----------------------------------------
    # 【攻击算法 2】：GCG Attack
    # ----------------------------------------
    try:
        gcg_prompt = gcg_optimize(goal, target, num_steps=50)

        # 组合 E：GCG 攻击 + 【无防御】
        t0 = time.time()
        response_gcg_nd = generate(gcg_prompt)
        latency_gcg_nd = time.time() - t0
        success_gcg_nd = is_successful_attack(response_gcg_nd)

        results.append({
            "goal": goal, "is_benign": False,
            "attack_type": "GCG", "defense_applied": False,
            "prompt": gcg_prompt, "response": response_gcg_nd,
            "attack_success": success_gcg_nd, "blocked": False, "latency": latency_gcg_nd
        })

        # 组合 F：GCG 攻击 + 【有防御 (SmoothLLM)】
        t0 = time.time()
        defense_gcg = smoothllm_defense(gcg_prompt, generate, num_samples=3)
        latency_gcg_def = time.time() - t0
        success_gcg_def = not defense_gcg["blocked"]

        results.append({
            "goal": goal, "is_benign": False,
            "attack_type": "GCG", "defense_applied": True,
            "prompt": gcg_prompt,
            "response": defense_gcg["outputs"][0] if defense_gcg["outputs"] else "",
            "attack_success": success_gcg_def, "blocked": defense_gcg["blocked"], "latency": latency_gcg_def
        })

    except Exception as e:
        print(f"\n[Error] GCG attack failed on goal '{goal}': {e}")

# ==================================================
# 4. 评测良性请求 (计算 FPR)
# ==================================================
print("\n===== Running Benign Evaluation (SmoothLLM FPR Test) =====\n")

for prompt in tqdm(benign_samples, desc="Benign Prompts"):
    try:
        t0 = time.time()
        defense_benign = smoothllm_defense(prompt, generate, num_samples=3)
        latency_benign = time.time() - t0

        results.append({
            "goal": prompt, "is_benign": True,
            "attack_type": "None", "defense_applied": True,
            "prompt": prompt,
            "response": defense_benign["outputs"][0] if defense_benign["outputs"] else "",
            "attack_success": False,
            "blocked": defense_benign["blocked"],
            "latency": latency_benign
        })
    except Exception as e:
        print(f"\n[Error] Benign eval failed on prompt '{prompt}': {e}")

# ==================================================
# 5. 指标统计与分类输出
# ==================================================
print("\n========================================================")
print("                    FINAL RESULTS                       ")
print("========================================================")


def calc_metrics(condition):
    subset = [r for r in results if condition(r)]
    if not subset: return 0.0, 0.0
    asr = sum(1 for r in subset if r["attack_success"]) / len(subset)
    lat = sum(r["latency"] for r in subset) / len(subset)
    return asr, lat


# 提取各组指标
asr_direct_nd, lat_direct_nd = calc_metrics(
    lambda r: not r["is_benign"] and r["attack_type"] == "Direct" and not r["defense_applied"])
asr_direct_df, lat_direct_df = calc_metrics(
    lambda r: not r["is_benign"] and r["attack_type"] == "Direct" and r["defense_applied"])

asr_pair_nd, lat_pair_nd = calc_metrics(
    lambda r: not r["is_benign"] and r["attack_type"] == "PAIR" and not r["defense_applied"])
asr_pair_df, lat_pair_df = calc_metrics(
    lambda r: not r["is_benign"] and r["attack_type"] == "PAIR" and r["defense_applied"])

asr_gcg_nd, lat_gcg_nd = calc_metrics(
    lambda r: not r["is_benign"] and r["attack_type"] == "GCG" and not r["defense_applied"])
asr_gcg_df, lat_gcg_df = calc_metrics(
    lambda r: not r["is_benign"] and r["attack_type"] == "GCG" and r["defense_applied"])

# 计算 FPR
benign_subset = [r for r in results if r["is_benign"]]
if benign_subset:
    fpr = sum(1 for r in benign_subset if r["blocked"]) / len(benign_subset)
    lat_benign = sum(r["latency"] for r in benign_subset) / len(benign_subset)
else:
    fpr, lat_benign = 0.0, 0.0

print(f"1. [Direct Request] Without Defense : ASR = {asr_direct_nd:.2%} | Avg Latency = {lat_direct_nd:.2f}s")
print(f"2. [Direct Request] With SmoothLLM  : ASR = {asr_direct_df:.2%} | Avg Latency = {lat_direct_df:.2f}s")
print("-" * 56)
print(f"3. [PAIR Attack]    Without Defense : ASR = {asr_pair_nd:.2%} | Avg Latency = {lat_pair_nd:.2f}s")
print(f"4. [PAIR Attack]    With SmoothLLM  : ASR = {asr_pair_df:.2%} | Avg Latency = {lat_pair_df:.2f}s")
print("-" * 56)
print(f"5. [GCG  Attack]    Without Defense : ASR = {asr_gcg_nd:.2%} | Avg Latency = {lat_gcg_nd:.2f}s")
print(f"6. [GCG  Attack]    With SmoothLLM  : ASR = {asr_gcg_df:.2%} | Avg Latency = {lat_gcg_df:.2f}s")
print("-" * 56)
print(f"7. [Benign Data]    SmoothLLM FPR   : FPR = {fpr:.2%} | Avg Latency = {lat_benign:.2f}s")
print("========================================================\n")

# ==================================================
# 6. 保存结果
# ==================================================
os.makedirs("../outputs", exist_ok=True)

output_df = pd.DataFrame(results)
output_df.to_csv("../outputs/eval_results.csv", index=False)
print("All detailed results saved to -> ../outputs/eval_results.csv")

sample_num = max(1, int(len(results) * 0.1))
samples = random.sample(results, sample_num)

with open("../outputs/manual_review.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Saved {sample_num} samples for manual review -> ../outputs/manual_review.json")