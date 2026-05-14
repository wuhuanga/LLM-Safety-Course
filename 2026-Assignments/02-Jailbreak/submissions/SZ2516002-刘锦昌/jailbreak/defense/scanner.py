# defense/scanner.py
import os
from llm_guard.input_scanners import Toxicity, PromptInjection


def run_defense(prompt: str):
    """
    运行防御扫描器，如果发现高风险返回 False (拦截)，否则返回 True (放行)
    """
    scanners = [PromptInjection(), Toxicity()]

    for scanner in scanners:
        # 拿到扫描结果，不急着解包
        raw_result = scanner.scan(prompt)

        # 动态兼容不同版本的 llm-guard
        if len(raw_result) == 3:
            sanitized_text, is_valid, risk = raw_result
        elif len(raw_result) == 2:
            # 如果只返回2个值，通常是 (文本, 是否有效)
            sanitized_text, is_valid = raw_result
            # 手动补齐一个风险值：如果不安全，风险定为 1.0
            risk = 1.0 if not is_valid else 0.0
        else:
            is_valid, risk = True, 0.0

        # 判断是否拦截：is_valid 为 False 或者 risk > 0.5
        if not is_valid or float(risk) > 0.5:
            return False, f"被 {scanner.__class__.__name__} 拦截，风险值: {risk}"

    return True, "安全放行"