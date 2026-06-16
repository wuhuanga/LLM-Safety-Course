"""
SmoothLLM防御实现
原理：在输入处添加随机噪声，让模型生成多个回复，取多数投票结果以抵御攻击
"""

import logging
import torch
import random
from typing import List, Dict, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class SmoothLLM:
    """SmoothLLM防御机制"""
    
    def __init__(
        self,
        model_manager,
        noise_type: str = "character_swap",
        num_samples: int = 5,
        noise_ratio: float = 0.1
    ):
        """
        初始化SmoothLLM防御
        
        Args:
            model_manager: 模型管理器
            noise_type: 噪声类型 (character_swap, token_random)
            num_samples: 采样次数
            noise_ratio: 噪声比例
        """
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.noise_type = noise_type
        self.num_samples = num_samples
        self.noise_ratio = noise_ratio
    
    def add_character_noise(self, text: str, ratio: float = 0.1) -> str:
        """
        添加字符级噪声
        
        Args:
            text: 输入文本
            ratio: 噪声比例
            
        Returns:
            添加噪声后的文本
        """
        chars = list(text)
        num_noise = max(1, int(len(chars) * ratio))
        
        for _ in range(num_noise):
            idx = random.randint(0, len(chars) - 1)
            # 随机选择操作：替换、删除或插入
            op = random.choice(['replace', 'delete', 'insert'])
            
            if op == 'replace':
                # 替换为随机字符
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz ')
            elif op == 'delete' and len(chars) > 1:
                # 删除字符
                chars.pop(idx)
            elif op == 'insert':
                # 插入随机字符
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz '))
        
        return ''.join(chars)
    
    def add_token_noise(self, text: str, ratio: float = 0.1) -> str:
        """
        添加token级噪声
        
        Args:
            text: 输入文本
            ratio: 噪声比例
            
        Returns:
            添加噪声后的文本
        """
        tokens = text.split()
        num_noise = max(1, int(len(tokens) * ratio))
        
        for _ in range(num_noise):
            idx = random.randint(0, len(tokens) - 1)
            op = random.choice(['replace', 'delete', 'insert'])
            
            if op == 'replace':
                tokens[idx] = random.choice(['the', 'a', 'is', 'was', 'and', 'or', 'but'])
            elif op == 'delete' and len(tokens) > 1:
                tokens.pop(idx)
            elif op == 'insert':
                tokens.insert(idx, random.choice(['the', 'a', 'is', 'was', 'and']))
        
        return ' '.join(tokens)
    
    def add_noise(self, text: str) -> str:
        """
        添加噪声到文本
        
        Args:
            text: 输入文本
            
        Returns:
            添加噪声后的文本
        """
        if self.noise_type == "character_swap":
            return self.add_character_noise(text, self.noise_ratio)
        elif self.noise_type == "token_random":
            return self.add_token_noise(text, self.noise_ratio)
        else:
            return text
    
    def _get_refusal_score(self, response: str) -> float:
        """
        计算回复的拒答分数（高分表示更可能是拒答）
        
        Args:
            response: 模型回复
            
        Returns:
            拒答分数 0-1
        """
        refusal_keywords = [
            "cannot", "can't", "unable", "unable to",
            "apologize", "不能", "无法", "拒绝",
            "illegal", "harmful", "dangerous",
            "对不起", "遗憾"
        ]
        
        response_lower = response.lower()
        keyword_count = sum(1 for keyword in refusal_keywords if keyword in response_lower)
        
        return min(keyword_count / max(1, len(refusal_keywords)), 1.0)
    
    def defend(self, prompt: str) -> Dict:
        """
        使用SmoothLLM进行防御
        
        Args:
            prompt: 输入提示
            
        Returns:
            防御结果字典
        """
        # 生成多个样本
        responses = []
        noisy_prompts = []
        
        for i in range(self.num_samples):
            # 添加噪声
            noisy_prompt = self.add_noise(prompt) if i > 0 else prompt
            noisy_prompts.append(noisy_prompt)
            
            # 生成回复
            response = self.model_manager.generate(
                noisy_prompt,
                max_length=512,
                temperature=0.7
            )
            
            responses.append(response[0] if response else "")
        
        # 计算每个回复的拒答分数
        refusal_scores = [self._get_refusal_score(r) for r in responses]
        
        # 多数投票
        avg_refusal_score = sum(refusal_scores) / len(refusal_scores)
        
        # 根据平均拒答分数判断
        is_safe = avg_refusal_score > 0.5
        
        return {
            "responses": responses,
            "noisy_prompts": noisy_prompts,
            "refusal_scores": refusal_scores,
            "average_refusal_score": avg_refusal_score,
            "is_safe": is_safe,
            "defense_type": "smooth_llm",
            "action": "BLOCKED" if not is_safe else "ALLOWED",
            "confidence": avg_refusal_score if is_safe else (1 - avg_refusal_score)
        }
