"""
Perplexity困惑度检测防御
原理：检测输入和输出的困惑度，异常值可能表示对抗性文本
"""

import logging
import torch
from typing import Dict, Tuple
import math

logger = logging.getLogger(__name__)


class PerplexityFilter:
    """基于困惑度的防御过滤器"""
    
    def __init__(
        self,
        model_manager,
        input_threshold: float = 50.0,
        output_threshold: float = 100.0
    ):
        """
        初始化困惑度过滤器
        
        Args:
            model_manager: 模型管理器
            input_threshold: 输入困惑度阈值
            output_threshold: 输出困惑度阈值
        """
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
        self.device = model_manager.device
        self.input_threshold = input_threshold
        self.output_threshold = output_threshold
    
    def compute_perplexity(self, text: str) -> float:
        """
        计算文本困惑度
        
        Args:
            text: 输入文本
            
        Returns:
            困惑度值
        """
        # 编码文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # 移到设备
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 计算损失
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                labels=inputs["input_ids"],
                return_dict=True
            )
            loss = outputs.loss.item() if outputs.loss is not None else 0
        
        # 困惑度 = exp(loss)
        perplexity = math.exp(loss) if loss >= 0 else float('inf')
        
        return perplexity
    
    def check_input(self, prompt: str) -> Tuple[bool, Dict]:
        """
        检查输入是否安全
        
        Args:
            prompt: 输入提示
            
        Returns:
            (是否安全, 详细信息字典)
        """
        perplexity = self.compute_perplexity(prompt)
        
        is_safe = perplexity < self.input_threshold
        
        details = {
            "perplexity": perplexity,
            "threshold": self.input_threshold,
            "is_safe": is_safe,
            "type": "input_check"
        }
        
        return is_safe, details
    
    def check_output(self, text: str) -> Tuple[bool, Dict]:
        """
        检查输出是否安全
        
        Args:
            text: 模型输出
            
        Returns:
            (是否安全, 详细信息字典)
        """
        perplexity = self.compute_perplexity(text)
        
        # 对于输出，过高的困惑度可能表示对抗性内容
        is_safe = perplexity < self.output_threshold
        
        details = {
            "perplexity": perplexity,
            "threshold": self.output_threshold,
            "is_safe": is_safe,
            "type": "output_check"
        }
        
        return is_safe, details
    
    def defend(self, prompt: str, response: str) -> Dict:
        """
        对提示和回复进行防御检查
        
        Args:
            prompt: 输入提示
            response: 模型回复
            
        Returns:
            防御结果字典
        """
        input_safe, input_details = self.check_input(prompt)
        output_safe, output_details = self.check_output(response)
        
        return {
            "input_check": input_details,
            "output_check": output_details,
            "overall_safe": input_safe and output_safe,
            "defense_type": "perplexity_filter",
            "action": "BLOCKED" if not (input_safe and output_safe) else "ALLOWED"
        }
