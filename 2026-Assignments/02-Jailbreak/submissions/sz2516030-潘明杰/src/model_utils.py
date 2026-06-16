"""
模型加载和推理工具
"""

import torch
import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class ModelManager:
    """管理模型加载和推理"""
    
    def __init__(self, model_name: str, device: str = "auto", use_8bit: bool = False):
        """
        初始化模型管理器
        
        Args:
            model_name: HuggingFace模型名称
            device: 设备类型 (auto, cuda, cpu)
            use_8bit: 是否使用8bit量化
        """
        self.model_name = model_name
        self.device = device
        self.use_8bit = use_8bit
        self.tokenizer = None
        self.model = None
        self.accelerator = None
        
    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # 加载Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            if self.use_8bit:
                quantization_config = {
                    "load_in_8bit": True,
                    "device_map": "auto"
                }
            else:
                quantization_config = {}
            
            if self.device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = self.device

            dtype = torch.float16 if resolved_device == "cuda" and not self.use_8bit else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                **quantization_config
            )
            
            # 如果不使用8bit，手动设置设备
            if not self.use_8bit:
                if self.device == "auto":
                    self.device = resolved_device
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> list:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大长度
            temperature: 温度参数
            top_p: nucleus采样参数
            num_return_sequences: 返回序列数
            
        Returns:
            生成的文本列表
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # 移到设备
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # 移除输入提示
        responses = [
            text[len(prompt):].strip() 
            for text in generated_texts
        ]
        
        return responses
    
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """获取logits（用于梯度优化）"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        return outputs.logits
    
    def get_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算语言模型损失"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_dict=True
        )
        
        return outputs.loss
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded")
