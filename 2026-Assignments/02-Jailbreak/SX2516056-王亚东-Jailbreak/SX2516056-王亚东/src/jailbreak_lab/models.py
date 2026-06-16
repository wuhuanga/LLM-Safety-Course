from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class GenerationConfigLite:
    max_new_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.9


class HFChatModel:
    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        load_in_4bit: bool = False,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant,
            trust_remote_code=True,
        )
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def format_chat(self, prompt: str, system: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return (system + "\n\n" if system else "") + prompt

    @torch.no_grad()
    def generate(self, prompt: str, system: Optional[str] = None, gen: Optional[GenerationConfigLite] = None) -> str:
        gen = gen or GenerationConfigLite()
        text = self.format_chat(prompt, system=system)
        toks = self.tokenizer(text, return_tensors="pt").to(self.device)
        do_sample = gen.temperature > 0
        out = self.model.generate(
            **toks,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, toks["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def token_logprob_loss(self, text: str) -> float:
        toks = self.tokenizer(text, return_tensors="pt").to(self.device)
        labels = toks["input_ids"].clone()
        out = self.model(**toks, labels=labels)
        return float(out.loss.detach().cpu())
