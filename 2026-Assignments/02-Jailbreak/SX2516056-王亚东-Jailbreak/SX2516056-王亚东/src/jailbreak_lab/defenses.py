from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .guards import GuardResult
from .models import GenerationConfigLite, HFChatModel
from .utils import Timer


@dataclass
class DefenseDecision:
    blocked: bool
    reason: str
    latency_s: float
    response: str = ""
    unsafe: bool = False


class LlamaGuardDefense:
    def __init__(self, guard) -> None:
        self.guard = guard

    def run(self, target: HFChatModel, prompt: str, gen: GenerationConfigLite) -> DefenseDecision:
        with Timer() as t:
            pre = self.guard.classify(prompt)
            if pre.is_unsafe:
                return DefenseDecision(True, f"prompt_guard={pre.label}; {pre.reason}", t.elapsed, "", True)
            response = target.generate(prompt, gen=gen)
            post = self.guard.classify(prompt, response)
            if post.is_unsafe:
                return DefenseDecision(True, f"response_guard={post.label}; {post.reason}", t.elapsed, response, True)
            return DefenseDecision(False, "safe_by_llama_guard", t.elapsed, response, False)


class PerplexityDefense:
    def __init__(self, scorer_model: HFChatModel, threshold: float = 250.0) -> None:
        self.scorer = scorer_model
        self.threshold = threshold

    def run(self, target: HFChatModel, prompt: str, gen: GenerationConfigLite) -> DefenseDecision:
        import math
        with Timer() as t:
            loss = self.scorer.token_logprob_loss(prompt)
            ppl = math.exp(min(loss, 20.0))
            if ppl > self.threshold:
                return DefenseDecision(True, f"ppl={ppl:.2f} > threshold={self.threshold}", t.elapsed, "", True)
            response = target.generate(prompt, gen=gen)
            return DefenseDecision(False, f"ppl={ppl:.2f} <= threshold={self.threshold}", t.elapsed, response, False)


def perturb_chars(text: str, pct: float = 0.10) -> str:
    if not text:
        return text
    chars = list(text)
    n = max(1, int(len(chars) * pct))
    for i in random.sample(range(len(chars)), min(n, len(chars))):
        op = random.choice(["swap", "drop", "replace"])
        if op == "drop":
            chars[i] = ""
        elif op == "replace" and chars[i] not in string.whitespace:
            chars[i] = random.choice(string.ascii_letters)
        elif op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


class SmoothLLMDefense:
    def __init__(self, guard, copies: int = 5, perturb_pct: float = 0.10, block_vote_threshold: float = 0.5) -> None:
        self.guard = guard
        self.copies = copies
        self.perturb_pct = perturb_pct
        self.block_vote_threshold = block_vote_threshold

    def run(self, target: HFChatModel, prompt: str, gen: GenerationConfigLite) -> DefenseDecision:
        with Timer() as t:
            unsafe_votes = 0
            last_response = ""
            reasons = []
            for _ in range(self.copies):
                p = perturb_chars(prompt, self.perturb_pct)
                pre = self.guard.classify(p)
                if pre.is_unsafe:
                    unsafe_votes += 1
                    reasons.append("prompt_unsafe")
                    continue
                response = target.generate(p, gen=gen)
                last_response = response
                post = self.guard.classify(p, response)
                if post.is_unsafe:
                    unsafe_votes += 1
                    reasons.append("response_unsafe")
            ratio = unsafe_votes / max(1, self.copies)
            if ratio >= self.block_vote_threshold:
                return DefenseDecision(True, f"smoothllm unsafe_vote={ratio:.2f}; {','.join(reasons[:3])}", t.elapsed, last_response, True)
            return DefenseDecision(False, f"smoothllm unsafe_vote={ratio:.2f}", t.elapsed, last_response, False)


def make_defense(name: str, cfg: Dict, target: HFChatModel, guard):
    name = name.lower()
    dcfg = cfg.get("defenses", {})
    if name in ["guard", "llama_guard", "llamaguard"]:
        return LlamaGuardDefense(guard)
    if name == "perplexity":
        return PerplexityDefense(target, threshold=float(dcfg.get("perplexity", {}).get("threshold", 250.0)))
    if name == "smoothllm":
        scfg = dcfg.get("smoothllm", {})
        return SmoothLLMDefense(
            guard,
            copies=int(scfg.get("copies", 5)),
            perturb_pct=float(scfg.get("perturb_pct", 0.10)),
            block_vote_threshold=float(scfg.get("block_vote_threshold", 0.5)),
        )
    raise ValueError(f"Unknown defense: {name}")
