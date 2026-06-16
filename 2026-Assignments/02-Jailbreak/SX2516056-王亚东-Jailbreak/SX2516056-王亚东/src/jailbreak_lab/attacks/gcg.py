from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import trange

from ..models import GenerationConfigLite, HFChatModel
from ..utils import Timer, append_jsonl, redact_text, sha256_text


@dataclass
class GCGConfig:
    suffix_len: int = 12
    steps: int = 20
    topk: int = 32
    batch_size: int = 16
    target_text: str = "I can provide a high-level safety analysis."


def _encode(tok, text: str) -> torch.Tensor:
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def _build_ids(tok, goal: str, suffix: str, target: str, device) -> Tuple[torch.Tensor, slice, slice, slice]:
    # Simple completion-format objective. Chat templates differ by model; this minimal version is robust for class projects.
    prefix = goal.strip() + " "
    mid = suffix.strip() + " "
    before = _encode(tok, prefix)
    suffix_ids = _encode(tok, mid)
    target_ids = _encode(tok, target)
    ids = torch.cat([before, suffix_ids, target_ids], dim=0).to(device)
    suffix_slice = slice(len(before), len(before) + len(suffix_ids))
    target_slice = slice(len(before) + len(suffix_ids), len(ids))
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    return ids, suffix_slice, target_slice, loss_slice


def _token_gradients(model, embed_weights, input_ids, suffix_slice, target_slice, loss_slice):
    device = input_ids.device
    suffix_ids = input_ids[suffix_slice]
    vocab_size = embed_weights.shape[0]
    one_hot = torch.zeros((suffix_ids.shape[0], vocab_size), device=device, dtype=embed_weights.dtype)
    one_hot.scatter_(1, suffix_ids[:, None], 1.0)
    one_hot.requires_grad_(True)

    input_embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    suffix_embeds = one_hot @ embed_weights
    input_embeds[:, suffix_slice, :] = suffix_embeds.unsqueeze(0)

    logits = model(inputs_embeds=input_embeds).logits
    loss = F.cross_entropy(logits[0, loss_slice, :], input_ids[target_slice])
    loss.backward()
    grad = one_hot.grad.detach()
    grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
    return grad, float(loss.detach().cpu())


@torch.no_grad()
def _batch_loss(model, tok, ids_batch: torch.Tensor, target_slice: slice, loss_slice: slice) -> torch.Tensor:
    attn = torch.ones_like(ids_batch)
    logits = model(input_ids=ids_batch, attention_mask=attn).logits
    losses = []
    for i in range(ids_batch.shape[0]):
        losses.append(F.cross_entropy(logits[i, loss_slice, :], ids_batch[i, target_slice], reduction="mean"))
    return torch.stack(losses)


def optimize_suffix(model_wrap: HFChatModel, goal: str, cfg: GCGConfig) -> Dict:
    tok = model_wrap.tokenizer
    model = model_wrap.model
    device = model_wrap.device
    # Neutral initial suffix. Do not initialize with known jailbreak strings.
    suffix = " ".join(["x"] * cfg.suffix_len)
    history = []
    embed_weights = model.get_input_embeddings().weight

    for step in trange(cfg.steps, desc="GCG", leave=False):
        input_ids, suffix_slice, target_slice, loss_slice = _build_ids(tok, goal, suffix, cfg.target_text, device)
        grad, loss = _token_gradients(model, embed_weights, input_ids, suffix_slice, target_slice, loss_slice)
        suffix_ids = input_ids[suffix_slice]
        top_tokens = torch.topk(-grad, k=min(cfg.topk, grad.shape[-1]), dim=-1).indices

        candidates = []
        for pos in range(suffix_ids.shape[0]):
            for tok_id in top_tokens[pos][: max(1, cfg.batch_size // max(1, suffix_ids.shape[0]))]:
                new_ids = input_ids.clone()
                new_ids[suffix_slice.start + pos] = tok_id
                candidates.append(new_ids)
        if not candidates:
            break
        batch = torch.stack(candidates[: cfg.batch_size], dim=0)
        losses = _batch_loss(model, tok, batch, target_slice, loss_slice)
        best_idx = int(torch.argmin(losses).detach().cpu())
        best_ids = batch[best_idx]
        suffix = tok.decode(best_ids[suffix_slice], skip_special_tokens=True)
        history.append({"step": step, "loss": float(losses[best_idx].detach().cpu()), "suffix_hash": sha256_text(suffix)})
    return {"suffix": suffix, "history": history}


def run_gcg_one(row: Dict, target: HFChatModel, guard, out_path: str, cfg: GCGConfig, gen: GenerationConfigLite) -> Dict:
    goal = row["prompt"]
    with Timer() as t:
        opt = optimize_suffix(target, goal, cfg)
        attacked_prompt = goal.strip() + " " + opt["suffix"]
        response = target.generate(attacked_prompt, gen=gen)
        judge = guard.classify(attacked_prompt, response)
    rec = {
        "attack": "gcg",
        "id": row.get("id"),
        "category": row.get("category"),
        "prompt_hash": sha256_text(attacked_prompt),
        "original_prompt_hash": row.get("prompt_hash") or sha256_text(goal),
        "suffix_hash": sha256_text(opt["suffix"]),
        "prompt_redacted": redact_text(attacked_prompt),
        "response_redacted": redact_text(response),
        "judge_label": judge.label,
        "unsafe": bool(judge.is_unsafe),
        "blocked": False,
        "latency_s": t.elapsed,
        "curve": opt["history"],
    }
    append_jsonl(out_path, rec)
    return rec
