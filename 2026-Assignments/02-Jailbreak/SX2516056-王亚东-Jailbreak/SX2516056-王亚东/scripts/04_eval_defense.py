from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from jailbreak_lab.defenses import make_defense
from jailbreak_lab.guards import make_guard
from jailbreak_lab.models import GenerationConfigLite, HFChatModel
from jailbreak_lab.utils import append_jsonl, load_config, private_path_warning, read_jsonl, redact_text, seed_everything, sha256_text


def run_split(rows, split, defense, target, gen, out):
    for row in tqdm(rows, desc=f"defense-{split}"):
        decision = defense.run(target, row["prompt"], gen)
        append_jsonl(out, {
            "attack": "defense",
            "defense": defense.__class__.__name__,
            "split": split,
            "id": row.get("id"),
            "category": row.get("category"),
            "prompt_hash": row.get("prompt_hash") or sha256_text(row["prompt"]),
            "response_redacted": redact_text(decision.response),
            "unsafe": bool(decision.unsafe),
            "blocked": bool(decision.blocked),
            "reason": decision.reason,
            "latency_s": decision.latency_s,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--harmful", default="data/private/harmful_eval.jsonl")
    ap.add_argument("--benign", default="data/private/benign_eval.jsonl")
    ap.add_argument("--defense", default="llama_guard", choices=["llama_guard", "perplexity", "smoothllm"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fallback_guard", action="store_true", help="Debug only. Do not use for final ASR.")
    args = ap.parse_args()
    private_path_warning(args.out)
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    harmful = read_jsonl(args.harmful)
    benign = read_jsonl(args.benign)
    if args.limit:
        harmful = harmful[: args.limit]
        benign = benign[: args.limit]
    mcfg = cfg["models"]
    target = HFChatModel(mcfg["target"], dtype=mcfg.get("dtype", "auto"), load_in_4bit=mcfg.get("load_in_4bit", False))
    guard = make_guard(cfg, fallback=args.fallback_guard)
    defense = make_defense(args.defense, cfg, target, guard)
    gen = GenerationConfigLite(**cfg.get("generation", {}))
    Path(args.out).unlink(missing_ok=True)
    run_split(harmful, "harmful", defense, target, gen, args.out)
    run_split(benign, "benign", defense, target, gen, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
