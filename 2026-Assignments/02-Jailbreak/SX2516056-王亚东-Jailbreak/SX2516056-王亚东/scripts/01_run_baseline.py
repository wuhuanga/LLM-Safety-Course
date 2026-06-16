from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from jailbreak_lab.guards import make_guard
from jailbreak_lab.models import GenerationConfigLite, HFChatModel
from jailbreak_lab.utils import Timer, append_jsonl, load_config, private_path_warning, read_jsonl, redact_text, seed_everything, sha256_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["harmful", "benign"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fallback_guard", action="store_true", help="Debug only. Do not use for final ASR.")
    args = ap.parse_args()
    private_path_warning(args.out)
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    rows = read_jsonl(cfg["data"]["harmful_out" if args.split == "harmful" else "benign_out"])
    if args.limit:
        rows = rows[: args.limit]
    mcfg = cfg["models"]
    target = HFChatModel(mcfg["target"], dtype=mcfg.get("dtype", "auto"), load_in_4bit=mcfg.get("load_in_4bit", False))
    guard = make_guard(cfg, fallback=args.fallback_guard)
    gen = GenerationConfigLite(**cfg.get("generation", {}))
    Path(args.out).unlink(missing_ok=True)
    for row in tqdm(rows, desc=f"baseline-{args.split}"):
        with Timer() as t:
            response = target.generate(row["prompt"], gen=gen)
            judge = guard.classify(row["prompt"], response)
        append_jsonl(args.out, {
            "attack": "baseline",
            "split": args.split,
            "id": row.get("id"),
            "category": row.get("category"),
            "prompt_hash": row.get("prompt_hash") or sha256_text(row["prompt"]),
            "response_redacted": redact_text(response),
            "unsafe": bool(judge.is_unsafe),
            "judge_label": judge.label,
            "blocked": False,
            "latency_s": t.elapsed,
        })
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
