from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from jailbreak_lab.attacks.gcg import GCGConfig, run_gcg_one
from jailbreak_lab.guards import make_guard
from jailbreak_lab.models import GenerationConfigLite, HFChatModel
from jailbreak_lab.utils import load_config, private_path_warning, read_jsonl, require_ack, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ack_responsible_use", action="store_true")
    ap.add_argument("--fallback_guard", action="store_true", help="Debug only. Do not use for final ASR.")
    args = ap.parse_args()
    require_ack(args.ack_responsible_use)
    private_path_warning(args.out)
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    rows = read_jsonl(cfg["data"]["harmful_out"])
    if args.limit:
        rows = rows[: args.limit]
    mcfg = cfg["models"]
    target = HFChatModel(mcfg["target"], dtype=mcfg.get("dtype", "auto"), load_in_4bit=False)
    guard = make_guard(cfg, fallback=args.fallback_guard)
    gen = GenerationConfigLite(**cfg.get("generation", {}))
    gcfg = GCGConfig(**cfg.get("attacks", {}).get("gcg", {}))
    Path(args.out).unlink(missing_ok=True)
    for row in tqdm(rows, desc="gcg"):
        run_gcg_one(row, target, guard, args.out, gcfg, gen)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
