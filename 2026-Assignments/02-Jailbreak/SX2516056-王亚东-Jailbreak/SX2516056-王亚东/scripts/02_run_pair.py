from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from jailbreak_lab.attacks.pair_lite import PairConfig, run_pair_one
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
    target = HFChatModel(mcfg["target"], dtype=mcfg.get("dtype", "auto"), load_in_4bit=mcfg.get("load_in_4bit", False))
    attacker = HFChatModel(mcfg.get("attacker", mcfg["target"]), dtype=mcfg.get("dtype", "auto"), load_in_4bit=mcfg.get("load_in_4bit", False))
    guard = make_guard(cfg, fallback=args.fallback_guard)
    gen = GenerationConfigLite(**cfg.get("generation", {}))
    pcfg = PairConfig(**cfg.get("attacks", {}).get("pair", {}))
    Path(args.out).unlink(missing_ok=True)
    for row in tqdm(rows, desc="pair"):
        run_pair_one(row, target, attacker, guard, args.out, pcfg, gen)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
