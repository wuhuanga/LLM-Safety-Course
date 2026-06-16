from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from .defenses import make_defense
from .guards import make_guard
from .models import GenerationConfigLite, HFChatModel
from .utils import load_config, redact_text, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--defense", default="llama_guard", choices=["llama_guard", "perplexity", "smoothllm"])
    ap.add_argument("--fallback_guard", action="store_true", help="Debug only. Do not use for final ASR.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    mcfg = cfg["models"]
    gen = GenerationConfigLite(**cfg.get("generation", {}))
    target = HFChatModel(mcfg["target"], dtype=mcfg.get("dtype", "auto"), load_in_4bit=mcfg.get("load_in_4bit", False))
    guard = make_guard(cfg, fallback=args.fallback_guard)
    defense = make_defense(args.defense, cfg, target, guard)
    console = Console()
    console.print("输入 prompt，输入 :q 退出。")
    while True:
        prompt = input("prompt> ").strip()
        if prompt in {":q", "quit", "exit"}:
            break
        decision = defense.run(target, prompt, gen)
        visible_response = "[已脱敏：防御判定为 unsafe]" if decision.unsafe else decision.response
        console.print(Panel.fit(
            f"blocked={decision.blocked}\nreason={decision.reason}\nlatency={decision.latency_s:.3f}s\n\nresponse:\n{visible_response}",
            title="Defense Decision"
        ))


if __name__ == "__main__":
    main()
