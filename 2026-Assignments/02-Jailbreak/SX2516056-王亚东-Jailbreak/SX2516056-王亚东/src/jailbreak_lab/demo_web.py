from __future__ import annotations

import argparse

import gradio as gr

from .defenses import make_defense
from .guards import make_guard
from .models import GenerationConfigLite, HFChatModel
from .utils import load_config, seed_everything


def build_app(config_path: str, defense_name: str, fallback_guard: bool = False):
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    mcfg = cfg["models"]
    gen = GenerationConfigLite(**cfg.get("generation", {}))
    target = HFChatModel(mcfg["target"], dtype=mcfg.get("dtype", "auto"), load_in_4bit=mcfg.get("load_in_4bit", False))
    guard = make_guard(cfg, fallback=fallback_guard)
    defense = make_defense(defense_name, cfg, target, guard)

    def infer(prompt: str):
        decision = defense.run(target, prompt, gen)
        response = "[已脱敏：防御判定为 unsafe]" if decision.unsafe else decision.response
        return response, str(decision.blocked), decision.reason, f"{decision.latency_s:.3f}s"

    with gr.Blocks(title="LLM Jailbreak Defense Demo") as demo:
        gr.Markdown("# LLM Jailbreak Attack & Defense Demo\n输入 prompt，展示模型回复、防御判定、拦截依据和延迟。unsafe 回复默认脱敏。")
        inp = gr.Textbox(label="Prompt", lines=5)
        btn = gr.Button("Run")
        out = gr.Textbox(label="Model Response / Redacted", lines=8)
        blocked = gr.Textbox(label="Blocked")
        reason = gr.Textbox(label="Defense Reason")
        latency = gr.Textbox(label="Latency")
        btn.click(infer, inputs=inp, outputs=[out, blocked, reason, latency])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--defense", default="llama_guard", choices=["llama_guard", "perplexity", "smoothllm"])
    ap.add_argument("--fallback_guard", action="store_true", help="Debug only. Do not use for final ASR.")
    args = ap.parse_args()
    build_app(args.config, args.defense, args.fallback_guard).launch()


if __name__ == "__main__":
    main()
