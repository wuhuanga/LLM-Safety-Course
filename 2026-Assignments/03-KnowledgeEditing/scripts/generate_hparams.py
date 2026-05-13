import argparse
import json
import os
from pathlib import Path

import yaml
from transformers import AutoConfig


def _get_num_layers(cfg):
    for key in [
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
        "num_hidden",
    ]:
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise ValueError("Could not determine number of layers from model config")


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_hparams(model_name, output_dir, device, num_layers=None, config_path=None):
    if num_layers is None:
        if config_path:
            cfg = AutoConfig.from_pretrained(config_path)
        else:
            cfg = AutoConfig.from_pretrained(model_name)
        num_layers = _get_num_layers(cfg)
    center = max(0, num_layers // 2)

    rome_layers = [center]
    memit_layers = list(range(max(0, center - 2), min(num_layers, center + 3)))

    stats_dir = str(Path(output_dir).parent / "data" / "stats")
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    rome = {
        "alg_name": "ROME",
        "model_name": model_name,
        "stats_dir": stats_dir,
        "device": device,
        "layers": rome_layers,
        "fact_token": "subject_last",
        "v_num_grad_steps": 20,
        "v_lr": 5e-1,
        "v_loss_layer": num_layers - 1,
        "v_weight_decay": 1e-3,
        "clamp_norm_factor": 4,
        "kl_factor": 0.0625,
        "mom2_adjustment": False,
        "context_template_length_params": [[5, 10], [10, 10]],
        "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
        "layer_module_tmp": "model.layers.{}",
        "mlp_module_tmp": "model.layers.{}.mlp",
        "attn_module_tmp": "model.layers.{}.self_attn",
        "ln_f_module": "model.norm",
        "lm_head_module": "lm_head",
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 20000,
        "mom2_dtype": "float32",
        "model_parallel": False,
        "fp16": False,
    }

    memit = {
        "alg_name": "MEMIT",
        "model_name": model_name,
        "stats_dir": stats_dir,
        "device": device,
        "layers": memit_layers,
        "clamp_norm_factor": 4,
        "layer_selection": "all",
        "fact_token": "subject_last",
        "v_num_grad_steps": 20,
        "v_lr": 5e-1,
        "v_loss_layer": num_layers - 1,
        "v_weight_decay": 1e-3,
        "kl_factor": 0.0625,
        "mom2_adjustment": True,
        "mom2_update_weight": 15000,
        "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
        "layer_module_tmp": "model.layers.{}",
        "mlp_module_tmp": "model.layers.{}.mlp",
        "attn_module_tmp": "model.layers.{}.self_attn",
        "ln_f_module": "model.norm",
        "lm_head_module": "lm_head",
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 20000,
        "mom2_dtype": "float32",
        "model_parallel": False,
    }

    out_dir = Path(output_dir)
    _write_yaml(out_dir / "rome_qwen2.5-0.5b.yaml", rome)
    _write_yaml(out_dir / "memit_qwen2.5-0.5b.yaml", memit)

    meta = {
        "model": model_name,
        "num_layers": num_layers,
        "rome_layers": rome_layers,
        "memit_layers": memit_layers,
        "stats_dir": stats_dir,
    }
    with open(out_dir / "hparams_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="../hparams")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    build_hparams(args.model, args.output_dir, args.device, args.num_layers, args.config)
    print("Hparams written to", args.output_dir)


if __name__ == "__main__":
    main()
