from __future__ import annotations

import argparse

from jailbreak_lab.data import load_benign_xstest, load_harmful_jbb, write_private_and_manifest
from jailbreak_lab.utils import load_config, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    dcfg = cfg["data"]
    harmful = load_harmful_jbb(
        dcfg["harmful_dataset"],
        min_categories=int(dcfg.get("min_categories", 4)),
        samples_per_category=int(dcfg.get("samples_per_category", 8)),
    )
    benign = load_benign_xstest(dcfg["benign_dataset"], n=int(dcfg.get("benign_samples", 32)))
    write_private_and_manifest(
        harmful,
        benign,
        dcfg["harmful_out"],
        dcfg["benign_out"],
        "data/public_samples/manifest.jsonl",
    )
    print(f"Wrote harmful={len(harmful)} benign={len(benign)}")
    print("Private raw data: data/private/ ; public redacted manifest: data/public_samples/manifest.jsonl")


if __name__ == "__main__":
    main()
