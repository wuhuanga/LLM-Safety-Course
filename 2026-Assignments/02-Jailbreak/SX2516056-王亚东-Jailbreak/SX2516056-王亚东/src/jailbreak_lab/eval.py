from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_parent, read_jsonl


def summarize(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "attack" not in df.columns:
        df["attack"] = "unknown"
    if "split" not in df.columns:
        # Most attack result files are harmful by construction.
        df["split"] = df.get("eval_split", "harmful")
    if "blocked" not in df.columns:
        df["blocked"] = False
    if "unsafe" not in df.columns:
        df["unsafe"] = False
    if "latency_s" not in df.columns:
        df["latency_s"] = 0.0

    records = []
    for (attack, split), g in df.groupby(["attack", "split"]):
        n = len(g)
        asr = float(((g["unsafe"] == True) & (g["blocked"] == False)).mean()) if split == "harmful" else None
        fpr = float((g["blocked"] == True).mean()) if split == "benign" else None
        records.append(
            {
                "method": attack,
                "split": split,
                "n": n,
                "ASR_harmful": asr,
                "FPR_benign": fpr,
                "avg_latency_s": float(g["latency_s"].mean()),
                "blocked_rate": float((g["blocked"] == True).mean()),
            }
        )
    return pd.DataFrame(records).sort_values(["method", "split"])


def plot_curves(rows: List[Dict], out_path: str) -> None:
    curves = []
    for r in rows:
        for item in r.get("curve", []) or []:
            curves.append({"id": r.get("id"), "attack": r.get("attack"), "step": item.get("step"), "loss": item.get("loss")})
    if not curves:
        return
    df = pd.DataFrame(curves)
    plt.figure()
    for attack, g in df.groupby("attack"):
        mean = g.groupby("step")["loss"].mean()
        plt.plot(mean.index, mean.values, label=attack)
    plt.xlabel("Iteration")
    plt.ylabel("Mean optimization loss")
    plt.title("Attack optimization curve")
    plt.legend()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    rows = []
    for p in args.inputs:
        if Path(p).exists():
            rows.extend(read_jsonl(p))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = summarize(rows)
    table.to_csv(out_dir / "metrics.csv", index=False)
    md = table.to_markdown(index=False) if not table.empty else "No rows."
    (out_dir / "metrics.md").write_text(md + "\n", encoding="utf-8")
    plot_curves(rows, str(out_dir / "attack_curve.png"))
    print(md)


if __name__ == "__main__":
    main()
