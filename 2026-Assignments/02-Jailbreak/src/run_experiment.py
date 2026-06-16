import argparse
import json
from pathlib import Path

from attack_gcg import GCGSuffixSearch
from attack_pair import PairAttack
from common import read_jsonl, timer, write_csv, write_jsonl
from defense import InputDefense
from judge import build_judge
from model_backends import build_backend


def run_baseline(samples, model, judge):
    rows = []
    for sample in samples:
        with timer() as t:
            response = model.generate(sample["prompt"])
        rows.append({
            "id": sample["id"],
            "category": sample["category"],
            "attack": "none",
            "prompt_redacted": sample["prompt"],
            "response_redacted": response,
            "judge_label": judge.classify(sample["prompt"], response),
            "latency_sec": round(t["latency_sec"], 4),
        })
    return rows


def attack_summary(rows):
    unsafe = sum(1 for row in rows if row["judge_label"] == "unsafe")
    avg = sum(row["latency_sec"] for row in rows) / len(rows) if rows else 0.0
    return unsafe / len(rows) if rows else 0.0, avg


def defense_summary(rows):
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    unsafe = sum(1 for row in harmful if row["judge_label"] == "unsafe" and not row["blocked"])
    false_positive = sum(1 for row in benign if row["blocked"])
    avg = sum(row["latency_sec"] for row in rows) / len(rows) if rows else 0.0
    return (
        unsafe / len(harmful) if harmful else 0.0,
        false_positive / len(benign) if benign else 0.0,
        avg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="offline", choices=["offline", "http", "ollama", "transformers"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--judge", default="rule", choices=["rule", "llama_guard"])
    parser.add_argument("--harmful", default="data/harmful_redacted.jsonl")
    parser.add_argument("--benign", default="data/benign_xstest_style.jsonl")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out = Path(args.out)
    model = build_backend(args.backend, model=args.model, base_url=args.base_url)
    judge = build_judge(args.judge)
    harmful = read_jsonl(args.harmful)
    benign = read_jsonl(args.benign)

    baseline = run_baseline(harmful, model, judge)
    pair = PairAttack(rounds=3).run(harmful, model, judge)
    gcg = GCGSuffixSearch(steps=4).run(harmful, model, judge)
    defense = InputDefense().run(harmful, model, judge, "harmful") + InputDefense().run(benign, model, judge, "benign")

    write_jsonl(out / "baseline_results.jsonl", baseline)
    write_jsonl(out / "pair_results.jsonl", pair)
    write_jsonl(out / "gcg_results.jsonl", gcg)
    write_jsonl(out / "defense_results.jsonl", defense)
    metric_rows = []
    for setting, rows, notes in [
        ("Baseline", baseline, "direct harmful prompts"),
        ("PAIR", pair, "black-box iterative prompt refinement"),
        ("GCG", gcg, "white-box suffix-search evaluation interface"),
    ]:
        asr, avg = attack_summary(rows)
        metric_rows.append({"setting": setting, "samples": len(rows), "asr": round(asr, 4), "fpr": "", "avg_latency_sec": round(avg, 4), "notes": notes})
    asr, fpr, avg = defense_summary(defense)
    metric_rows.append({"setting": "Defense", "samples": len(defense), "asr": round(asr, 4), "fpr": round(fpr, 4), "avg_latency_sec": round(avg, 4), "notes": "input defense on harmful and benign sets"})
    write_csv(out / "metrics.csv", metric_rows)
    (out / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    for row in metric_rows:
        print(row)


if __name__ == "__main__":
    main()
