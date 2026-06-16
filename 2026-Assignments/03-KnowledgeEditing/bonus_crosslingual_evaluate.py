import json
from pathlib import Path
from statistics import mean
from typing import Any


ROME_PATH = Path("results/rome_crosslingual_results.json")
MEMIT_PATH = Path("results/memit_crosslingual_results.json")
OUTPUT_PATH = Path("results/crosslingual_metrics.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, list):
        if not value:
            return default
        return scalar(value[0], default=default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> float:
    return round(value * 100, 2)


def unwrap_results(raw_data: Any) -> tuple[list[dict], dict]:
    if isinstance(raw_data, dict) and "results" in raw_data:
        return raw_data["results"], raw_data.get("meta", {})
    return raw_data, {}


def summarize_crosslingual(edit_results: list[dict]) -> tuple[dict, list[dict]]:
    per_case = []

    for item in edit_results:
        metrics = item.get("metrics", {})
        post = metrics.get("post", {})
        crosslingual_eval = item.get("crosslingual_eval", {})

        es = scalar(post.get("rewrite_acc"))
        ps_en = scalar(post.get("rephrase_acc"))
        ns = scalar(post.get("locality_acc"))
        ps_zh = 1.0 if crosslingual_eval.get("zh_rephrase_hit") else 0.0
        bilingual = 1.0 if crosslingual_eval.get("rephrase_hit") and crosslingual_eval.get("zh_rephrase_hit") else 0.0

        per_case.append(
            {
                "id": item.get("id"),
                "prompt": item.get("prompt"),
                "subject": item.get("subject"),
                "target_new": item.get("target_new"),
                "zh_target_new": item.get("zh_target_new"),
                "ES": es,
                "PS_en": ps_en,
                "PS_zh": ps_zh,
                "NS": ns,
                "bilingual_success": bilingual,
                "rewrite_answer": item.get("generations", {}).get("rewrite_answer", ""),
                "rephrase_answer": item.get("generations", {}).get("rephrase_answer", ""),
                "zh_rephrase_answer": item.get("generations", {}).get("zh_rephrase_answer", ""),
                "locality_answer": item.get("generations", {}).get("locality_answer", ""),
            }
        )

    summary = {
        "total": len(per_case),
        "ES": pct(mean(case["ES"] for case in per_case)) if per_case else 0.0,
        "PS_en": pct(mean(case["PS_en"] for case in per_case)) if per_case else 0.0,
        "PS_zh": pct(mean(case["PS_zh"] for case in per_case)) if per_case else 0.0,
        "NS": pct(mean(case["NS"] for case in per_case)) if per_case else 0.0,
        "crosslingual_gap": pct(
            max(0.0, mean(case["PS_en"] for case in per_case) - mean(case["PS_zh"] for case in per_case))
        ) if per_case else 0.0,
        "bilingual_success_rate": pct(mean(case["bilingual_success"] for case in per_case)) if per_case else 0.0,
    }
    return summary, per_case


def print_summary(label: str, summary: dict, meta: dict | None = None):
    print(f"\n===== {label} Cross-lingual Summary =====")
    print(f"Total samples: {summary['total']}")
    print(f"ES / English rewrite success: {summary['ES']}%")
    print(f"PS_en / English paraphrase success: {summary['PS_en']}%")
    print(f"PS_zh / Chinese paraphrase success: {summary['PS_zh']}%")
    print(f"NS / Locality: {summary['NS']}%")
    print(f"Cross-lingual gap: {summary['crosslingual_gap']}%")
    print(f"Bilingual success rate: {summary['bilingual_success_rate']}%")
    if meta:
        if "elapsed_seconds" in meta:
            print(f"Elapsed seconds: {meta['elapsed_seconds']}")
        if "peak_memory_mb" in meta:
            print(f"Peak GPU memory (MB): {meta['peak_memory_mb']}")


def print_table(label: str, per_case: list[dict], limit: int | None = None):
    print(f"\n===== Per-case {label} Cross-lingual Metrics =====")
    print(f"{'ID':<4} {'ES':>8} {'EN':>8} {'ZH':>8} {'NS':>8} {'BI':>8}  Prompt")
    rows = per_case if limit is None else per_case[:limit]
    for case in rows:
        print(
            f"{case['id']:<4} "
            f"{pct(case['ES']):>7.2f}% "
            f"{pct(case['PS_en']):>7.2f}% "
            f"{pct(case['PS_zh']):>7.2f}% "
            f"{pct(case['NS']):>7.2f}% "
            f"{pct(case['bilingual_success']):>7.2f}%  "
            f"{case['prompt']}"
        )
    if limit is not None and len(per_case) > limit:
        print(f"... omitted {len(per_case) - limit} additional rows")


def main():
    output = {}

    if ROME_PATH.exists():
        rome_raw = load_json(ROME_PATH)
        rome_results, rome_meta = unwrap_results(rome_raw)
        rome_summary, rome_per_case = summarize_crosslingual(rome_results)
        output["rome"] = rome_summary
        output["rome_per_case"] = rome_per_case
        if rome_meta:
            output["rome_meta"] = rome_meta
        print_summary("ROME", rome_summary, rome_meta)
        print_table("ROME", rome_per_case)
    else:
        print(f"ROME cross-lingual result not found, skip: {ROME_PATH}")

    if MEMIT_PATH.exists():
        memit_raw = load_json(MEMIT_PATH)
        memit_results, memit_meta = unwrap_results(memit_raw)
        memit_summary, memit_per_case = summarize_crosslingual(memit_results)
        output["memit"] = memit_summary
        output["memit_per_case"] = memit_per_case
        if memit_meta:
            output["memit_meta"] = memit_meta
        print_summary("MEMIT", memit_summary, memit_meta)
        print_table("MEMIT", memit_per_case, limit=20)
    else:
        print(f"MEMIT cross-lingual result not found, skip: {MEMIT_PATH}")

    if not output:
        raise FileNotFoundError("No cross-lingual result files found. Run the bonus editing scripts first.")

    save_json(output, OUTPUT_PATH)
    print(f"\nSaved cross-lingual metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
