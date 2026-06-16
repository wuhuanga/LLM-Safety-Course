import json
from pathlib import Path
from statistics import mean
from typing import Any


BASELINE_PATH = Path("results/baseline_results.json")
ROME_PATH = Path("results/rome_results.json")
MEMIT_PATH = Path("results/memit_results.json")
OUTPUT_PATH = Path("results/metrics.json")


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


def summarize_baseline(baseline_results: list[dict]) -> dict:
    total = len(baseline_results)
    target_hits = sum(1 for item in baseline_results if item.get("matches_target_new"))
    ground_truth_hits = sum(1 for item in baseline_results if item.get("matches_ground_truth"))
    return {
        "total": total,
        "already_matches_target_new_count": target_hits,
        "already_matches_target_new_rate": pct(target_hits / total) if total else 0.0,
        "matches_ground_truth_count": ground_truth_hits,
        "matches_ground_truth_rate": pct(ground_truth_hits / total) if total else 0.0,
    }


def unwrap_edit_results(raw_data: Any) -> tuple[list[dict], dict]:
    if isinstance(raw_data, dict) and "results" in raw_data:
        meta = raw_data.get("meta", {})
        return raw_data["results"], meta
    return raw_data, {}


def summarize_edit_results(edit_results: list[dict]) -> tuple[dict, list[dict]]:
    per_case = []

    for item in edit_results:
        metrics = item.get("metrics", {})
        pre = metrics.get("pre", {})
        post = metrics.get("post", {})

        pre_rewrite = scalar(pre.get("rewrite_acc"))
        pre_rephrase = scalar(pre.get("rephrase_acc"))
        post_rewrite = scalar(post.get("rewrite_acc"))
        post_rephrase = scalar(post.get("rephrase_acc"))
        post_locality = scalar(post.get("locality_acc"))

        per_case.append(
            {
                "id": item.get("id"),
                "prompt": item.get("prompt"),
                "target_new": item.get("target_new"),
                "ground_truth": item.get("ground_truth"),
                "subject": item.get("subject"),
                "pre_rewrite_acc": pre_rewrite,
                "pre_rephrase_acc": pre_rephrase,
                "post_rewrite_acc": post_rewrite,
                "post_rephrase_acc": post_rephrase,
                "post_locality_acc": post_locality,
                "ES": post_rewrite,
                "PS": post_rephrase,
                "NS": post_locality,
            }
        )

    es_values = [case["ES"] for case in per_case]
    ps_values = [case["PS"] for case in per_case]
    ns_values = [case["NS"] for case in per_case]

    summary = {
        "total": len(per_case),
        "ES": pct(mean(es_values)) if es_values else 0.0,
        "PS": pct(mean(ps_values)) if ps_values else 0.0,
        "NS": pct(mean(ns_values)) if ns_values else 0.0,
        "pre_rewrite_acc": pct(mean(case["pre_rewrite_acc"] for case in per_case)) if per_case else 0.0,
        "pre_rephrase_acc": pct(mean(case["pre_rephrase_acc"] for case in per_case)) if per_case else 0.0,
    }
    return summary, per_case


def print_edit_summary(label: str, summary: dict, meta: dict | None = None):
    print(f"\n===== {label} Summary =====")
    print(f"Total samples: {summary['total']}")
    print(f"ES / Efficacy: {summary['ES']}%")
    print(f"PS / Generalization: {summary['PS']}%")
    print(f"NS / Locality: {summary['NS']}%")
    print(f"Pre rewrite acc: {summary['pre_rewrite_acc']}%")
    print(f"Pre rephrase acc: {summary['pre_rephrase_acc']}%")
    if meta:
        if "dataset_source" in meta:
            print(f"Dataset source: {meta['dataset_source']}")
        if "elapsed_seconds" in meta:
            print(f"Elapsed seconds: {meta['elapsed_seconds']}")
        if "peak_memory_mb" in meta:
            print(f"Peak GPU memory (MB): {meta['peak_memory_mb']}")


def print_table(label: str, per_case: list[dict], limit: int | None = None):
    print(f"\n===== Per-case {label} Metrics =====")
    print(f"{'ID':<4} {'ES':>8} {'PS':>8} {'NS':>8}  Prompt")
    rows = per_case if limit is None else per_case[:limit]
    for case in rows:
        print(
            f"{case['id']:<4} "
            f"{pct(case['ES']):>7.2f}% "
            f"{pct(case['PS']):>7.2f}% "
            f"{pct(case['NS']):>7.2f}%  "
            f"{case['prompt']}"
        )
    if limit is not None and len(per_case) > limit:
        print(f"... omitted {len(per_case) - limit} additional rows")


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline result not found: {BASELINE_PATH}")
    if not ROME_PATH.exists():
        raise FileNotFoundError(f"ROME result not found: {ROME_PATH}")

    baseline_results = load_json(BASELINE_PATH)
    rome_raw = load_json(ROME_PATH)
    rome_results, rome_meta = unwrap_edit_results(rome_raw)

    baseline_summary = summarize_baseline(baseline_results)
    rome_summary, rome_per_case = summarize_edit_results(rome_results)

    output = {
        "baseline": baseline_summary,
        "rome": rome_summary,
        "rome_per_case": rome_per_case,
    }
    if rome_meta:
        output["rome_meta"] = rome_meta

    memit_summary = None
    memit_per_case = None
    memit_meta = None
    if MEMIT_PATH.exists():
        memit_raw = load_json(MEMIT_PATH)
        memit_results, memit_meta = unwrap_edit_results(memit_raw)
        memit_summary, memit_per_case = summarize_edit_results(memit_results)
        output["memit"] = memit_summary
        output["memit_per_case"] = memit_per_case
        if memit_meta:
            output["memit_meta"] = memit_meta

    save_json(output, OUTPUT_PATH)

    print("===== Baseline Summary =====")
    print(f"Total samples: {baseline_summary['total']}")
    print(
        "Already matches target_new: "
        f"{baseline_summary['already_matches_target_new_count']}/{baseline_summary['total']} "
        f"({baseline_summary['already_matches_target_new_rate']}%)"
    )
    print(
        "Matches ground_truth: "
        f"{baseline_summary['matches_ground_truth_count']}/{baseline_summary['total']} "
        f"({baseline_summary['matches_ground_truth_rate']}%)"
    )

    print_edit_summary("ROME", rome_summary, rome_meta)
    print_table("ROME", rome_per_case)

    if memit_summary is not None and memit_per_case is not None:
        print_edit_summary("MEMIT", memit_summary, memit_meta)
        print_table("MEMIT", memit_per_case, limit=20)
    else:
        print(f"\nMEMIT result not found, skip MEMIT summary: {MEMIT_PATH}")

    print(f"\nSaved metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
