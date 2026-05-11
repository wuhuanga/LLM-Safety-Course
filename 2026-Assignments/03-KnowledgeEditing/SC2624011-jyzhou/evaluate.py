import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from utils import load_json, mean_or_none, numeric_values


def post_metric_objects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("algorithm") == "ROME":
        objects = []
        for case in payload.get("case_results", []):
            metrics = case.get("metrics", [])
            if isinstance(metrics, list):
                objects.extend(metrics)
            elif isinstance(metrics, dict):
                objects.append(metrics)
        return objects

    metrics = payload.get("metrics", [])
    if isinstance(metrics, list):
        return metrics
    if isinstance(metrics, dict):
        return [metrics]
    return []


def summarize_payload(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    metric_objects = post_metric_objects(payload)
    post_objects = [m.get("post", {}) for m in metric_objects if isinstance(m, dict)]

    es = mean_or_none(value for post in post_objects for value in numeric_values(post.get("rewrite_acc")))
    ps = mean_or_none(value for post in post_objects for value in numeric_values(post.get("rephrase_acc")))

    locality_values = []
    for post in post_objects:
        locality_values.extend(numeric_values(post.get("locality", {})))
    ns = mean_or_none(locality_values)

    return {
        "method": payload.get("algorithm", Path(path).stem),
        "num_records": payload.get("num_records"),
        "ES": es,
        "PS": ps,
        "NS": ns,
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "peak_memory_gb": payload.get("peak_memory_gb"),
        "source": path,
    }


def percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ES, PS and NS metrics.")
    parser.add_argument("--inputs", nargs="+", default=["results/rome_metrics.json", "results/memit_metrics.json"])
    parser.add_argument("--out", default="results/final_metrics.csv")
    args = parser.parse_args()

    rows = [summarize_payload(path) for path in args.inputs if Path(path).exists()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with Path(args.out).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "num_records",
                "ES",
                "PS",
                "NS",
                "elapsed_seconds",
                "peak_memory_gb",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("| Method | Records | ES | PS | NS | Time(s) | Peak Mem(GB) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        elapsed = row["elapsed_seconds"]
        memory = row["peak_memory_gb"]
        print(
            f"| {row['method']} | {row['num_records']} | {percent(row['ES'])} | "
            f"{percent(row['PS'])} | {percent(row['NS'])} | "
            f"{elapsed if elapsed is not None else ''} | {memory if memory is not None else ''} |"
        )


if __name__ == "__main__":
    main()
