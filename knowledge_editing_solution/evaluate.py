"""Compute ES, PS, and NS metrics for knowledge editing outputs.

Two metric views are saved:
- ``easyedit_metrics``: EasyEdit's internal rewrite/rephrase/locality accuracy.
- ``generation_metrics``: strict substring hits on post-edit free generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from editing_utils import load_json, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate knowledge editing results.")
    parser.add_argument("--baseline", default="outputs/baseline.json", help="Path to baseline outputs.")
    parser.add_argument("--rome", default="outputs/rome_results.json", help="Path to ROME outputs.")
    parser.add_argument("--memit", default="outputs/memit_results.json", help="Path to MEMIT outputs.")
    parser.add_argument("--output", default="outputs/metrics.json", help="Path to save computed metrics.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing result files instead of failing. Useful while running experiments incrementally.",
    )
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def mean(values: list[bool] | list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def get_results(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected key 'results' to be a list in {path}.")
    return results


def generation_method_metrics(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    results = get_results(payload, path)
    efficacy = [as_bool(item.get("contains_target_new")) for item in results]
    paraphrase = [as_bool(item.get("rephrase_contains_target_new")) for item in results]
    neighborhood = [as_bool(item.get("locality_contains_ground_truth")) for item in results]

    return {
        "num_records": len(results),
        "ES": mean(efficacy),
        "PS": mean(paraphrase),
        "NS": mean(neighborhood),
        "counts": {
            "efficacy_hits": sum(efficacy),
            "paraphrase_hits": sum(paraphrase),
            "neighborhood_hits": sum(neighborhood),
        },
    }


def values_from_metric(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        values: list[float] = []
        for item in value:
            values.extend(values_from_metric(item))
        return values
    return []


def locality_values(post: dict[str, Any]) -> list[float]:
    locality = post.get("locality")
    if not isinstance(locality, dict):
        return []
    values: list[float] = []
    for key, value in locality.items():
        if key.endswith("_acc"):
            values.extend(values_from_metric(value))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key.endswith("_acc"):
                    values.extend(values_from_metric(sub_value))
    return values


def extract_easyedit_records(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if isinstance(payload.get("easyedit_metrics"), list):
        return [item for item in payload["easyedit_metrics"] if isinstance(item, dict)]

    records = []
    for result in get_results(payload, path):
        metrics = result.get("easyedit_metrics")
        if isinstance(metrics, dict):
            records.append(metrics)
    return records


def easyedit_method_metrics(payload: dict[str, Any], path: Path) -> dict[str, Any] | None:
    records = extract_easyedit_records(payload, path)
    if not records:
        return None

    efficacy: list[float] = []
    paraphrase: list[float] = []
    neighborhood: list[float] = []

    for record in records:
        post = record.get("post")
        if not isinstance(post, dict):
            continue
        efficacy.extend(values_from_metric(post.get("rewrite_acc")))
        paraphrase.extend(values_from_metric(post.get("rephrase_acc")))
        neighborhood.extend(locality_values(post))

    return {
        "num_records": len(records),
        "ES": mean(efficacy),
        "PS": mean(paraphrase),
        "NS": mean(neighborhood),
        "counts": {
            "efficacy_values": len(efficacy),
            "paraphrase_values": len(paraphrase),
            "neighborhood_values": len(neighborhood),
        },
    }


def load_if_exists(name: str, path_text: str, allow_missing: bool) -> tuple[str, Path, dict[str, Any]] | None:
    path = resolve_path(path_text)
    if not path.exists():
        if allow_missing:
            print(f"Skipping missing {name} result file: {path}")
            return None
        raise FileNotFoundError(f"Missing {name} result file: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return name, path, payload


def main() -> None:
    args = parse_args()
    requested = [
        ("baseline", args.baseline),
        ("rome", args.rome),
        ("memit", args.memit),
    ]
    loaded = [
        item
        for name, path in requested
        if (item := load_if_exists(name, path, args.allow_missing)) is not None
    ]

    generation_metrics: dict[str, Any] = {}
    easyedit_metrics: dict[str, Any] = {}
    for name, path, payload in loaded:
        generation_metrics[name] = generation_method_metrics(payload, path)
        easyedit = easyedit_method_metrics(payload, path)
        if easyedit is not None:
            easyedit_metrics[name] = easyedit

    payload = {
        "metrics": easyedit_metrics or generation_metrics,
        "easyedit_metrics": easyedit_metrics,
        "generation_metrics": generation_metrics,
        "metric_definitions": {
            "ES": "Efficacy Score. In easyedit_metrics this is rewrite_acc; in generation_metrics it is strict target_new substring hit.",
            "PS": "Paraphrase Score. In easyedit_metrics this is rephrase_acc; in generation_metrics it is strict rephrase substring hit.",
            "NS": "Neighborhood Score. In easyedit_metrics this is locality_acc; in generation_metrics it is strict locality answer hit.",
        },
    }
    save_json(args.output, payload)

    print("Saved metrics to", resolve_path(args.output))
    print("EasyEdit internal metrics:")
    for name, values in easyedit_metrics.items():
        print(
            f"{name}: ES={values['ES']:.4f}, PS={values['PS']:.4f}, "
            f"NS={values['NS']:.4f} ({values['num_records']} records)"
        )
    print("Free-generation strict metrics:")
    for name, values in generation_metrics.items():
        print(
            f"{name}: ES={values['ES']:.4f}, PS={values['PS']:.4f}, "
            f"NS={values['NS']:.4f} ({values['num_records']} records)"
        )


if __name__ == "__main__":
    main()
