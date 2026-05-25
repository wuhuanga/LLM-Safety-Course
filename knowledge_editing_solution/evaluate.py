"""Compute ES, PS, and NS metrics for knowledge editing outputs."""

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


def mean(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def get_results(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected key 'results' to be a list in {path}.")
    return results


def method_metrics(payload: dict[str, Any], path: Path) -> dict[str, Any]:
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

    metrics: dict[str, Any] = {}
    for name, path, payload in loaded:
        metrics[name] = method_metrics(payload, path)

    payload = {
        "metrics": metrics,
        "metric_definitions": {
            "ES": "Efficacy Score: fraction of direct edited prompts whose generation contains target_new.",
            "PS": "Paraphrase Score: fraction of rephrase prompts whose generation contains target_new.",
            "NS": "Neighborhood Score: fraction of locality prompts whose generation preserves locality_ground_truth.",
        },
    }
    save_json(args.output, payload)

    print("Saved metrics to", resolve_path(args.output))
    for name, values in metrics.items():
        print(
            f"{name}: ES={values['ES']:.4f}, PS={values['PS']:.4f}, "
            f"NS={values['NS']:.4f} ({values['num_records']} records)"
        )


if __name__ == "__main__":
    main()
