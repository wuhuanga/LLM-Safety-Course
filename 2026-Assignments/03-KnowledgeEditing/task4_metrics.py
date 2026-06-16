import argparse
import json
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_task2(data: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = data["results"]
    total = len(results)
    es = sum(1 for item in results if item.get("rewrite_success") is True)
    ps_total = sum(1 for item in results if item.get("rephrase_success") is not None)
    ps = sum(1 for item in results if item.get("rephrase_success") is True)
    ns_total = sum(1 for item in results if item.get("locality_preserved") is not None)
    ns = sum(1 for item in results if item.get("locality_preserved") is True)

    return {
        "total_samples": total,
        "ES_count": es,
        "ES_rate": es / total if total else 0.0,
        "PS_count": ps,
        "PS_rate": ps / ps_total if ps_total else None,
        "NS_count": ns,
        "NS_rate": ns / ns_total if ns_total else None,
    }


def summarize_task3(data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = data["summary"]["task4_metrics"]
    return {
        "total_samples": metrics["sample_size"],
        "ES_count": metrics["efficacy_success_count"],
        "ES_rate": metrics["efficacy_success_rate"],
        "PS_count": metrics["generalization_success_count"],
        "PS_rate": metrics["generalization_success_rate"],
        "NS_count": metrics["locality_preserved_count"],
        "NS_rate": metrics["locality_preserved_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task2", "task3"], required=True)
    parser.add_argument("--input_path", required=True)
    args = parser.parse_args()

    data = load_json(args.input_path)
    if args.task == "task2":
        summary = summarize_task2(data)
    else:
        summary = summarize_task3(data)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
