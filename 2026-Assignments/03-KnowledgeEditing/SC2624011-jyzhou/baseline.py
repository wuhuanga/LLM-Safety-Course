import argparse
from typing import Any, Dict

from tqdm import tqdm

from utils import contains_answer, generate_text, load_causal_lm, load_json, save_json, set_seed


def evaluate_record(model, tokenizer, record: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    prompt_output = generate_text(model, tokenizer, record["prompt"], max_new_tokens=max_new_tokens)
    rephrase_output = generate_text(
        model,
        tokenizer,
        record["rephrase_prompt"],
        max_new_tokens=max_new_tokens,
    )
    locality_output = generate_text(
        model,
        tokenizer,
        record["locality_prompt"],
        max_new_tokens=max_new_tokens,
    )

    return {
        **record,
        "baseline_output": prompt_output,
        "baseline_rephrase_output": rephrase_output,
        "baseline_locality_output": locality_output,
        "baseline_contains_target": contains_answer(prompt_output, record["target_new"]),
        "baseline_contains_ground_truth": contains_answer(prompt_output, record.get("ground_truth")),
        "locality_contains_ground_truth": contains_answer(
            locality_output,
            record.get("locality_ground_truth"),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline generation before knowledge editing.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data", default="data/custom_10.json")
    parser.add_argument("--out", default="results/baseline_custom.json")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    records = load_json(args.data)
    model, tokenizer = load_causal_lm(args.model, dtype=args.dtype, device_map=args.device_map)

    results = []
    for record in tqdm(records, desc="baseline"):
        results.append(evaluate_record(model, tokenizer, record, args.max_new_tokens))

    save_json(
        {
            "model": args.model,
            "num_records": len(results),
            "results": results,
        },
        args.out,
    )

    print("\nBaseline summary")
    print("case_id\tcontains_target\tcontains_ground_truth\tprompt_output")
    for item in results:
        print(
            f"{item['case_id']}\t{item['baseline_contains_target']}\t"
            f"{item['baseline_contains_ground_truth']}\t{item['baseline_output'][:120]}"
        )


if __name__ == "__main__":
    main()
