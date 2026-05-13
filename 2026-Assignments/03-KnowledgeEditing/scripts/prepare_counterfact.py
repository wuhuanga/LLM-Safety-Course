import argparse
import json
import random
from pathlib import Path

import requests

DEFAULT_URLS = [
    "https://rome.baulab.info/data/dsets/counterfact.json",
    "https://rome.baulab.info/data/dsets/counterfact-train.json",
]


def _pick_text(value, default=""):
    if isinstance(value, dict):
        if "str" in value:
            return value["str"]
        if "text" in value:
            return value["text"]
    if isinstance(value, list) and value:
        return _pick_text(value[0], default)
    if isinstance(value, str):
        return value
    return default


def _pick_prompt_list(record, keys):
    for key in keys:
        val = record.get(key)
        if val:
            if isinstance(val, list) and val:
                if isinstance(val[0], dict) and "prompt" in val[0]:
                    return val[0]["prompt"]
                return _pick_text(val[0])
            return _pick_text(val)
    return None


def _normalize_prompt(prompt, subject):
    if prompt and subject and "{}" in prompt:
        try:
            return prompt.format(subject)
        except Exception:
            return prompt.replace("{}", subject)
    return prompt


def _convert_record(record):
    rr = record.get("requested_rewrite", {})
    subject = rr.get("subject") or record.get("subject")

    prompt = rr.get("prompt") or record.get("prompt")
    prompt = _normalize_prompt(_pick_text(prompt), subject)

    target_new = _pick_text(rr.get("target_new") or record.get("target_new"))
    ground_truth = _pick_text(rr.get("target_true") or rr.get("target_old") or record.get("ground_truth"))

    rephrase_prompt = _pick_prompt_list(record, ["paraphrase_prompts", "rephrase_prompts", "rephrase_prompt"])
    locality_prompt = _pick_prompt_list(record, ["neighborhood_prompts", "locality_prompts", "locality_prompt"])

    if not rephrase_prompt:
        rephrase_prompt = prompt
    if not locality_prompt:
        locality_prompt = prompt

    locality_ground_truth = _pick_text(record.get("locality_ground_truth")) or ground_truth

    if not all([prompt, target_new, ground_truth]):
        return None

    return {
        "prompt": prompt,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "rephrase_prompt": rephrase_prompt,
        "locality_prompt": locality_prompt,
        "locality_ground_truth": locality_ground_truth,
        "subject": subject or "",
    }


def _load_source(source):
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=60)
        resp.raise_for_status()
        return resp.json()
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="CounterFact json path or URL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = None
    if args.source:
        data = _load_source(args.source)
    else:
        for url in DEFAULT_URLS:
            try:
                data = _load_source(url)
                break
            except Exception:
                continue

    if data is None:
        raise RuntimeError("Failed to download CounterFact. Provide --source to a local file.")

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    converted = []
    for rec in data:
        item = _convert_record(rec)
        if item:
            converted.append(item)

    rnd = random.Random(args.seed)
    rnd.shuffle(converted)
    converted = converted[: args.size]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    print(f"Wrote {len(converted)} records to {out_path}")


if __name__ == "__main__":
    main()
