import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EASYEDIT_ROOT = PROJECT_ROOT / "EasyEdit"
if str(EASYEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYEDIT_ROOT))

from easyeditor import BaseEditor, ROMEHyperParams


DATA_PATH = Path("data/custom_edits.json")
OUTPUT_PATH = Path("results/rome_results.json")
HPARAMS_PATH = Path("EasyEdit/hparams/ROME/qwen2.5-0.5b.yaml")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_subject(prompt: str) -> str:
    fallback_rules = [
        ("The CEO of ", " is"),
        ("The head coach of ", " is"),
        ("The prime minister of ", " is"),
        ("The monarch of ", " is"),
        ("The president of ", " is"),
        ("The capital of ", " is"),
    ]
    for prefix, suffix in fallback_rules:
        if prompt.startswith(prefix) and prompt.endswith(suffix):
            return prompt[len(prefix):-len(suffix)].strip()

    raise ValueError(f"Could not infer subject from prompt: {prompt}")


def extract_core_metrics(metric: dict) -> dict:
    pre = metric.get("pre", {})
    post = metric.get("post", {})
    locality_post = post.get("locality", {})

    locality_acc = None
    for key, value in locality_post.items():
        if key.endswith("_acc"):
            locality_acc = value
            break

    return {
        "case_id": metric.get("case_id"),
        "time": metric.get("time"),
        "requested_rewrite": metric.get("requested_rewrite"),
        "pre": {
            "rewrite_acc": pre.get("rewrite_acc"),
            "rephrase_acc": pre.get("rephrase_acc"),
            "rewrite_ppl": pre.get("rewrite_ppl"),
            "ood_acc": pre.get("ood_acc"),
        },
        "post": {
            "rewrite_acc": post.get("rewrite_acc"),
            "rephrase_acc": post.get("rephrase_acc"),
            "rewrite_ppl": post.get("rewrite_ppl"),
            "ood_acc": post.get("ood_acc"),
            "locality_acc": locality_acc,
        },
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if not HPARAMS_PATH.exists():
        raise FileNotFoundError(f"ROME hparams file not found: {HPARAMS_PATH}")

    samples = load_json(DATA_PATH)
    hparams = ROMEHyperParams.from_hparams(str(HPARAMS_PATH))

    all_results = []

    for idx, item in enumerate(samples, start=1):
        prompt = item["prompt"]
        target_new = item["target_new"]
        ground_truth = item["ground_truth"]
        rephrase_prompt = item["rephrase_prompt"]
        locality_prompt = item["locality_prompt"]
        locality_ground_truth = item["locality_ground_truth"]
        subject = build_subject(prompt)

        print(f"\n===== ROME Edit [{idx}/{len(samples)}] =====")
        print(f"Prompt: {prompt}")
        print(f"Target new: {target_new}")
        print(f"Ground truth: {ground_truth}")
        print(f"Subject: {subject}")

        editor = BaseEditor.from_hparams(hparams)
        metrics, _, _ = editor.edit(
            prompts=prompt,
            target_new=target_new,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompt,
            locality_inputs={
                "neighborhood": {
                    "prompt": locality_prompt,
                    "ground_truth": locality_ground_truth,
                }
            },
            subject=subject,
            sequential_edit=False,
            verbose=False,
        )

        raw_metric = metrics[0]
        result = {
            "id": idx,
            "prompt": prompt,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "rephrase_prompt": rephrase_prompt,
            "locality_prompt": locality_prompt,
            "locality_ground_truth": locality_ground_truth,
            "subject": subject,
            "metrics": extract_core_metrics(raw_metric),
            "raw_metrics": raw_metric,
        }
        all_results.append(result)
        save_json(all_results, OUTPUT_PATH)

        post_metrics = result["metrics"]["post"]
        print("Post-edit summary:")
        print(f"  rewrite_acc: {post_metrics['rewrite_acc']}")
        print(f"  rephrase_acc: {post_metrics['rephrase_acc']}")
        print(f"  locality_acc: {post_metrics['locality_acc']}")

    print("\n===== ROME Editing Finished =====")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
