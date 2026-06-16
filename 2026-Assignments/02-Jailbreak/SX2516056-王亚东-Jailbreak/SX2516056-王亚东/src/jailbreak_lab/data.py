from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from .utils import redact_text, sha256_text, write_jsonl


PROMPT_KEYS = ["prompt", "goal", "Goal", "behavior", "Behavior", "instruction", "text"]
CATEGORY_KEYS = ["category", "Category", "harm_category", "type", "prompt_type", "label"]
ID_KEYS = ["id", "ID", "behavior_id", "BehaviorID", "idx"]


def _pick(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return default


def _first_split(ds: Any):
    if isinstance(ds, dict):
        # Prefer common split names, otherwise first available split.
        for name in ["train", "test", "validation", "eval"]:
            if name in ds:
                return ds[name]
        return next(iter(ds.values()))
    return ds


def load_harmful_jbb(dataset_name: str, min_categories: int, samples_per_category: int) -> List[Dict[str, Any]]:
    ds = _first_split(load_dataset(dataset_name))
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for i, row in enumerate(ds):
        prompt = _pick(row, PROMPT_KEYS)
        if not prompt:
            continue
        category = _pick(row, CATEGORY_KEYS, default="unknown")
        behavior_id = _pick(row, ID_KEYS, default=f"harmful-{i}")
        buckets[category].append(
            {
                "id": behavior_id,
                "category": category,
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "split": "harmful",
                "source": dataset_name,
            }
        )

    selected: List[Dict[str, Any]] = []
    for cat in sorted(buckets.keys())[: max(min_categories, 1)]:
        selected.extend(buckets[cat][:samples_per_category])

    if len({r["category"] for r in selected}) < min_categories:
        raise RuntimeError(
            f"Only found {len({r['category'] for r in selected})} categories; "
            f"need at least {min_categories}. Check dataset fields."
        )
    return selected


def _is_safe_xstest(row: Dict[str, Any]) -> bool:
    vals = " ".join(str(row.get(k, "")).lower() for k in ["label", "type", "prompt_type", "category"])
    if "unsafe" in vals:
        return False
    if "safe" in vals or "benign" in vals:
        return True
    # XSTest HF mirrors often have safe prompts first or a final label; fallback keeps all rows.
    return True


def load_benign_xstest(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    ds = _first_split(load_dataset(dataset_name))
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        prompt = _pick(row, PROMPT_KEYS)
        if not prompt or not _is_safe_xstest(row):
            continue
        category = _pick(row, CATEGORY_KEYS, default="xstest_safe")
        out.append(
            {
                "id": _pick(row, ID_KEYS, default=f"xstest-{i}"),
                "category": category,
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "split": "benign",
                "source": dataset_name,
            }
        )
        if len(out) >= n:
            break
    return out


def write_private_and_manifest(harmful: List[Dict[str, Any]], benign: List[Dict[str, Any]], harmful_out: str, benign_out: str, manifest_out: str) -> None:
    write_jsonl(harmful_out, harmful)
    write_jsonl(benign_out, benign)
    manifest = []
    for r in harmful + benign:
        manifest.append(
            {
                "id": r["id"],
                "split": r["split"],
                "category": r["category"],
                "prompt_hash": r["prompt_hash"],
                "prompt_redacted": redact_text(r["prompt"]),
                "source": r["source"],
            }
        )
    write_jsonl(manifest_out, manifest)
