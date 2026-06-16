"""Shared helpers for the knowledge editing assignment scripts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "prompt",
    "target_new",
    "ground_truth",
    "rephrase_prompt",
    "locality_prompt",
    "locality_ground_truth",
}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate.resolve()
    return Path(__file__).resolve().parent / candidate


def load_json(path: str | Path) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, payload: Any) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_custom_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {resolve_path(path)}.")

    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object.")
        missing = sorted(REQUIRED_FIELDS.difference(record))
        if missing:
            raise ValueError(f"Record {index} is missing required fields: {', '.join(missing)}")
        normalized = dict(record)
        normalized.setdefault("subject", infer_subject(normalized["prompt"]))
        validated.append(normalized)

    return validated[:limit] if limit is not None else validated


def first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return first_text(value[0])
    if isinstance(value, dict):
        if "str" in value:
            return first_text(value["str"])
        if "text" in value:
            return first_text(value["text"])
    return str(value)


def infer_subject(prompt: str) -> str:
    text = prompt.strip().rstrip("?")
    for suffix in [" is", " was", " are"]:
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)].strip()
    lowered = text.lower()
    prefixes = ["the current ", "current "]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            lowered = text.lower()
            break

    for prefix in ["ceo of ", "prime minister of ", "president of ", "chief executive of "]:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()

    for separator in [" is ", " was ", " are ", " belongs to ", " belong to "]:
        position = lowered.find(separator)
        if position > 0:
            return text[:position].strip()

    if text.lower().startswith("who is "):
        return text[7:].strip()
    if text.lower().startswith("which "):
        return text.split(" does ", 1)[-1].split(" belong", 1)[0].strip()
    return text.strip()


def normalize_edit_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    prompt = raw.get("prompt")
    target_new = raw.get("target_new")
    if isinstance(target_new, dict):
        target_new = target_new.get("str") or target_new.get("text")
    ground_truth = raw.get("ground_truth") or raw.get("target_true")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("str") or ground_truth.get("text")

    if not prompt or not target_new or ground_truth is None:
        requested = raw.get("requested_rewrite")
        if isinstance(requested, dict):
            prompt = prompt or requested.get("prompt")
            target_new = target_new or first_text(requested.get("target_new"))
            ground_truth = ground_truth if ground_truth is not None else first_text(requested.get("target_true"))

    if not prompt or not target_new or ground_truth is None:
        return None

    rephrase_prompt = raw.get("rephrase_prompt")
    if not rephrase_prompt:
        paraphrases = raw.get("paraphrase_prompts") or raw.get("paraphrase_prompt")
        rephrase_prompt = first_text(paraphrases) if paraphrases else prompt

    locality_prompt = raw.get("locality_prompt")
    locality_ground_truth = raw.get("locality_ground_truth")
    locality = raw.get("locality")
    if (not locality_prompt or locality_ground_truth is None) and isinstance(locality, dict):
        for examples in locality.values():
            if isinstance(examples, list) and examples:
                example = examples[0]
                if isinstance(example, dict):
                    locality_prompt = locality_prompt or example.get("prompt")
                    locality_ground_truth = (
                        locality_ground_truth
                        if locality_ground_truth is not None
                        else first_text(example.get("ground_truth"))
                    )
                    break

    if not locality_prompt:
        locality_prompt = prompt
    if locality_ground_truth is None:
        locality_ground_truth = first_text(ground_truth)

    subject = raw.get("subject") or infer_subject(str(prompt))
    return {
        "subject": str(subject),
        "prompt": str(prompt),
        "target_new": first_text(target_new),
        "ground_truth": first_text(ground_truth),
        "rephrase_prompt": str(rephrase_prompt),
        "locality_prompt": str(locality_prompt),
        "locality_ground_truth": first_text(locality_ground_truth),
    }


def load_benchmark_records(
    *,
    repo_id: str,
    repo_file: str,
    limit: int,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency huggingface_hub. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=repo_file,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    raw_records = load_json(downloaded)
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a JSON list in downloaded benchmark file {downloaded}.")

    records: list[dict[str, Any]] = []
    for raw in raw_records:
        if isinstance(raw, dict):
            normalized = normalize_edit_record(raw)
            if normalized is not None:
                records.append(normalized)
        if len(records) >= limit:
            break

    if len(records) < limit:
        raise ValueError(f"Only found {len(records)} usable records in {downloaded}; requested {limit}.")
    return records


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def device_for_yaml(device: str) -> int | str:
    return int(device) if device.isdigit() else device


def write_runtime_hparams(config_path: str | Path, *, model_name: str, device: str) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency pyyaml. Install it with: pip install pyyaml") from exc

    config = load_json_like_yaml(config_path)
    config["model_name"] = model_name
    config["device"] = device_for_yaml(device)

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="easyedit_hparams_",
        encoding="utf-8",
        delete=False,
    )
    with temp:
        yaml.safe_dump(config, temp, sort_keys=False, allow_unicode=True)
    return Path(temp.name)


def load_json_like_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency pyyaml. Install it with: pip install pyyaml") from exc

    with resolve_path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {resolve_path(path)}.")
    return data


def maybe_cuda_start() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.reset_peak_memory_stats()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_mb_start": round(torch.cuda.memory_allocated() / 1024**2, 2),
    }


def maybe_cuda_finish(stats: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return stats

    if not torch.cuda.is_available():
        return stats
    stats = dict(stats)
    stats.update(
        {
            "memory_allocated_mb_end": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }
    )
    return stats


def cleanup_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_editor_tokenizer(editor: Any) -> Any:
    for attr in ("tok", "tokenizer"):
        tokenizer = getattr(editor, attr, None)
        if tokenizer is not None:
            return tokenizer
    raise AttributeError("Could not find tokenizer on EasyEdit editor; expected editor.tok or editor.tokenizer.")
