from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def redact_text(text: str, keep: int = 24) -> str:
    """Return a non-actionable snippet for reports/logs.

    This deliberately avoids exposing full harmful prompts or model outputs.
    """
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return ""
    if len(text) <= keep:
        return text[: max(4, keep // 2)] + " ... [已脱敏]"
    return text[:keep] + " ... [已脱敏]"


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Timer:
    start: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def require_ack(ack: bool) -> None:
    if not ack:
        raise SystemExit(
            "This script can run adversarial safety tests. Re-run with "
            "--ack_responsible_use after reading RESPONSIBLE_USE.md."
        )


def private_path_warning(path: str | Path) -> None:
    path = str(path)
    if "private" not in path and "raw" not in path:
        print(
            "[warning] You are writing attack/evaluation outputs outside a private path. "
            "Do not commit raw harmful prompts, model outputs, or adversarial suffixes."
        )
