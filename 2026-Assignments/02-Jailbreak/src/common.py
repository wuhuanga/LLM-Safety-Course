import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer():
    box = {"latency_sec": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["latency_sec"] = time.perf_counter() - start


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["setting", "samples", "asr", "fpr", "avg_latency_sec", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def redact(text, max_len=180):
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"
