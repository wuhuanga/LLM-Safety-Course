"""Sample ZsRE records from the KnowEdit dataset for MEMIT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from editing_utils import load_benchmark_records, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and sample ZsRE records used by MEMIT.")
    parser.add_argument("--output", default="data/ZsRE.json", help="Path to save sampled records.")
    parser.add_argument("--limit", type=int, default=500, help="Number of ZsRE records to sample.")
    parser.add_argument("--benchmark-repo", default="zjunlp/KnowEdit", help="Hugging Face dataset repo.")
    parser.add_argument(
        "--benchmark-file",
        default="benchmark/ZsRE/ZsRE-test-all.json",
        help="File inside the Hugging Face dataset repo.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_benchmark_records(
        repo_id=args.benchmark_repo,
        repo_file=args.benchmark_file,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    save_json(args.output, records)

    print("=" * 80)
    print("ZsRE SAMPLE SUMMARY")
    print("=" * 80)
    print(f"Total sampled records: {len(records)}")
    print(f"Output file: {Path(args.output)}")
    if records:
        print("First record:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
