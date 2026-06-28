#!/usr/bin/env python3
"""Download / cache benchmarks used in P1 Phase 1."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks import BENCHMARK_SPECS, load_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="gsm8k,math500,aime24",
        help="Comma-separated benchmark names",
    )
    parser.add_argument("--max-examples", type=int, default=5, help="Rows to load per benchmark (warm cache)")
    args = parser.parse_args()

    names = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    print("Phase 1 datasets (cached under ~/.cache/huggingface):")
    for spec_name, spec in BENCHMARK_SPECS.items():
        print(f"  - {spec_name}: {spec['hf_id']} [{spec['split']}]")

    for name in names:
        print(f"\nLoading {name}...")
        try:
            examples = load_benchmark(name, max_examples=args.max_examples)
            print(f"  OK — {len(examples)} examples (preview max={args.max_examples})")
            if examples:
                print(f"  Sample prompt: {examples[0].prompt[:120]}...")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
