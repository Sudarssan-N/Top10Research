#!/usr/bin/env python3
"""
Download / prepare datasets for this problem.
For now just prints instructions; flesh out per-problem.
"""
from datasets import load_dataset
import os

def main():
    print("Preparing data (scaffold)...")
    print("MATH / GSM8K example:")
    try:
        ds = load_dataset("gsm8k", "main", split="train[:100]")
        print(f"  Loaded sample of GSM8K: {len(ds)} examples")
    except Exception as e:
        print("  Could not load (offline or network). Will need manual or cached data.")
    print("Datasets will be cached by huggingface under ~/.cache/huggingface")
    # TODO: add AIME download (many use https://github.com/hendrycks/math or community parquet)

if __name__ == "__main__":
    main()
