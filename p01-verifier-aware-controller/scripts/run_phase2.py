#!/usr/bin/env python3
"""P1 Phase 2 — extract frozen hiddens + labels, train controller on cache."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.phase2 import run_phase2_extract, run_phase2_smoke, run_phase2_train


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="P1 Phase 2 feature pipeline")
    parser.add_argument("--config", type=str, default="configs/phase2.yaml")
    parser.add_argument("--mode", choices=["extract", "train", "both"], default="both")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--label-n", type=int, default=None)
    parser.add_argument("--pooling", type=str, choices=["last_token", "mean_prompt"], default=None)
    parser.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    data_dir = args.data_dir or cfg.data_dir
    pooling = args.pooling or cfg.feature_pooling
    label_n = args.label_n or cfg.label_n

    if args.smoke_test:
        run_phase2_smoke(cfg, output_dir=args.output_dir)
        run_phase2_train(cfg, mock=True, data_dir=os.path.join(cfg.output_dir, "phase2_smoke_data"), output_dir=args.output_dir or os.path.join(cfg.output_dir, "phase2_smoke"))
        print("\n=== Phase 2 smoke complete (extract + train) ===")
        return

    benchmarks = parse_str_list(args.benchmarks) if args.benchmarks else None

    if args.mode in ("extract", "both"):
        run_phase2_extract(
            cfg,
            benchmarks=benchmarks,
            max_examples=args.max_examples,
            label_n=label_n,
            pooling=pooling,
            mock=args.mock,
            data_dir=data_dir,
            output_dir=cfg.output_dir,
            force=args.force,
        )

    if args.mode in ("train", "both"):
        run_phase2_train(
            cfg,
            benchmarks=benchmarks,
            mock=args.mock,
            data_dir=data_dir,
            output_dir=cfg.output_dir,
        )


if __name__ == "__main__":
    main()