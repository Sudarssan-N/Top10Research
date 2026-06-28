#!/usr/bin/env python3
"""P1 Phase 3 — train ablations + controller-guided evaluation."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.phase3 import run_phase3_eval, run_phase3_full, run_phase3_smoke, run_phase3_train


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="P1 Phase 3")
    parser.add_argument("--config", type=str, default="configs/phase3.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both", "full"], default="both")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    data_dir = args.data_dir or cfg.data_dir
    benchmarks = parse_str_list(args.benchmarks) if args.benchmarks else None

    if args.smoke_test:
        run_phase3_smoke(cfg, output_dir=args.output_dir)
        print("\n=== Phase 3 smoke complete ===")
        return

    out_dir = args.output_dir or os.path.join(cfg.output_dir, "phase3")

    if args.mode == "full":
        run_phase3_full(cfg, max_examples=args.max_examples, mock=args.mock, data_dir=data_dir, output_dir=out_dir)
        return

    if args.mode in ("train", "both"):
        run_phase3_train(cfg, benchmarks=benchmarks, data_dir=data_dir, mock=args.mock, output_dir=out_dir)

    if args.mode in ("eval", "both"):
        run_phase3_eval(
            cfg,
            checkpoint_dir=out_dir,
            benchmarks=benchmarks,
            max_examples=args.max_examples,
            mock=args.mock,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()