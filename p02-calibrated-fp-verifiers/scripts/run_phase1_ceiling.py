#!/usr/bin/env python3
"""P2 Phase 1: ceiling analysis and verifier baseline reproduction."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.phase1 import run_phase1_ceiling, run_phase1_smoke


def _parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="P2 Phase 1 — verifier ceiling baselines")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use mock generator")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated: math500,gsm8k,mock")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-values", type=str, default=None, help="Comma-separated N values, e.g. 1,2,4,8,16")
    parser.add_argument("--verifiers", type=str, default=None, help="Comma-separated verifier names")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.base_model:
        cfg.base_model = args.base_model
    if args.benchmarks:
        cfg.benchmarks = _parse_list(args.benchmarks)

    if args.smoke_test:
        run_phase1_smoke(cfg, output_dir=args.output_dir)
        return

    n_values = [int(x) for x in _parse_list(args.n_values)] if args.n_values else None
    verifier_names = _parse_list(args.verifiers) if args.verifiers else None
    mock = args.mock or cfg.base_model.lower() in ("mock", "none", "smoke")

    run_phase1_ceiling(
        config=cfg,
        benchmarks=cfg.benchmarks,
        max_examples=args.max_examples,
        n_values=n_values,
        verifier_names=verifier_names,
        mock=mock,
        target_fpr=args.target_fpr,
        output_dir=args.output_dir or os.path.join(cfg.output_dir, "phase1"),
    )


if __name__ == "__main__":
    main()