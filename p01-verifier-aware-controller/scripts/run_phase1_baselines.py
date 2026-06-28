#!/usr/bin/env python3
"""P1 Phase 1 — reproduce test-time compute baselines (best-of-N Pareto curves)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.phase1 import run_phase1_baselines, run_phase1_smoke


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="P1 Phase 1 baseline reproduction")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Mock generator + 6 examples (no GPU/HF)")
    parser.add_argument("--mock", action="store_true", help="Use MockGenerator even if model is real")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated: gsm8k,math500,aime24,mock")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap examples per benchmark (debug)")
    parser.add_argument("--n-values", type=str, default=None, help="Comma-separated N for best-of-N, e.g. 1,2,4,8,16")
    parser.add_argument("--strategies", type=str, default=None, help="Comma-separated baseline strategies")
    parser.add_argument("--token-budgets", type=str, default=None, help="Fixed max_new_tokens sweep, e.g. 128,256,512")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.smoke_test:
        results = run_phase1_smoke(cfg, output_dir=args.output_dir)
        print("\n=== Phase 1 smoke complete ===")
        for bench, data in results.get("benchmarks", {}).items():
            print(f"\n{bench}:")
            for n, smap in data.get("best_of_n", {}).get("sweep", {}).items():
                for strat, m in smap.items():
                    print(f"  N={n} {strat}: acc={m['accuracy']:.3f} avg_samples={m['avg_samples']:.1f}")
        return

    benchmarks = parse_str_list(args.benchmarks) if args.benchmarks else None
    n_values = parse_int_list(args.n_values) if args.n_values else None
    strategies = parse_str_list(args.strategies) if args.strategies else None
    token_budgets = parse_int_list(args.token_budgets) if args.token_budgets else None

    run_phase1_baselines(
        config=cfg,
        benchmarks=benchmarks,
        max_examples=args.max_examples,
        n_values=n_values,
        strategies=strategies,
        mock=args.mock,
        token_budgets=token_budgets,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()