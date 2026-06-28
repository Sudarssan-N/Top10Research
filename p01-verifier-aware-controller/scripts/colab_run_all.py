#!/usr/bin/env python3
"""
Run P1 Phases 1→2→3 on Colab GPU with sensible caps.

Example (in Colab after cd to project):
  python scripts/colab_run_all.py --max-examples 200 --label-n 8
  python scripts/colab_run_all.py --smoke-test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.phase1 import run_phase1_baselines
from src.phase2 import run_phase2_extract
from src.phase3 import run_phase3_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colab.yaml")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--label-n", type=int, default=8)
    parser.add_argument("--n-values", default="1,4,8,16")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    cfg.label_n = args.label_n
    cfg.n_samples_best_of_n = max(cfg.n_samples_best_of_n, args.label_n)

    import torch
    print("=" * 60)
    print("P1 Colab pipeline")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  base_model: {cfg.base_model}")
    print(f"  max_examples: {args.max_examples}")
    print("=" * 60)

    if args.smoke_test:
        from src.phase3 import run_phase3_smoke
        run_phase3_smoke(cfg)
        return

    out = cfg.output_dir
    n_values = [int(x) for x in args.n_values.split(",")]

    if not args.skip_phase1:
        print("\n>>> Phase 1 baselines")
        run_phase1_baselines(
            cfg,
            max_examples=args.max_examples,
            n_values=n_values,
            output_dir=os.path.join(out, "phase1"),
        )

    if not args.skip_extract:
        print("\n>>> Phase 2 feature extract")
        run_phase2_extract(
            cfg,
            max_examples=args.max_examples,
            label_n=args.label_n,
            output_dir=os.path.join(out, "phase2"),
            force=True,
        )

    print("\n>>> Phase 3 train + eval")
    run_phase3_full(
        cfg,
        max_examples=args.max_examples,
        data_dir=cfg.data_dir,
        output_dir=os.path.join(out, "phase3"),
    )

    print("\n" + "=" * 60)
    print("DONE. Artifacts under:", out)
    print("  phase1/phase1_summary.json")
    print("  phase2/phase2_extract_summary.json")
    print("  data/features/...")
    print("  phase3/phase3_train_summary.json")
    print("  phase3/phase3_eval_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()