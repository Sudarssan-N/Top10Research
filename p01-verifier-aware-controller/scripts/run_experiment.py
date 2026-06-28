#!/usr/bin/env python3
"""
Main entry point for P1: Verifier-Aware Learned Controller for Test-Time Compute
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig
from src.utils import set_seed
from src.core import VerifierAwareController as ProblemController   # P1 specific
from src.train import train
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both", "phase1", "phase2", "phase3"], default="both")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny mock without loading big models")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None, help="Phase 1: cap examples per benchmark")
    parser.add_argument("--mock", action="store_true", help="Phase 1/2: use mock generator/backbone")
    parser.add_argument("--force", action="store_true", help="Phase 2: re-extract features")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir

    set_seed(cfg.seed)
    device = cfg.get_device()
    print(f"Running in mode={args.mode} on device={device}")
    print(f"Config: base_model={cfg.base_model}")

    if args.smoke_test:
        print("=== SMOKE TEST ===")
        model = ProblemController(hidden_size=2048, controller_dim=cfg.controller_hidden_dim)
        dummy_hidden = torch.randn(4, 2048)
        out = model(dummy_hidden)
        print("Controller output keys:", list(out.keys()))
        # Demo budget decision (P1 key contribution)
        budgets = model.decide_budget(out["value"], out["verifier_trust"], base_budget=128, max_budget=1024)
        print("Sample decided budgets:", budgets.tolist())
        print("Smoke test passed.")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "smoke_results.json"), "w") as f:
            f.write('{"status": "ok", "mode": "smoke"}')
        return

    if args.mode == "phase1":
        from src.phase1 import run_phase1_baselines, run_phase1_smoke
        if args.smoke_test:
            run_phase1_smoke(cfg, output_dir=args.output_dir)
        else:
            run_phase1_baselines(cfg, max_examples=args.max_examples, mock=args.mock, output_dir=args.output_dir)
        return

    if args.mode == "phase3":
        from src.phase3 import run_phase3_smoke, run_phase3_full
        if args.smoke_test:
            run_phase3_smoke(cfg, output_dir=args.output_dir)
        else:
            run_phase3_full(
                cfg,
                max_examples=args.max_examples,
                mock=args.mock,
                output_dir=args.output_dir or os.path.join(cfg.output_dir, "phase3"),
            )
        return

    if args.mode == "phase2":
        from src.phase2 import run_phase2_extract, run_phase2_smoke, run_phase2_train
        if args.smoke_test:
            run_phase2_smoke(cfg, output_dir=args.output_dir)
            run_phase2_train(
                cfg,
                mock=True,
                data_dir=os.path.join(cfg.output_dir, "phase2_smoke_data"),
                output_dir=args.output_dir or os.path.join(cfg.output_dir, "phase2_smoke"),
            )
        else:
            run_phase2_extract(
                cfg,
                max_examples=args.max_examples,
                mock=args.mock,
                force=args.force,
                output_dir=args.output_dir,
            )
            run_phase2_train(cfg, mock=args.mock, output_dir=args.output_dir)
        return

    # Train from cached Phase 2 features if available, else synthetic demo
    from src.dataset import merge_feature_caches, build_dataloaders
    from src.features import feature_cache_path

    feature_paths = [
        feature_cache_path(cfg.data_dir, cfg.base_model, b) for b in cfg.benchmarks
    ]
    try:
        meta, records = merge_feature_caches(feature_paths)
        hidden_size = int(meta.get("hidden_size", 2048))
        print(f"Loading {len(records)} cached features (hidden_size={hidden_size})")
        model = ProblemController(
            hidden_size=hidden_size,
            controller_dim=cfg.controller_hidden_dim,
            num_layers=cfg.num_controller_layers,
        )
        train_loader, _ = build_dataloaders(records, batch_size=cfg.train_batch_size, seed=cfg.seed)
    except (FileNotFoundError, RuntimeError):
        print("No feature cache found — using synthetic demo loader. Run phase2 first.")
        model = ProblemController(hidden_size=2048, controller_dim=cfg.controller_hidden_dim)
        from src.utils import get_dummy_loader
        train_loader = get_dummy_loader(num_batches=6, hidden_size=2048)

    if args.mode in ("train", "both"):
        model = train(cfg, train_loader, model, device)

    if args.mode in ("eval", "both"):
        metrics = evaluate_model(model, cfg)
        print("Eval metrics:", metrics)

    print("Done. See results/ for outputs.")

if __name__ == "__main__":
    main()
