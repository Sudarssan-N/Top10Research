#!/usr/bin/env python3
"""
Main entry point for P2: Calibrated, False-Positive-Bounded Verifiers for Best-of-N
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig
from src.utils import set_seed
from src.core import FPRBoundedVerifier as ProblemVerifier  # P2 specific
from src.train import train
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both", "phase1"], default="both")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny mock without loading big models")
    parser.add_argument("--output_dir", type=str, default=None)
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

    if args.mode == "phase1":
        from src.phase1 import run_phase1_ceiling, run_phase1_smoke
        if args.smoke_test:
            run_phase1_smoke(cfg, output_dir=args.output_dir)
        else:
            run_phase1_ceiling(cfg, mock=cfg.base_model.lower() in ("mock", "none", "smoke"), output_dir=args.output_dir)
        return

    if args.smoke_test:
        print("=== SMOKE TEST ===")
        model = ProblemVerifier(hidden_size=2048, probe_dim=cfg.controller_hidden_dim)
        dummy_hidden = torch.randn(4, 2048)
        out = model(dummy_hidden)
        print("Verifier output keys:", list(out.keys()))
        # Demo FPR / threshold finding (core P2 idea)
        dummy_scores = out["score"]
        dummy_labels = torch.tensor([1., 0., 1., 0.])
        thresh = model.find_threshold_for_target_fpr(dummy_scores, dummy_labels, target_fpr=0.1)
        metrics = model.compute_fpr_precision(dummy_scores, dummy_labels, thresh)
        print(f"Demo low-FPR threshold ~{thresh:.2f}: FPR={metrics['fpr']:.3f}, precision={metrics['precision']:.3f}")
        print("Smoke test passed.")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "smoke_results.json"), "w") as f:
            f.write('{"status": "ok", "mode": "smoke"}')
        return

    # TODO: real data loaders, real base model hidden extraction, real training
    model = ProblemVerifier(hidden_size=2048, probe_dim=cfg.controller_hidden_dim)

    if args.mode in ("train", "both"):
        from src.utils import get_dummy_loader
        dummy_loader = get_dummy_loader(num_batches=6, hidden_size=2048)
        model = train(cfg, dummy_loader, model, device)

    if args.mode in ("eval", "both"):
        metrics = evaluate_model(model, cfg)
        print("Eval metrics:", metrics)

    print("Done. See results/ for outputs.")

if __name__ == "__main__":
    main()
