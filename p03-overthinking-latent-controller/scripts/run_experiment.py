#!/usr/bin/env python3
"""
Main entry point for P3: Latent Controller for Adaptive Overthinking Reduction
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig
from src.utils import set_seed
from src.core import OverthinkController as ProblemController  # P3 specific
from src.train import train
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
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

    if args.smoke_test:
        print("=== SMOKE TEST ===")
        model = ProblemController(hidden_size=2048, probe_dim=cfg.controller_hidden_dim)
        dummy_hidden = torch.randn(4, 2048)
        out = model(dummy_hidden)
        print("Controller output keys:", list(out.keys()))
        # Demo early stop decision (P3 key)
        decisions = model.decide_early_stop_batch(dummy_hidden, threshold=0.55)
        print("Sample early-stop decisions (0=continue, 1=stop):", decisions.tolist())
        print("Smoke test passed.")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "smoke_results.json"), "w") as f:
            f.write('{"status": "ok", "mode": "smoke"}')
        return

    # TODO: real data loaders, real base model hidden extraction, real training
    model = ProblemController(hidden_size=2048, probe_dim=cfg.controller_hidden_dim)

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
