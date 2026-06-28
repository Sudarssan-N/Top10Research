#!/usr/bin/env python3
"""
Main entry point for P7: Automated Discovery of Novel Biases in LLM-as-Judge
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig
from src.utils import set_seed
from src.core import BiasDiscoveryProbe as ProblemProbe  # P7 specific
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
        model = ProblemProbe(hidden_size=2048)
        emb_a = torch.randn(4, 2048)
        emb_b = torch.randn(4, 2048)
        out = model(emb_a, emb_b)
        print("Probe output keys:", list(out.keys()))
        b = model.detect_bias(out["bias_score"])
        print(f"Demo bias strength: {b['bias_strength']:.4f}")
        print("Smoke test passed.")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "smoke_results.json"), "w") as f:
            f.write('{"status": "ok", "mode": "smoke"}')
        return

    model = ProblemProbe(hidden_size=2048)

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
