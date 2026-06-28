import torch
from typing import Dict, List
from .config import ExperimentConfig
# Lazy imports inside functions for heavy optional deps (datasets, sympy, etc.)
# from datasets import load_dataset
# TODO: add generation + math eval code using sympy for answer extraction etc.

def evaluate_model(model, config: ExperimentConfig, split: str = "test") -> Dict:
    # Lazy import
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
    device = config.get_device()
    model.eval().to(device)

    results = {"accuracy": 0.0, "avg_tokens": 0.0, "samples": 0}
    # TODO: implement full generation loop over benchmarks
    # For now return dummy
    print("[evaluate] Running evaluation scaffold (replace with real generations + grading)")
    return results

def compute_ece(probs, labels, n_bins=10):
    # Expected Calibration Error helper (useful for P4 and verifier work)
    import numpy as np
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0: continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return ece
