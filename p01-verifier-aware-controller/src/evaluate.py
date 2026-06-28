import torch
from typing import Dict, List, Optional

from .config import ExperimentConfig


def evaluate_model(model, config: ExperimentConfig, split: str = "test") -> Dict:
    """Controller evaluation (Phase 2+). Phase 1 baselines use src.phase1 instead."""
    device = config.get_device()
    model.eval().to(device)
    print("[evaluate] Controller eval scaffold — use scripts/run_phase1_baselines.py for Phase 1.")
    return {"accuracy": 0.0, "avg_tokens": 0.0, "samples": 0}


def run_baselines(
    config: ExperimentConfig,
    max_examples: Optional[int] = None,
    mock: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """Entry point for Phase 1 baseline reproduction."""
    from .phase1 import run_phase1_baselines

    return run_phase1_baselines(
        config=config,
        max_examples=max_examples,
        mock=mock,
        output_dir=output_dir,
    )


def compute_ece(probs, labels, n_bins=10):
    import numpy as np

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return ece