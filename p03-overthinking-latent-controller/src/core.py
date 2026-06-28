
"""
P3: Latent Controller for Adaptive Overthinking Reduction

Lightweight probe on frozen hidden states to predict convergence / "is it time to stop thinking?"
Outputs stop probability + helper to decide early termination during generation.
Reduces the ~18x token bloat on easy problems while preserving accuracy.
"""
import torch
import torch.nn as nn
from typing import Dict

class OverthinkController(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = hidden_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, probe_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = probe_dim
        self.backbone = nn.Sequential(*layers)
        self.stop_head = nn.Linear(probe_dim, 1)  # logit for "stop now / converged"

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(hidden_state)
        logit = self.stop_head(feat).squeeze(-1)
        stop_prob = torch.sigmoid(logit)
        return {"logit": logit, "stop_prob": stop_prob}

    def should_stop(self, hidden_state: torch.Tensor, threshold: float = 0.6, current_steps: int = 0, min_steps: int = 32) -> bool:
        """Simple policy: stop if prob high and past min reasoning steps."""
        if current_steps < min_steps:
            return False
        out = self.forward(hidden_state)
        return bool(out["stop_prob"].item() > threshold)

    def decide_early_stop_batch(self, hiddens: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
        """Vectorized decision for a batch of current hidden states."""
        out = self.forward(hiddens)
        return (out["stop_prob"] > threshold).float()
