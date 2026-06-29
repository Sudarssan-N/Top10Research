
"""
P6: Context-Length-Induced Degradation Under Perfect Retrieval

>>> PIVOT NOTE (see ../research.md, 2026-06-29) <<<
The scaffolded plan (show length hurts, then reorder/calibrate) is already published
(Found in the Middle arXiv:2406.16008; the anchor arXiv:2510.05381 ships a recitation
fix). So `ContextDegradationProbe` below is DEMOTED to an optional diagnostic, NOT the
contribution. The real contribution is the controlled experiment in `evaluate.py`:
orthogonalize {token count} x {absolute position of the gold evidence} under
certified-perfect retrieval, show degradation persists when attention dilution is
masked out (-> mechanism is positional/RoPE, not dilution), then build a
mechanism-targeted training-free fix that beats recitation.
"""
import torch
import torch.nn as nn
from typing import Dict, List

class ContextDegradationProbe(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, 1)
        )

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        deg_logit = self.net(hidden).squeeze(-1)
        return {"degradation_score": torch.sigmoid(deg_logit), "logit": deg_logit}

    def suggest_reorder(self, positions: List[int], degradation_scores: torch.Tensor) -> List[int]:
        """Simple: move high-degradation (middle?) items or prioritize low-deg first."""
        # Demo: sort by increasing degradation (put "safe" context first)
        idx = torch.argsort(degradation_scores)
        return [positions[i] for i in idx.tolist()]
