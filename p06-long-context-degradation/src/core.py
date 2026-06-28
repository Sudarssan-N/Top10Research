
"""
P6: Context-Length-Induced Degradation Under Perfect Retrieval

Probe for attention dilution / degradation signal + simple mitigation (reorder suggestion or calibration).
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
