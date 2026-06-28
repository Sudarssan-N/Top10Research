
"""
P8: Step-Wise On-Policy Distillation for Small Reasoning Models (Tool Use)

Probe for divergence reweighting during distillation + step importance.
"""
import torch
import torch.nn as nn
from typing import Dict

class DistillationReweightProbe(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, 1)
        )

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        w = self.net(hidden).squeeze(-1)
        return {"weight_logit": w, "importance": torch.sigmoid(w)}

    def reweight_loss(self, student_logits, teacher_logits, step_importance):
        # Placeholder for on-policy step-wise reweight
        return (step_importance * (student_logits - teacher_logits).pow(2)).mean()
