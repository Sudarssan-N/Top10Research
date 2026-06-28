
"""
P10: Dynamic Heterogeneous Drafter Selection for Speculative Decoding

Small router / bandit that chooses among heterogeneous drafters per context
for best acceptance rate + wall-clock speedup.
"""
import torch
import torch.nn as nn
from typing import Dict, List

class DrafterSelector(nn.Module):
    def __init__(self, hidden_size: int = 2048, num_drafters: int = 4, probe_dim: int = 32):
        super().__init__()
        self.num_drafters = num_drafters
        self.router = nn.Sequential(
            nn.Linear(hidden_size, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, num_drafters)
        )

    def forward(self, context_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = self.router(context_hidden)
        return {"drafter_logits": scores, "probs": torch.softmax(scores, dim=-1)}

    def select_drafter(self, context_hidden: torch.Tensor) -> int:
        out = self.forward(context_hidden)
        return int(torch.argmax(out["probs"], dim=-1).item())

    def estimate_acceptance(self, selected: int, context_features: torch.Tensor = None) -> float:
        # Placeholder; in real would be learned or profiled
        return 0.65 + 0.1 * (selected % 3)  # fake varying acceptance
