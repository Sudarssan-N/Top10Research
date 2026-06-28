
"""
P9: Token-Efficient Agent Memory with Bounded Reasoning Cost

Lightweight memory management policy: scores items for keep/forget to bound context.
"""
import torch
import torch.nn as nn
from typing import Dict, List

class MemoryPolicyProbe(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, 1)
        )

    def forward(self, item_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        keep_logit = self.net(item_hidden).squeeze(-1)
        return {"keep_score": torch.sigmoid(keep_logit), "logit": keep_logit}

    def select_memory(self, items: List[torch.Tensor], budget_tokens: int, per_item_cost: int = 50) -> List[int]:
        # Greedy keep by score
        scores = torch.stack([self.forward(it)["keep_score"] for it in items]).squeeze()
        k = max(1, min(len(items), budget_tokens // per_item_cost))
        _, idx = torch.topk(scores, k=k)
        return idx.tolist()
