
"""
P7: Automated Discovery of Novel Biases in LLM-as-Judge

Contrastive probe + bias detector. Takes pair embeddings or judge logits,
outputs bias score + suggested perturbation for discovery.
"""
import torch
import torch.nn as nn
from typing import Dict

class BiasDiscoveryProbe(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, probe_dim),  # concat of pair
            nn.ReLU(),
            nn.Linear(probe_dim, 1)
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        concat = torch.cat([emb_a, emb_b], dim=-1)
        bias_logit = self.net(concat).squeeze(-1)
        return {"bias_score": torch.sigmoid(bias_logit), "logit": bias_logit}

    def detect_bias(self, score_diff: torch.Tensor) -> Dict[str, float]:
        # Demo: magnitude indicates bias strength
        return {"bias_strength": score_diff.abs().mean().item()}
