
"""
P1 Core: Verifier-Aware Test-Time Compute Controller

A small MLP or transformer head on top of frozen base model hidden states
that predicts:
  - difficulty / expected gain from more compute
  - verifier reliability / false-positive risk for this query
Then uses a stopping policy (Gittins / Pandora style or learned threshold)
to decide how much compute (samples / tokens) to allocate.
"""
import torch
import torch.nn as nn
from typing import Dict, Literal, Union

ControllerVariant = Literal["verifier_aware", "difficulty_only"]


class _ControllerBackbone(nn.Module):
    def __init__(self, hidden_size: int, controller_dim: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = hidden_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, controller_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = controller_dim
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(controller_dim, 1)


class VerifierAwareController(_ControllerBackbone):
    def __init__(self, hidden_size: int = 2048, controller_dim: int = 128, num_layers: int = 2):
        super().__init__(hidden_size, controller_dim, num_layers)
        self.trust_head = nn.Linear(controller_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.variant = "verifier_aware"

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hidden_state: [batch, hidden] or mean-pooled
        feat = self.backbone(hidden_state)
        value = self.value_head(feat).squeeze(-1)
        trust = self.sigmoid(self.trust_head(feat)).squeeze(-1)  # 0..1
        return {"value": value, "verifier_trust": trust}

    def decide_num_samples(
        self,
        value: torch.Tensor,
        trust: torch.Tensor,
        min_n: int = 1,
        max_n: int = 16,
    ) -> torch.Tensor:
        """Map value + trust to integer sample count k ∈ [min_n, max_n]."""
        value = torch.clamp(value, 0.0, 1.0)
        trust = torch.clamp(trust, 0.25, 1.0)
        raw = min_n + (max_n - min_n) * value * trust
        return raw.round().to(torch.int64).clamp(min_n, max_n)

    def decide_budget(
        self,
        value: torch.Tensor,
        trust: torch.Tensor,
        base_budget: int = 256,
        max_budget: int = 2048,
        trust_threshold: float = 0.6,
    ) -> torch.Tensor:
        scale = torch.clamp(trust, 0.3, 1.0)
        budget = (base_budget + (value * 1024) * scale).clamp(base_budget, max_budget).int()
        return budget


class DifficultyOnlyController(_ControllerBackbone):
    """Ablation: value head only; trust fixed to 1.0 at inference."""

    def __init__(self, hidden_size: int = 2048, controller_dim: int = 128, num_layers: int = 2):
        super().__init__(hidden_size, controller_dim, num_layers)
        self.variant = "difficulty_only"

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(hidden_state)
        value = self.value_head(feat).squeeze(-1)
        trust = torch.ones_like(value)
        return {"value": value, "verifier_trust": trust}

    def decide_num_samples(self, value: torch.Tensor, trust: torch.Tensor, min_n: int = 1, max_n: int = 16) -> torch.Tensor:
        value = torch.clamp(value, 0.0, 1.0)
        raw = min_n + (max_n - min_n) * value
        return raw.round().to(torch.int64).clamp(min_n, max_n)

    def decide_budget(self, value: torch.Tensor, trust: torch.Tensor, base_budget: int = 256, max_budget: int = 2048, trust_threshold: float = 0.6) -> torch.Tensor:
        budget = (base_budget + (value * 1024)).clamp(base_budget, max_budget).int()
        return budget


def build_controller(
    variant: ControllerVariant = "verifier_aware",
    hidden_size: int = 2048,
    controller_dim: int = 128,
    num_layers: int = 2,
) -> nn.Module:
    if variant == "difficulty_only":
        return DifficultyOnlyController(hidden_size, controller_dim, num_layers)
    return VerifierAwareController(hidden_size, controller_dim, num_layers)
