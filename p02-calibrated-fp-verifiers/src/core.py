
"""
P2: Calibrated, False-Positive-Bounded Verifiers for Best-of-N

Lightweight probe on frozen base hidden states that acts as a verifier.
Trained with precision-first / asymmetric loss to minimize FPR at operating points
relevant to best-of-N selection. Includes threshold tuning for target FPR and
ceiling analysis helpers.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

class FPRBoundedVerifier(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 128, num_layers: int = 2, alpha: float = 0.8):
        super().__init__()
        layers = []
        in_dim = hidden_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, probe_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = probe_dim
        self.backbone = nn.Sequential(*layers)
        self.score_head = nn.Linear(probe_dim, 1)  # logit for "is_correct / high quality"
        self.alpha = alpha  # precision emphasis weight (used in loss externally)

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hidden_state: [batch, hidden]
        feat = self.backbone(hidden_state)
        logit = self.score_head(feat).squeeze(-1)
        score = torch.sigmoid(logit)
        return {"logit": logit, "score": score}

    def predict(self, hidden_state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        out = self.forward(hidden_state)
        return (out["score"] >= threshold).float()

    def compute_fpr_precision(self, scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, float]:
        """Compute FPR and precision at a given threshold (for best-of-N ceiling analysis)."""
        preds = (scores >= threshold).float()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        fpr = fp / (fp + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return {"fpr": fpr, "precision": precision, "recall": recall, "threshold": threshold}

    def find_threshold_for_target_fpr(self, scores: torch.Tensor, labels: torch.Tensor, target_fpr: float = 0.05) -> float:
        """Binary search for threshold achieving approx target FPR (key for bounded selection)."""
        thresholds = torch.linspace(0, 1, 101)
        best_thresh = 0.5
        best_diff = float("inf")
        for t in thresholds:
            m = self.compute_fpr_precision(scores, labels, t.item())
            diff = abs(m["fpr"] - target_fpr)
            if diff < best_diff:
                best_diff = diff
                best_thresh = t.item()
        return best_thresh
