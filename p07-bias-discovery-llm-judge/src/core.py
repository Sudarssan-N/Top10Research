
"""
P7: Automated Discovery of Novel Biases in LLM-as-Judge

>>> PIVOT NOTE (see ../research.md, 2026-06-29) <<<
The framing is substantially SCOOPED: BiasScope (arXiv:2602.09383, ICLR 2026) already
does automated discovery + correctness-preserving counterfactual perturbations. This
`BiasDiscoveryProbe` (a small MLP on concatenated pair embeddings) is NOT the real
method and should be treated as a placeholder. The defensible residual is CAUSAL
validation in the executably-verifiable code-judge domain (CodeJudgeBench): use an LLM
to generate quality-preserving perturbations of code answers where unit-test pass/fail
holds quality FIXED by construction, query the judge, and estimate a flip-rate ATE with
bootstrap CIs + mediation to rule out confounds. The real pipeline lives in evaluate.py
(prompting + API judge + causal estimator), not in a trained probe.
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
