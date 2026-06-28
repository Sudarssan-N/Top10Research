"""Calibration metrics and post-hoc temperature scaling for verifier scores."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_ece(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 15,
    adaptive: bool = True,
) -> float:
    """Expected Calibration Error with optional adaptive binning (Nixon et al. 2019)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return 0.0

    if adaptive:
        order = np.argsort(probs)
        probs = probs[order]
        labels = labels[order]
        bin_ids = np.floor(np.linspace(0, len(probs), n_bins + 1)).astype(int)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_ids[i], bin_ids[i + 1]
            if hi <= lo:
                continue
            mask = slice(lo, hi)
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += (hi - lo) / len(probs) * abs(bin_acc - bin_conf)
        return float(ece)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (probs > lo) & (probs <= hi)
        else:
            mask = (probs > lo) & (probs <= hi + 1e-8)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, list]:
    """Return per-bin confidence/accuracy for reliability diagrams."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (probs > lo) & (probs <= hi)
        else:
            mask = (probs > lo) & (probs <= hi + 1e-8)
        if mask.sum() == 0:
            continue
        confs.append(float(probs[mask].mean()))
        accs.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))
    return {"confidence": confs, "accuracy": accs, "counts": counts}


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.05,
) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017)."""
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)

    temp = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits / temp.clamp(min=1e-3)
        loss = F.binary_cross_entropy_with_logits(scaled, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temp.detach().clamp(min=1e-3).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-3)


def precision_at_target_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    target_fpr: float = 0.05,
) -> Dict[str, float]:
    """Find threshold achieving ~target FPR and report precision/recall."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    best = {"threshold": 0.5, "fpr": 1.0, "precision": 0.0, "recall": 0.0}
    best_diff = float("inf")
    for t in np.linspace(0.0, 1.0, 201):
        preds = scores >= t
        tp = int((preds & (labels == 1)).sum())
        fp = int((preds & (labels == 0)).sum())
        tn = int((~preds & (labels == 0)).sum())
        fn = int((~preds & (labels == 1)).sum())
        fpr = fp / (fp + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        diff = abs(fpr - target_fpr)
        if diff < best_diff:
            best_diff = diff
            best = {
                "threshold": float(t),
                "fpr": float(fpr),
                "precision": float(precision),
                "recall": float(recall),
            }
    best["target_fpr"] = target_fpr
    return best