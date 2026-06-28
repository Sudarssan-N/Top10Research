"""Best-of-N ceiling analysis under imperfect verifiers (Stroebl / Agrawal)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import precision_at_target_fpr


@dataclass
class CandidateRecord:
    completion: str
    is_correct: bool
    score: float


@dataclass
class SelectionDecision:
    strategy: str
    verifier: str
    n: int
    threshold: float
    chosen_completion: str
    is_correct: bool
    chosen_score: float
    had_correct_in_pool: bool
    verifier_accepted_wrong: bool = False
    metadata: Dict = field(default_factory=dict)


def select_argmax_verifier(
    candidates: Sequence[CandidateRecord],
    threshold: float = 0.0,
    strategy: str = "argmax",
) -> SelectionDecision:
    """Pick highest-scoring candidate (optionally gated by threshold)."""
    pool = list(candidates)
    if strategy == "conservative":
        pool = [c for c in pool if c.score >= threshold] or list(candidates)
    elif strategy == "pessimistic_min":
        # Use minimum score across candidates as conservative aggregate — pick lowest-risk among top half
        pool = sorted(pool, key=lambda c: c.score)[: max(1, len(pool) // 2)]

    chosen = max(pool, key=lambda c: c.score)
    had_correct = any(c.is_correct for c in candidates)
    return SelectionDecision(
        strategy=strategy,
        verifier="",
        n=len(candidates),
        threshold=threshold,
        chosen_completion=chosen.completion,
        is_correct=chosen.is_correct,
        chosen_score=chosen.score,
        had_correct_in_pool=had_correct,
        verifier_accepted_wrong=(not chosen.is_correct) and chosen.score >= threshold,
    )


def compute_confusion_at_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    threshold: float,
) -> Dict[str, float]:
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int64)
    preds = scores >= threshold
    tp = int((preds & (labels == 1)).sum())
    fp = int((preds & (labels == 0)).sum())
    tn = int((~preds & (labels == 0)).sum())
    fn = int((~preds & (labels == 1)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fp / (fp + tn + 1e-8),
        "fnr": fn / (fn + tp + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "threshold": threshold,
    }


def theoretical_ceiling_from_rates(
    p_correct_sample: float,
    verifier_fp_rate: float,
    verifier_fn_rate: float,
    n: int,
) -> float:
    """
    Approximate best-of-N ceiling when verifier picks argmax among N samples.

    Simplified Stroebl-style bound: at least one correct in pool with prob 1-(1-p)^N,
    but verifier may pick a wrong accepted sample due to FP.
    """
    p_pool = 1.0 - (1.0 - p_correct_sample) ** max(n, 1)
    # If a correct sample exists, verifier picks it with prob (1-fnr) among competing wrong highs
    p_pick_correct_given_pool = (1.0 - verifier_fn_rate) * p_pool
    # Residual failure: wrong sample gets top score
    p_fp_dominate = verifier_fp_rate * (1.0 - p_pool) + verifier_fn_rate * p_pool * 0.5
    return float(np.clip(p_pick_correct_given_pool - 0.5 * p_fp_dominate, 0.0, 1.0))


def estimate_scorer_snr(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> Dict[str, float]:
    """Dalal-style diagnostic: separation between correct/incorrect score distributions."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return {"delta": 0.0, "sigma": 1.0, "snr": 0.0, "k_hat": 1.0}
    delta = float(pos.mean() - neg.mean())
    sigma = float(np.sqrt(0.5 * (pos.var() + neg.var()) + 1e-8))
    snr = (delta / sigma) ** 2
    k_hat = max(1.0, 1.0 + snr)
    return {"delta": delta, "sigma": sigma, "snr": float(snr), "k_hat": float(k_hat)}


def accuracy_vs_n_curve(
    per_query_candidates: Dict[str, List[CandidateRecord]],
    n_values: Sequence[int],
    threshold: float = 0.5,
    strategy: str = "argmax",
) -> Dict[str, Dict]:
    """Sweep N and compute selection accuracy + FP acceptance rate."""
    curve = {}
    for n in n_values:
        decisions = []
        for qid, cands in per_query_candidates.items():
            subset = cands[:n]
            if not subset:
                continue
            decisions.append(select_argmax_verifier(subset, threshold=threshold, strategy=strategy))
        if not decisions:
            continue
        acc = sum(d.is_correct for d in decisions) / len(decisions)
        fp_accept = sum(d.verifier_accepted_wrong for d in decisions) / len(decisions)
        had_correct = sum(d.had_correct_in_pool for d in decisions) / len(decisions)
        curve[str(n)] = {
            "accuracy": acc,
            "fp_accept_rate": fp_accept,
            "pass_at_n": had_correct,
            "num_queries": len(decisions),
        }
    return curve


def aggregate_verifier_metrics(
    all_scores: Sequence[float],
    all_labels: Sequence[bool],
    target_fpr: float = 0.05,
) -> Dict:
    op = precision_at_target_fpr(all_scores, [int(x) for x in all_labels], target_fpr=target_fpr)
    snr = estimate_scorer_snr(all_scores, all_labels)
    return {"precision_at_fpr": op, "scorer_snr": snr}