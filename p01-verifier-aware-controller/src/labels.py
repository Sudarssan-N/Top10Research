"""Offline labels for controller training (value + verifier trust)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .baselines import SampleRecord, majority_vote
from .benchmarks import BenchmarkExample
from .grading import grade_completion, normalize_answer


@dataclass
class ControllerLabels:
    target_value: float
    target_trust: float
    target_optimal_k_norm: float
    pass_at_1: bool
    pass_at_n: bool
    marginal_gain: float
    sample_agreement: float
    majority_correct: bool
    verifier_fp_event: bool
    optimal_k: int
    raw: Dict


def _pass_at_k(samples: Sequence[SampleRecord], k: int) -> bool:
    return any(s.is_correct for s in samples[:k])


def _majority_is_correct(samples: Sequence[SampleRecord], example: BenchmarkExample) -> bool:
    _, ext = majority_vote(samples)
    if ext is None:
        return False
    answer_type = example.metadata.get("answer_type", "math")
    ok, _ = grade_completion(f"\\boxed{{{ext}}}", example.gold_answer, answer_type=answer_type)
    return ok


def _sample_agreement(samples: Sequence[SampleRecord]) -> float:
    preds = [normalize_answer(s.extracted) for s in samples if s.extracted is not None]
    preds = [p for p in preds if p is not None]
    if not preds:
        return 0.0
    counts = Counter(preds)
    return counts.most_common(1)[0][1] / len(preds)


def compute_controller_labels(
    example: BenchmarkExample,
    samples: Sequence[SampleRecord],
    max_n: int,
) -> ControllerLabels:
    """
    Value target: normalized optimal k (Damani-style — how much compute is useful).
    Trust target: agreement among samples penalized when majority fails despite oracle success (FP proxy).
    """
    n = min(len(samples), max_n)
    samples = list(samples[:n])
    pass1 = _pass_at_k(samples, 1)
    passn = _pass_at_k(samples, n)

    optimal_k = n + 1
    for k in range(1, n + 1):
        if _pass_at_k(samples, k):
            optimal_k = k
            break
    if optimal_k > n:
        optimal_k = n

    # Value: fraction of compute ladder needed (0=easy/pass@1, 1=need all N)
    if pass1:
        target_optimal_k_norm = 0.0
    else:
        target_optimal_k_norm = (optimal_k - 1) / max(1, n - 1)

    marginal_gain = float(passn) - float(pass1)
    # Regression target in [0,1]: marginal gain + difficulty signal
    target_value = min(1.0, 0.5 * marginal_gain + 0.5 * target_optimal_k_norm)

    agreement = _sample_agreement(samples)
    majority_ok = _majority_is_correct(samples, example)
    # FP event: oracle could succeed (pass@n) but majority vote fails
    verifier_fp_event = bool(passn and not majority_ok)

    # Trust: high when samples agree AND majority is reliable; penalize FP events
    target_trust = agreement
    if verifier_fp_event:
        target_trust *= 0.25
    elif not majority_ok and not pass1:
        target_trust *= 0.5
    target_trust = float(max(0.0, min(1.0, target_trust)))

    return ControllerLabels(
        target_value=target_value,
        target_trust=target_trust,
        target_optimal_k_norm=target_optimal_k_norm,
        pass_at_1=pass1,
        pass_at_n=passn,
        marginal_gain=marginal_gain,
        sample_agreement=agreement,
        majority_correct=majority_ok,
        verifier_fp_event=verifier_fp_event,
        optimal_k=optimal_k,
        raw={
            "n_samples": n,
            "max_n": max_n,
        },
    )