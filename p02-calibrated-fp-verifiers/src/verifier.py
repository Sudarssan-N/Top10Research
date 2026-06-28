"""Verifier scorers for P2 ceiling analysis and training labels."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ScoredCandidate:
    completion: str
    is_correct: bool
    score: float
    logit: float | None = None


class VerifierScorer(ABC):
    name: str = "base"

    @abstractmethod
    def score_candidates(self, prompt: str, completions: Sequence[str], correctness: Sequence[bool]) -> List[float]:
        """Return a score in [0, 1] per candidate (higher = verifier thinks correct)."""


class OracleScorer(VerifierScorer):
    """Upper bound: perfect scores on correct samples only."""

    name = "oracle"

    def score_candidates(self, prompt: str, completions: Sequence[str], correctness: Sequence[bool]) -> List[float]:
        return [1.0 if c else 0.0 for c in correctness]


class OutcomeHeuristicScorer(VerifierScorer):
    """Cobbe-style outcome proxy: length + boxed answer presence (mock without a trained RM)."""

    name = "outcome_heuristic"

    def score_candidates(self, prompt: str, completions: Sequence[str], correctness: Sequence[bool]) -> List[float]:
        scores = []
        for text in completions:
            has_box = "\\boxed" in text
            length_bonus = min(len(text.split()) / 80.0, 0.25)
            base = 0.45 + length_bonus + (0.15 if has_box else 0.0)
            scores.append(min(base, 0.95))
        return scores


class NoisyOptimisticScorer(VerifierScorer):
    """Simulates positivity-bias PRM: high scores on wrong answers with configurable FP rate."""

    name = "noisy_optimistic_prm"

    def __init__(self, seed: int = 42, fp_boost: float = 0.35):
        self.seed = seed
        self.fp_boost = fp_boost

    def score_candidates(self, prompt: str, completions: Sequence[str], correctness: Sequence[bool]) -> List[float]:
        rng = random.Random(self.seed + hash(prompt) % 10_000)
        scores = []
        for ok in correctness:
            if ok:
                scores.append(0.75 + rng.random() * 0.2)
            else:
                # False positive: verifier sometimes over-credits wrong answers
                if rng.random() < self.fp_boost:
                    scores.append(0.65 + rng.random() * 0.25)
                else:
                    scores.append(0.1 + rng.random() * 0.35)
        return scores


class PessimisticScorer(VerifierScorer):
    """Conservative scorer — lower scores, fewer FPs (Agrawal-style precision emphasis proxy)."""

    name = "pessimistic_prm"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def score_candidates(self, prompt: str, completions: Sequence[str], correctness: Sequence[bool]) -> List[float]:
        rng = random.Random(self.seed + hash(prompt) % 10_000)
        scores = []
        for ok in correctness:
            if ok:
                scores.append(0.55 + rng.random() * 0.35)
            else:
                scores.append(0.05 + rng.random() * 0.25)
        return scores


DEFAULT_VERIFIERS = {
    "oracle": OracleScorer,
    "outcome_heuristic": OutcomeHeuristicScorer,
    "noisy_optimistic_prm": NoisyOptimisticScorer,
    "pessimistic_prm": PessimisticScorer,
}


def build_verifier(name: str, seed: int = 42) -> VerifierScorer:
    key = name.lower().replace("-", "_")
    if key not in DEFAULT_VERIFIERS:
        raise ValueError(f"Unknown verifier '{name}'. Choose from: {list(DEFAULT_VERIFIERS)}")
    cls = DEFAULT_VERIFIERS[key]
    if key in ("noisy_optimistic_prm", "pessimistic_prm"):
        return cls(seed=seed)
    return cls()