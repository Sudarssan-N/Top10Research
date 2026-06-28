"""Phase 1 baseline selection strategies for test-time compute."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .grading import grade_completion, normalize_answer


@dataclass
class SampleRecord:
    completion: str
    is_correct: bool
    extracted: Optional[str]
    completion_tokens: int
    prompt_tokens: int


@dataclass
class BaselineDecision:
    strategy: str
    chosen_completion: str
    chosen_extracted: Optional[str]
    is_correct: bool
    num_samples: int
    total_completion_tokens: int
    total_tokens: int
    metadata: Dict = field(default_factory=dict)


def majority_vote(samples: Sequence[SampleRecord]) -> Tuple[str, Optional[str]]:
    preds = [normalize_answer(s.extracted) for s in samples if s.extracted is not None]
    preds = [p for p in preds if p is not None]
    if not preds:
        return samples[0].completion, samples[0].extracted
    counts = Counter(preds)
    winner, _ = counts.most_common(1)[0]
    for s in samples:
        if normalize_answer(s.extracted) == winner:
            return s.completion, s.extracted
    return samples[0].completion, samples[0].extracted


def select_single(samples: Sequence[SampleRecord]) -> BaselineDecision:
    s = samples[0]
    return BaselineDecision(
        strategy="single_n1",
        chosen_completion=s.completion,
        chosen_extracted=s.extracted,
        is_correct=s.is_correct,
        num_samples=1,
        total_completion_tokens=s.completion_tokens,
        total_tokens=s.prompt_tokens + s.completion_tokens,
    )


def select_best_of_n_majority(samples: Sequence[SampleRecord]) -> BaselineDecision:
    comp, ext = majority_vote(samples)
    # Correct if any sample with the winning extraction was graded correct.
    is_correct = any(
        s.is_correct and normalize_answer(s.extracted) == normalize_answer(ext) for s in samples
    )
    return BaselineDecision(
        strategy="best_of_n_majority",
        chosen_completion=comp,
        chosen_extracted=ext,
        is_correct=is_correct,
        num_samples=len(samples),
        total_completion_tokens=sum(s.completion_tokens for s in samples),
        total_tokens=samples[0].prompt_tokens * len(samples) + sum(s.completion_tokens for s in samples),
        metadata={"pass_at_n": any(s.is_correct for s in samples)},
    )


def select_pass_at_n(samples: Sequence[SampleRecord]) -> BaselineDecision:
    """Oracle selection ceiling: pick a correct sample if any exists."""
    correct = [s for s in samples if s.is_correct]
    chosen = correct[0] if correct else samples[0]
    return BaselineDecision(
        strategy="pass_at_n_oracle",
        chosen_completion=chosen.completion,
        chosen_extracted=chosen.extracted,
        is_correct=bool(correct),
        num_samples=len(samples),
        total_completion_tokens=sum(s.completion_tokens for s in samples),
        total_tokens=samples[0].prompt_tokens * len(samples) + sum(s.completion_tokens for s in samples),
        metadata={"had_correct_sample": bool(correct)},
    )


def select_random(samples: Sequence[SampleRecord]) -> BaselineDecision:
    import random
    s = random.choice(list(samples))
    return BaselineDecision(
        strategy="best_of_n_random",
        chosen_completion=s.completion,
        chosen_extracted=s.extracted,
        is_correct=s.is_correct,
        num_samples=len(samples),
        total_completion_tokens=sum(x.completion_tokens for x in samples),
        total_tokens=samples[0].prompt_tokens * len(samples) + sum(x.completion_tokens for x in samples),
    )


def select_longest(samples: Sequence[SampleRecord]) -> BaselineDecision:
    """Heuristic 'revision-heavy' baseline — prefer longest chain."""
    s = max(samples, key=lambda x: x.completion_tokens)
    return BaselineDecision(
        strategy="best_of_n_longest",
        chosen_completion=s.completion,
        chosen_extracted=s.extracted,
        is_correct=s.is_correct,
        num_samples=len(samples),
        total_completion_tokens=sum(x.completion_tokens for x in samples),
        total_tokens=samples[0].prompt_tokens * len(samples) + sum(x.completion_tokens for x in samples),
    )


def allocate_k_difficulty_only(prompt_token_len: int, tertiles: Tuple[int, int], max_k: int) -> int:
    """Damani-style simplified allocation: more tokens on harder (longer) prompts."""
    p33, p66 = tertiles
    if prompt_token_len <= p33:
        return 1
    if prompt_token_len <= p66:
        return max(2, max_k // 4)
    return max_k


def allocate_k_fixed(k: int) -> int:
    return k


PHASE1_STRATEGIES = {
    "single_n1": lambda samples, **_: select_single(samples[:1]),
    "best_of_n_majority": select_best_of_n_majority,
    "pass_at_n_oracle": select_pass_at_n,
    "best_of_n_random": select_random,
    "best_of_n_longest": select_longest,
}


def apply_strategy(name: str, samples: Sequence[SampleRecord]) -> BaselineDecision:
    fn = PHASE1_STRATEGIES[name]
    return fn(samples)


def aggregate_metrics(decisions: Sequence[BaselineDecision]) -> Dict:
    n = len(decisions)
    if n == 0:
        return {"accuracy": 0.0, "avg_samples": 0.0, "avg_output_tokens": 0.0, "avg_total_tokens": 0.0, "count": 0}
    return {
        "accuracy": sum(d.is_correct for d in decisions) / n,
        "avg_samples": sum(d.num_samples for d in decisions) / n,
        "avg_output_tokens": sum(d.total_completion_tokens for d in decisions) / n,
        "avg_total_tokens": sum(d.total_tokens for d in decisions) / n,
        "pass_at_n_rate": sum(d.metadata.get("pass_at_n", d.is_correct) for d in decisions) / n,
        "count": n,
    }