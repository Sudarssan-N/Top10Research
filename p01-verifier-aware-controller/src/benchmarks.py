"""Benchmark loaders for P1 Phase 1 baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

BENCHMARK_SPECS: Dict[str, Dict] = {
    "gsm8k": {
        "hf_id": "gsm8k",
        "hf_config": "main",
        "split": "test",
        "fallback_split": "train[:1319]",
        "prompt_field": "question",
        "answer_field": "answer",
        "answer_type": "gsm8k",
    },
    "math500": {
        "hf_id": "HuggingFaceH4/MATH-500",
        "hf_config": None,
        "split": "test",
        "fallback_split": "train",
        "prompt_field": "problem",
        "answer_field": "answer",
        "answer_type": "math",
    },
    "aime24": {
        "hf_id": "Maxwell-Jia/AIME_2024",
        "hf_config": None,
        "split": "train",
        "fallback_split": "train",
        "prompt_field": "Problem",
        "answer_field": "Answer",
        "answer_type": "aime",
    },
}


@dataclass
class BenchmarkExample:
    id: str
    benchmark: str
    prompt: str
    gold_answer: str
    metadata: Dict


def _load_hf_dataset(spec: Dict, max_examples: Optional[int]):
    from datasets import load_dataset

    kwargs = {"path": spec["hf_id"]}
    if spec.get("hf_config"):
        kwargs["name"] = spec["hf_config"]

    try:
        ds = load_dataset(**kwargs, split=spec["split"])
    except Exception:
        ds = load_dataset(**kwargs, split=spec["fallback_split"])

    if max_examples is not None:
        n = min(max_examples, len(ds))
        ds = ds.select(range(n))
    return ds


def load_benchmark(
    name: str,
    max_examples: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> List[BenchmarkExample]:
    """Load a benchmark by short name (gsm8k, math500, aime24)."""
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "math500": "math500",
        "math-500": "math500",
        "gsm8k": "gsm8k",
        "aime24": "aime24",
        "aime2024": "aime24",
    }
    key = aliases.get(key, key)
    if key not in BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from: {list(BENCHMARK_SPECS)}")

    spec = BENCHMARK_SPECS[key]
    ds = _load_hf_dataset(spec, max_examples)

    examples: List[BenchmarkExample] = []
    for i, row in enumerate(ds):
        row = dict(row)
        prompt = row.get(spec["prompt_field"]) or row.get("question") or row.get("problem")
        gold = row.get(spec["answer_field"]) or row.get("solution") or row.get("answer")
        if prompt is None or gold is None:
            continue
        examples.append(
            BenchmarkExample(
                id=f"{key}-{i}",
                benchmark=key,
                prompt=str(prompt).strip(),
                gold_answer=str(gold).strip(),
                metadata={"answer_type": spec["answer_type"], "raw": row},
            )
        )
    return examples


def iter_benchmarks(names: List[str], max_examples: Optional[int] = None) -> Iterator[BenchmarkExample]:
    for name in names:
        for ex in load_benchmark(name, max_examples=max_examples):
            yield ex


def mock_benchmark_examples(n: int = 8) -> List[BenchmarkExample]:
    """Tiny in-memory benchmark for smoke tests (no HF download)."""
    examples = []
    for i in range(n):
        a, b = i + 2, i + 3
        examples.append(
            BenchmarkExample(
                id=f"mock-{i}",
                benchmark="mock",
                prompt=f"What is {a} + {b}? Show your work and put the final answer in \\boxed{{}}.",
                gold_answer=f"\\boxed{{{a + b}}}",
                metadata={"answer_type": "math"},
            )
        )
    return examples