"""P2 Phase 1: ceiling analysis and verifier baseline reproduction."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .benchmarks import BenchmarkExample, load_benchmark, mock_benchmark_examples
from .ceiling import (
    CandidateRecord,
    accuracy_vs_n_curve,
    aggregate_verifier_metrics,
    compute_confusion_at_threshold,
    theoretical_ceiling_from_rates,
)
from .config import ExperimentConfig
from .generation import build_generator
from .grading import grade_completion
from .utils import set_seed
from .verifier import VerifierScorer, build_verifier


DEFAULT_N_VALUES = [1, 2, 4, 8, 16]
DEFAULT_VERIFIER_NAMES = ["noisy_optimistic_prm", "pessimistic_prm", "outcome_heuristic", "oracle"]


def _grade_generations(example: BenchmarkExample, generations) -> List[dict]:
    answer_type = example.metadata.get("answer_type", "math")
    rows = []
    for g in generations:
        ok, extracted = grade_completion(g.text, example.gold_answer, answer_type=answer_type)
        rows.append(
            {
                "completion": g.text,
                "is_correct": ok,
                "extracted": extracted,
                "completion_tokens": g.completion_tokens,
            }
        )
    return rows


def _build_candidate_pool(
    verifier: VerifierScorer,
    prompt: str,
    graded_rows: Sequence[dict],
) -> List[CandidateRecord]:
    completions = [r["completion"] for r in graded_rows]
    correctness = [r["is_correct"] for r in graded_rows]
    scores = verifier.score_candidates(prompt, completions, correctness)
    return [
        CandidateRecord(completion=r["completion"], is_correct=r["is_correct"], score=s)
        for r, s in zip(graded_rows, scores)
    ]


def run_verifier_ceiling_sweep(
    generator,
    examples: Sequence[BenchmarkExample],
    verifier_names: Sequence[str],
    n_values: Sequence[int],
    config: ExperimentConfig,
    target_fpr: float = 0.05,
    strategies: Sequence[str] = ("argmax", "conservative"),
) -> Dict:
    """Generate N samples per query, score with each verifier, measure ceiling curves."""
    max_n = max(n_values)
    per_example_graded: Dict[str, List[dict]] = {}

    for ex in tqdm(examples, desc="generate_for_ceiling"):
        gens = generator.generate(
            ex.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            n=max_n,
        )
        per_example_graded[ex.id] = _grade_generations(ex, gens)

    results = {"n_values": list(n_values), "verifiers": {}, "target_fpr": target_fpr}
    all_flat_scores: Dict[str, List[float]] = {v: [] for v in verifier_names}
    all_flat_labels: Dict[str, List[bool]] = {v: [] for v in verifier_names}

    for vname in verifier_names:
        verifier = build_verifier(vname, seed=config.seed)
        per_query_pools: Dict[str, List[CandidateRecord]] = {}
        for ex in examples:
            pool = _build_candidate_pool(verifier, ex.prompt, per_example_graded[ex.id])
            per_query_pools[ex.id] = pool
            for c in pool:
                all_flat_scores[vname].append(c.score)
                all_flat_labels[vname].append(c.is_correct)

        op = aggregate_verifier_metrics(all_flat_scores[vname], all_flat_labels[vname], target_fpr=target_fpr)
        threshold = op["precision_at_fpr"]["threshold"]
        conf = compute_confusion_at_threshold(all_flat_scores[vname], all_flat_labels[vname], threshold)

        p_correct = sum(any(r["is_correct"] for r in per_example_graded[ex.id][:max_n]) for ex in examples) / len(examples)
        theory = {
            str(n): theoretical_ceiling_from_rates(
                p_correct_sample=p_correct,
                verifier_fp_rate=conf["fpr"],
                verifier_fn_rate=conf["fnr"],
                n=n,
            )
            for n in n_values
        }

        strategy_curves = {}
        for strategy in strategies:
            strategy_curves[strategy] = accuracy_vs_n_curve(
                per_query_pools,
                n_values=n_values,
                threshold=threshold,
                strategy=strategy,
            )

        results["verifiers"][vname] = {
            "verifier": vname,
            "operating_point": op,
            "confusion_at_tau": conf,
            "theoretical_ceiling": theory,
            "empirical_curves": strategy_curves,
        }

    return results


def run_phase1_ceiling(
    config: ExperimentConfig,
    benchmarks: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    n_values: Optional[List[int]] = None,
    verifier_names: Optional[List[str]] = None,
    mock: bool = False,
    target_fpr: float = 0.05,
    output_dir: Optional[str] = None,
) -> Dict:
    set_seed(config.seed)
    benchmarks = benchmarks or config.benchmarks
    n_values = n_values or DEFAULT_N_VALUES
    verifier_names = verifier_names or DEFAULT_VERIFIER_NAMES
    output_dir = output_dir or os.path.join(config.output_dir, "phase1")
    os.makedirs(output_dir, exist_ok=True)

    generator = build_generator(config.base_model, device=config.get_device(), mock=mock)
    summary = {
        "phase": "p2_phase1_ceiling",
        "model": config.base_model,
        "mock": mock,
        "target_fpr": target_fpr,
        "benchmarks": {},
    }

    for bench_name in benchmarks:
        print(f"\n=== P2 Phase 1 ceiling: {bench_name} ===")
        if mock and bench_name == "mock":
            examples = mock_benchmark_examples(max_examples or 8)
        else:
            try:
                examples = load_benchmark(bench_name, max_examples=max_examples)
            except Exception as e:
                print(f"  Skipping {bench_name}: {e}")
                summary["benchmarks"][bench_name] = {"error": str(e)}
                continue

        print(f"  Loaded {len(examples)} examples")
        sweep = run_verifier_ceiling_sweep(
            generator,
            examples,
            verifier_names=verifier_names,
            n_values=n_values,
            config=config,
            target_fpr=target_fpr,
        )
        summary["benchmarks"][bench_name] = {
            "num_examples": len(examples),
            "sweep": sweep,
        }

        bench_dir = os.path.join(output_dir, bench_name)
        os.makedirs(bench_dir, exist_ok=True)
        with open(os.path.join(bench_dir, "ceiling_metrics.json"), "w") as f:
            json.dump(summary["benchmarks"][bench_name], f, indent=2)

    summary_path = os.path.join(output_dir, "p2_ceiling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nP2 Phase 1 summary written to {summary_path}")

    try:
        from .plots import plot_ceiling_curves
        plot_ceiling_curves(summary, os.path.join(output_dir, "p2_ceiling_curves.png"))
        print(f"Ceiling plot written to {os.path.join(output_dir, 'p2_ceiling_curves.png')}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    return summary


def run_phase1_smoke(config: ExperimentConfig, output_dir: Optional[str] = None) -> Dict:
    return run_phase1_ceiling(
        config=config,
        benchmarks=["mock"],
        max_examples=6,
        n_values=[1, 2, 4],
        verifier_names=["noisy_optimistic_prm", "pessimistic_prm", "oracle"],
        mock=True,
        target_fpr=0.05,
        output_dir=output_dir or os.path.join(config.output_dir, "phase1_smoke"),
    )