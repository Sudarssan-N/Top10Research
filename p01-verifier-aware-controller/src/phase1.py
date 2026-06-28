"""P1 Phase 1: reproduce test-time compute baselines (no controller)."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .baselines import (
    BaselineDecision,
    SampleRecord,
    aggregate_metrics,
    allocate_k_difficulty_only,
    apply_strategy,
)
from .benchmarks import BenchmarkExample, load_benchmark, mock_benchmark_examples
from .config import ExperimentConfig
from .generation import build_generator
from .grading import grade_completion
from .utils import set_seed


DEFAULT_N_VALUES = [1, 2, 4, 8, 16]
DEFAULT_STRATEGIES = [
    "single_n1",
    "best_of_n_majority",
    "pass_at_n_oracle",
    "best_of_n_random",
    "best_of_n_longest",
]


def _finalize_decision(decision: BaselineDecision, example: BenchmarkExample) -> BaselineDecision:
    """Re-grade aggregated choice against gold (fixes majority-vote correctness)."""
    answer_type = example.metadata.get("answer_type", "math")
    if decision.chosen_extracted:
        boxed = f"\\boxed{{{decision.chosen_extracted}}}"
        ok, ext = grade_completion(boxed, example.gold_answer, answer_type=answer_type)
        decision.is_correct = ok
        decision.chosen_extracted = ext
    return decision


def _grade_samples(example: BenchmarkExample, generations) -> List[SampleRecord]:
    answer_type = example.metadata.get("answer_type", "math")
    records = []
    for g in generations:
        ok, extracted = grade_completion(g.text, example.gold_answer, answer_type=answer_type)
        records.append(
            SampleRecord(
                completion=g.text,
                is_correct=ok,
                extracted=extracted,
                completion_tokens=g.completion_tokens,
                prompt_tokens=g.prompt_tokens,
            )
        )
    return records


def _prompt_length(generator, example: BenchmarkExample) -> int:
    if hasattr(generator, "prompt_token_count"):
        return generator.prompt_token_count(example.prompt)
    return len(example.prompt.split())


def run_difficulty_routed_baseline(
    generator,
    examples: Sequence[BenchmarkExample],
    max_k: int,
    config: ExperimentConfig,
    tertiles: Optional[tuple] = None,
) -> Dict:
    """Difficulty-only controller: allocate k by prompt length tertiles."""
    if tertiles is None:
        lengths = [_prompt_length(generator, ex) for ex in examples]
        sorted_lens = sorted(lengths)
        n = len(sorted_lens)
        p33 = sorted_lens[max(0, n // 3 - 1)]
        p66 = sorted_lens[max(0, (2 * n) // 3 - 1)]
        tertiles = (p33, p66)

    decisions: List[BaselineDecision] = []
    for ex in tqdm(examples, desc="difficulty_routed"):
        plen = _prompt_length(generator, ex)
        k = allocate_k_difficulty_only(plen, tertiles, max_k)
        gens = generator.generate(
            ex.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            n=k,
        )
        samples = _grade_samples(ex, gens)
        d = apply_strategy("best_of_n_majority", samples)
        decisions.append(_finalize_decision(d, ex))

    metrics = aggregate_metrics(decisions)
    metrics["tertiles"] = list(tertiles)
    metrics["max_k"] = max_k
    metrics["strategy"] = "difficulty_routed_majority"
    return {"metrics": metrics, "decisions": decisions}


def run_fixed_budget_sweep(
    generator,
    examples: Sequence[BenchmarkExample],
    token_budgets: Sequence[int],
    config: ExperimentConfig,
) -> Dict:
    """Fixed max_new_tokens per query (single sample)."""
    results = {}
    for budget in token_budgets:
        decisions: List[BaselineDecision] = []
        for ex in tqdm(examples, desc=f"fixed_budget_{budget}"):
            gens = generator.generate(
                ex.prompt,
                max_new_tokens=budget,
                temperature=config.temperature,
                n=1,
            )
            samples = _grade_samples(ex, gens)
            decisions.append(apply_strategy("single_n1", samples))
        results[str(budget)] = aggregate_metrics(decisions)
    return results


def run_best_of_n_sweep(
    generator,
    examples: Sequence[BenchmarkExample],
    n_values: Sequence[int],
    strategies: Sequence[str],
    config: ExperimentConfig,
) -> Dict:
    """Core Phase 1: generate up to max(n) samples once, evaluate each N and strategy."""
    max_n = max(n_values)
    per_example_samples: Dict[str, List[SampleRecord]] = {}

    for ex in tqdm(examples, desc="generate_samples"):
        gens = generator.generate(
            ex.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            n=max_n,
        )
        per_example_samples[ex.id] = _grade_samples(ex, gens)

    sweep = {}
    for n in n_values:
        sweep[str(n)] = {}
        for strategy in strategies:
            decisions: List[BaselineDecision] = []
            for ex in examples:
                samples = per_example_samples[ex.id][:n]
                if strategy == "single_n1":
                    d = apply_strategy("single_n1", samples[:1])
                else:
                    d = apply_strategy(strategy, samples)
                if strategy == "best_of_n_majority":
                    d = _finalize_decision(d, ex)
                decisions.append(d)
            sweep[str(n)][strategy] = aggregate_metrics(decisions)
    return {"n_values": list(n_values), "strategies": list(strategies), "sweep": sweep}


def run_phase1_baselines(
    config: ExperimentConfig,
    benchmarks: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    n_values: Optional[List[int]] = None,
    strategies: Optional[List[str]] = None,
    mock: bool = False,
    token_budgets: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    set_seed(config.seed)
    benchmarks = benchmarks or config.benchmarks
    n_values = n_values or DEFAULT_N_VALUES
    strategies = strategies or DEFAULT_STRATEGIES
    token_budgets = token_budgets or [128, 256, 512, 1024]
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    generator = build_generator(config.base_model, device=config.get_device(), mock=mock)
    all_results = {
        "phase": "phase1_baselines",
        "model": config.base_model,
        "mock": mock,
        "benchmarks": {},
    }

    for bench_name in benchmarks:
        print(f"\n=== Benchmark: {bench_name} ===")
        if mock and bench_name == "mock":
            examples = mock_benchmark_examples(max_examples or 8)
        else:
            try:
                examples = load_benchmark(bench_name, max_examples=max_examples)
            except Exception as e:
                print(f"  Skipping {bench_name}: {e}")
                all_results["benchmarks"][bench_name] = {"error": str(e)}
                continue

        print(f"  Loaded {len(examples)} examples")

        bon = run_best_of_n_sweep(generator, examples, n_values, strategies, config)
        diff = run_difficulty_routed_baseline(generator, examples, max_k=max(n_values), config=config)
        fixed = run_fixed_budget_sweep(generator, examples, token_budgets, config)

        bench_result = {
            "num_examples": len(examples),
            "best_of_n": bon,
            "difficulty_routed": diff["metrics"],
            "fixed_token_budget": fixed,
        }
        all_results["benchmarks"][bench_name] = bench_result

        _save_benchmark_artifacts(output_dir, bench_name, bench_result, bon, diff, examples)

    summary_path = os.path.join(output_dir, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPhase 1 summary written to {summary_path}")

    try:
        from .plots import plot_phase1_pareto
        plot_phase1_pareto(all_results, os.path.join(output_dir, "phase1_pareto.png"))
        print(f"Pareto plot written to {os.path.join(output_dir, 'phase1_pareto.png')}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    return all_results


def _save_benchmark_artifacts(output_dir, bench_name, bench_result, bon, diff, examples):
    bench_dir = os.path.join(output_dir, bench_name)
    os.makedirs(bench_dir, exist_ok=True)
    with open(os.path.join(bench_dir, "metrics.json"), "w") as f:
        json.dump(bench_result, f, indent=2)

    # Flat Pareto rows for easy analysis
    rows = []
    for n, strat_map in bon["sweep"].items():
        for strategy, metrics in strat_map.items():
            rows.append({
                "benchmark": bench_name,
                "n": int(n),
                "strategy": strategy,
                **metrics,
            })
    rows.append({"benchmark": bench_name, "n": None, "strategy": "difficulty_routed_majority", **diff["metrics"]})
    with open(os.path.join(bench_dir, "pareto_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)


def run_phase1_smoke(config: ExperimentConfig, output_dir: Optional[str] = None) -> Dict:
    """Fast local smoke: mock generator + mock benchmark, 6 examples, N in {1,2,4}."""
    return run_phase1_baselines(
        config=config,
        benchmarks=["mock"],
        max_examples=6,
        n_values=[1, 2, 4],
        strategies=["single_n1", "best_of_n_majority", "pass_at_n_oracle"],
        mock=True,
        token_budgets=[64, 128],
        output_dir=output_dir or os.path.join(config.output_dir, "phase1_smoke"),
    )