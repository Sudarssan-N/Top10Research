"""P1 Phase 3: full controller training (ablations) + controller-guided eval."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from .baselines import SampleRecord, aggregate_metrics, apply_strategy
from .benchmarks import BenchmarkExample, load_benchmark, mock_benchmark_examples
from .checkpoint import load_controller_checkpoint, save_controller_checkpoint
from .config import ExperimentConfig
from .core import build_controller
from .dataset import build_dataloaders, merge_feature_caches
from .features import build_backbone, feature_cache_path
from .generation import build_generator
from .grading import grade_completion
from .train import train_with_validation
from .utils import set_seed


VARIANTS = ["verifier_aware", "difficulty_only"]


def _feature_paths(config: ExperimentConfig, benchmarks: List[str], data_dir: str, mock: bool) -> List[str]:
    if mock:
        data_dir = os.path.join(config.output_dir, "phase2_smoke_data")
        benchmarks = ["mock"]
    return [feature_cache_path(data_dir, config.base_model, b) for b in benchmarks]


def run_phase3_train(
    config: ExperimentConfig,
    benchmarks: Optional[List[str]] = None,
    data_dir: str = "data",
    mock: bool = False,
    output_dir: Optional[str] = None,
    variants: Optional[List[str]] = None,
) -> Dict:
    """Train verifier-aware + difficulty-only ablation with validation."""
    set_seed(config.seed)
    output_dir = output_dir or os.path.join(config.output_dir, "phase3")
    os.makedirs(output_dir, exist_ok=True)

    benchmarks = benchmarks or config.benchmarks
    paths = _feature_paths(config, benchmarks, data_dir, mock)
    meta, records = merge_feature_caches(paths)
    if not records:
        raise RuntimeError("No feature cache. Run Phase 2 extract first.")

    hidden_size = int(meta.get("hidden_size", 2048))
    meta["hidden_size"] = hidden_size
    train_loader, val_loader = build_dataloaders(
        records,
        batch_size=config.train_batch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    device = config.get_device()
    results = {"phase": "phase3_train", "variants": {}, "meta": meta}

    for variant in variants or VARIANTS:
        print(f"\n=== Phase 3 train: {variant} ===")
        model = build_controller(
            variant=variant,
            hidden_size=hidden_size,
            controller_dim=config.controller_hidden_dim,
            num_layers=config.num_controller_layers,
        )
        model, info = train_with_validation(
            config,
            train_loader,
            val_loader,
            model,
            device,
            variant=variant,
            run_dir=os.path.join(output_dir, variant),
            feature_meta=meta,
        )
        ckpt = os.path.join(output_dir, variant, "checkpoints", variant, "best_controller.pt")
        results["variants"][variant] = {
            "checkpoint": ckpt,
            "best_val_loss": info["best_val_loss"],
            "history": info["history"],
        }

    summary_path = os.path.join(output_dir, "phase3_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Phase 3 train summary: {summary_path}")
    return results


def _controller_decide_k(model, hidden: torch.Tensor, max_n: int, device: str) -> int:
    model.eval()
    with torch.inference_mode():
        h = hidden.unsqueeze(0).to(device) if hidden.dim() == 1 else hidden.to(device)
        out = model(h)
        k = model.decide_num_samples(
            out["value"].float(),
            out["verifier_trust"].float(),
            min_n=1,
            max_n=max_n,
        )
        return int(k.view(-1)[0].item())


def evaluate_controller_on_examples(
    model,
    backbone,
    generator,
    examples: Sequence[BenchmarkExample],
    config: ExperimentConfig,
    max_n: int,
    strategy_name: str,
) -> Dict:
    """Controller picks k per query, generate k, majority vote."""
    device = config.get_device()
    decisions = []

    for ex in tqdm(examples, desc=f"eval_{strategy_name}"):
        hidden = backbone.encode([ex.prompt], pooling=config.feature_pooling)[0]
        k = _controller_decide_k(model, hidden, max_n=max_n, device=device)
        gens = generator.generate(
            ex.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            n=k,
        )
        samples: List[SampleRecord] = []
        answer_type = ex.metadata.get("answer_type", "math")
        for g in gens:
            ok, extracted = grade_completion(g.text, ex.gold_answer, answer_type=answer_type)
            samples.append(
                SampleRecord(
                    completion=g.text,
                    is_correct=ok,
                    extracted=extracted,
                    completion_tokens=g.completion_tokens,
                    prompt_tokens=g.prompt_tokens,
                )
            )
        d = apply_strategy("best_of_n_majority", samples)
        from .phase1 import _finalize_decision

        d = _finalize_decision(d, ex)
        d.strategy = strategy_name
        d.metadata["controller_k"] = k
        with torch.inference_mode():
            pred = model(hidden.unsqueeze(0).to(device))
            d.metadata["value_pred"] = float(pred["value"].view(-1)[0].item())
            d.metadata["trust_pred"] = float(pred["verifier_trust"].view(-1)[0].item())
        decisions.append(d)

    metrics = aggregate_metrics(decisions)
    metrics["strategy"] = strategy_name
    metrics["avg_controller_k"] = sum(d.metadata.get("controller_k", 1) for d in decisions) / max(len(decisions), 1)
    return {"metrics": metrics, "decisions": decisions}


def run_phase3_eval(
    config: ExperimentConfig,
    checkpoint_dir: Optional[str] = None,
    benchmarks: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    mock: bool = False,
    output_dir: Optional[str] = None,
    variants: Optional[List[str]] = None,
) -> Dict:
    """Evaluate trained controllers vs fixed-N reference on same examples."""
    set_seed(config.seed)
    output_dir = output_dir or os.path.join(config.output_dir, "phase3")
    checkpoint_dir = checkpoint_dir or output_dir
    benchmarks = benchmarks or (["mock"] if mock else config.benchmarks)
    max_n = config.n_samples_best_of_n
    device = config.get_device()

    backbone = build_backbone(config.base_model, device=device, mock=mock)
    generator = build_generator(config.base_model, device=device, mock=mock)

    results = {"phase": "phase3_eval", "benchmarks": {}}

    for bench in benchmarks:
        print(f"\n=== Phase 3 eval: {bench} ===")
        if mock and bench == "mock":
            examples = mock_benchmark_examples(max_examples or 10)
        else:
            examples = load_benchmark(bench, max_examples=max_examples)

        bench_out = {"num_examples": len(examples), "controllers": {}, "fixed_n_reference": {}}

        # Fixed-N reference (majority) on same examples for Pareto comparison
        for n in [1, max_n // 2, max_n]:
            n = max(1, int(n))
            gens_by_ex = []
            for ex in examples:
                gens = generator.generate(ex.prompt, max_new_tokens=config.max_new_tokens, temperature=config.temperature, n=n)
                samples = []
                answer_type = ex.metadata.get("answer_type", "math")
                for g in gens:
                    ok, ext = grade_completion(g.text, ex.gold_answer, answer_type=answer_type)
                    samples.append(SampleRecord(g.text, ok, ext, g.completion_tokens, g.prompt_tokens))
                d = apply_strategy("best_of_n_majority", samples)
                from .phase1 import _finalize_decision
                gens_by_ex.append(_finalize_decision(d, ex))
            bench_out["fixed_n_reference"][str(n)] = aggregate_metrics(gens_by_ex)

        for variant in variants or VARIANTS:
            ckpt_path = os.path.join(checkpoint_dir, variant, "checkpoints", variant, "best_controller.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Missing checkpoint: {ckpt_path}")
                continue
            model, payload = load_controller_checkpoint(ckpt_path, device=device)
            eval_name = f"controller_{variant}"
            out = evaluate_controller_on_examples(
                model, backbone, generator, examples, config, max_n=max_n, strategy_name=eval_name
            )
            bench_out["controllers"][variant] = out["metrics"]

        results["benchmarks"][bench] = bench_out
        os.makedirs(os.path.join(output_dir, bench), exist_ok=True)
        with open(os.path.join(output_dir, bench, "phase3_eval.json"), "w") as f:
            json.dump(bench_out, f, indent=2, default=str)

    summary_path = os.path.join(output_dir, "phase3_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    try:
        from .plots import plot_phase3_comparison
        plot_phase3_comparison(results, os.path.join(output_dir, "phase3_comparison.png"))
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Phase 3 eval summary: {summary_path}")
    return results


def run_phase3_smoke(config: ExperimentConfig, output_dir: Optional[str] = None) -> Dict:
    """End-to-end Phase 3 smoke: requires Phase 2 smoke features."""
    from .phase2 import run_phase2_smoke

    output_dir = output_dir or os.path.join(config.output_dir, "phase3_smoke")
    config.num_epochs = max(2, config.num_epochs)

    run_phase2_smoke(config)
    train_out = run_phase3_train(
        config,
        mock=True,
        output_dir=output_dir,
        variants=VARIANTS,
    )
    eval_out = run_phase3_eval(
        config,
        checkpoint_dir=output_dir,
        mock=True,
        output_dir=output_dir,
        max_examples=8,
    )
    return {"train": train_out, "eval": eval_out}


def run_phase3_full(
    config: ExperimentConfig,
    max_examples: Optional[int] = None,
    mock: bool = False,
    data_dir: str = "data",
    output_dir: Optional[str] = None,
    skip_train: bool = False,
) -> Dict:
    output_dir = output_dir or os.path.join(config.output_dir, "phase3")
    out = {}
    if not skip_train:
        out["train"] = run_phase3_train(config, data_dir=data_dir, mock=mock, output_dir=output_dir)
    out["eval"] = run_phase3_eval(
        config,
        checkpoint_dir=output_dir,
        max_examples=max_examples,
        mock=mock,
        output_dir=output_dir,
    )
    return out