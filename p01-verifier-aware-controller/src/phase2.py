"""P1 Phase 2: frozen hidden extraction + controller training labels."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .benchmarks import BenchmarkExample, load_benchmark, mock_benchmark_examples
from .config import ExperimentConfig
from .features import build_backbone, feature_cache_path, model_slug
from .generation import build_generator
from .grading import grade_completion
from .labels import compute_controller_labels
from .utils import set_seed


def _grade_generations(example: BenchmarkExample, generations) -> list:
    from .baselines import SampleRecord

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


def build_feature_records_for_examples(
    examples: Sequence[BenchmarkExample],
    backbone,
    generator,
    config: ExperimentConfig,
    label_n: int,
    pooling: str = "last_token",
    encode_batch_size: int = 4,
) -> List[Dict]:
    """Generate N samples per example, extract hiddens, compute labels."""
    prompts = [ex.prompt for ex in examples]
    hiddens = backbone.encode(prompts, pooling=pooling, batch_size=encode_batch_size)

    records = []
    for i, ex in enumerate(tqdm(examples, desc="label+generate")):
        gens = generator.generate(
            ex.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            n=label_n,
        )
        samples = _grade_generations(ex, gens)
        labels = compute_controller_labels(ex, samples, max_n=label_n)

        records.append(
            {
                "id": ex.id,
                "benchmark": ex.benchmark,
                "hidden": hiddens[i].clone(),
                "target_value": labels.target_value,
                "target_trust": labels.target_trust,
                "target_optimal_k_norm": labels.target_optimal_k_norm,
                "pass_at_1": labels.pass_at_1,
                "pass_at_n": labels.pass_at_n,
                "marginal_gain": labels.marginal_gain,
                "sample_agreement": labels.sample_agreement,
                "majority_correct": labels.majority_correct,
                "verifier_fp_event": labels.verifier_fp_event,
                "optimal_k": labels.optimal_k,
                "labels_raw": labels.raw,
            }
        )
    return records


def save_feature_cache(path: str, meta: Dict, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch_payload = {
        "meta": meta,
        "records": records,
    }
    import torch

    torch.save(torch_payload, path)


def extract_benchmark_features(
    benchmark: str,
    config: ExperimentConfig,
    max_examples: Optional[int] = None,
    label_n: Optional[int] = None,
    pooling: str = "last_token",
    mock: bool = False,
    data_dir: str = "data",
    force: bool = False,
) -> Dict:
    label_n = label_n or config.n_samples_best_of_n
    cache_path = feature_cache_path(data_dir, config.base_model, benchmark)

    if os.path.exists(cache_path) and not force:
        print(f"  Cache hit: {cache_path}")
        import torch

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return {"path": cache_path, "meta": payload["meta"], "num_records": len(payload["records"]), "cached": True}

    if mock and benchmark == "mock":
        examples = mock_benchmark_examples(max_examples or 12)
    else:
        examples = load_benchmark(benchmark, max_examples=max_examples)

    backbone = build_backbone(config.base_model, device=config.get_device(), mock=mock)
    generator = build_generator(config.base_model, device=config.get_device(), mock=mock)

    hidden_size = getattr(backbone, "hidden_size", 2048)
    records = build_feature_records_for_examples(
        examples,
        backbone,
        generator,
        config,
        label_n=label_n,
        pooling=pooling,
    )

    meta = {
        "model": config.base_model,
        "benchmark": benchmark,
        "pooling": pooling,
        "hidden_size": hidden_size,
        "label_n": label_n,
        "num_records": len(records),
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "mock": mock,
    }
    save_feature_cache(cache_path, meta, records)
    print(f"  Saved {len(records)} records -> {cache_path}")
    return {"path": cache_path, "meta": meta, "num_records": len(records), "cached": False}


def run_phase2_extract(
    config: ExperimentConfig,
    benchmarks: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    label_n: Optional[int] = None,
    pooling: str = "last_token",
    mock: bool = False,
    data_dir: str = "data",
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Dict:
    set_seed(config.seed)
    benchmarks = benchmarks or config.benchmarks
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "phase": "phase2_extract",
        "model": config.base_model,
        "mock": mock,
        "data_dir": data_dir,
        "benchmarks": {},
    }

    for bench in benchmarks:
        print(f"\n=== Phase 2 extract: {bench} ===")
        try:
            info = extract_benchmark_features(
                bench,
                config,
                max_examples=max_examples,
                label_n=label_n,
                pooling=pooling,
                mock=mock,
                data_dir=data_dir,
                force=force,
            )
            results["benchmarks"][bench] = info
        except Exception as e:
            print(f"  Failed: {e}")
            results["benchmarks"][bench] = {"error": str(e)}

    summary_path = os.path.join(output_dir, "phase2_extract_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nPhase 2 extract summary: {summary_path}")
    return results


def run_phase2_smoke(config: ExperimentConfig, output_dir: Optional[str] = None) -> Dict:
    return run_phase2_extract(
        config=config,
        benchmarks=["mock"],
        max_examples=10,
        label_n=4,
        mock=True,
        data_dir=os.path.join(config.output_dir, "phase2_smoke_data"),
        output_dir=output_dir or os.path.join(config.output_dir, "phase2_smoke"),
        force=True,
    )


def run_phase2_train(
    config: ExperimentConfig,
    feature_paths: Optional[List[str]] = None,
    data_dir: str = "data",
    benchmarks: Optional[List[str]] = None,
    mock: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """Train controller on cached Phase 2 features."""
    from .core import VerifierAwareController
    from .dataset import build_dataloaders, merge_feature_caches
    from .train import train, validate

    set_seed(config.seed)
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if feature_paths is None:
        benchmarks = benchmarks or config.benchmarks
        if mock:
            benchmarks = ["mock"]
            data_dir = os.path.join(config.output_dir, "phase2_smoke_data")
        feature_paths = [
            feature_cache_path(data_dir, config.base_model, b) for b in benchmarks
        ]

    meta, records = merge_feature_caches(feature_paths)
    if not records:
        raise RuntimeError("No feature records found. Run phase2 extract first.")

    hidden_size = int(meta.get("hidden_size", 2048))
    train_loader, val_loader = build_dataloaders(
        records,
        batch_size=config.train_batch_size,
        seed=config.seed,
    )

    model = VerifierAwareController(
        hidden_size=hidden_size,
        controller_dim=config.controller_hidden_dim,
        num_layers=config.num_controller_layers,
    )
    device = config.get_device()
    model = train(config, train_loader, model, device)

    val_metrics = validate(model, val_loader, device, config)
    ckpt_path = os.path.join(output_dir, "checkpoints", "best_controller.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "hidden_size": hidden_size,
            "config": {
                "controller_hidden_dim": config.controller_hidden_dim,
                "num_controller_layers": config.num_controller_layers,
            },
        },
        ckpt_path,
    )

    result = {
        "phase": "phase2_train",
        "checkpoint": ckpt_path,
        "hidden_size": hidden_size,
        "num_train_records": len(records),
        "val_metrics": val_metrics,
        "feature_meta": meta,
    }
    with open(os.path.join(output_dir, "phase2_train_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Phase 2 train complete. checkpoint={ckpt_path} val={val_metrics}")
    return result