"""Plotting helpers for Phase 1 baseline Pareto curves."""
from __future__ import annotations

from typing import Dict, List  # noqa: F401 used by plot_phase3_comparison

import matplotlib.pyplot as plt


def _collect_points(benchmark_data: Dict) -> List[Dict]:
    points = []
    bon = benchmark_data.get("best_of_n", {})
    for n_str, strat_map in bon.get("sweep", {}).items():
        n = int(n_str)
        for strategy, m in strat_map.items():
            points.append({
                "n": n,
                "strategy": strategy,
                "accuracy": m.get("accuracy", 0),
                "avg_samples": m.get("avg_samples", n),
                "avg_output_tokens": m.get("avg_output_tokens", 0),
            })
    dr = benchmark_data.get("difficulty_routed")
    if dr:
        points.append({
            "n": dr.get("avg_samples", 0),
            "strategy": "difficulty_routed_majority",
            "accuracy": dr.get("accuracy", 0),
            "avg_samples": dr.get("avg_samples", 0),
            "avg_output_tokens": dr.get("avg_output_tokens", 0),
        })
    return points


def plot_phase1_pareto(all_results: Dict, out_path: str):
    benchmarks = all_results.get("benchmarks", {})
    n_bench = len([b for b in benchmarks.values() if "error" not in b])
    if n_bench == 0:
        return

    fig, axes = plt.subplots(1, n_bench, figsize=(5 * n_bench, 4), squeeze=False)
    idx = 0
    for bench_name, data in benchmarks.items():
        if "error" in data:
            continue
        ax = axes[0, idx]
        points = _collect_points(data)
        for strategy in sorted({p["strategy"] for p in points}):
            sp = [p for p in points if p["strategy"] == strategy]
            sp = sorted(sp, key=lambda x: x["avg_samples"])
            ax.plot(
                [p["avg_samples"] for p in sp],
                [p["accuracy"] for p in sp],
                marker="o",
                label=strategy,
            )
        ax.set_xlabel("Avg samples / query")
        ax.set_ylabel("Accuracy")
        ax.set_title(bench_name)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        idx += 1

    model = all_results.get("model", "model")
    fig.suptitle(f"P1 Phase 1 baselines — {model}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase3_comparison(results: Dict, out_path: str):
    """Compare controller variants vs fixed-N reference."""
    benchmarks = results.get("benchmarks", {})
    valid = [(n, d) for n, d in benchmarks.items() if "controllers" in d]
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4), squeeze=False)
    for idx, (bench_name, data) in enumerate(valid):
        ax = axes[0, idx]
        points = []
        for n_str, m in data.get("fixed_n_reference", {}).items():
            points.append({
                "label": f"fixed_N={n_str}",
                "x": m.get("avg_samples", float(n_str)),
                "y": m.get("accuracy", 0),
            })
        for variant, m in data.get("controllers", {}).items():
            points.append({
                "label": f"ctrl_{variant}",
                "x": m.get("avg_controller_k", m.get("avg_samples", 0)),
                "y": m.get("accuracy", 0),
            })
        for p in points:
            ax.scatter(p["x"], p["y"], s=80)
            ax.annotate(p["label"], (p["x"], p["y"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Avg samples / query")
        ax.set_ylabel("Accuracy")
        ax.set_title(bench_name)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 3: Controller vs fixed-N", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)