"""Plotting helpers for P2 ceiling and calibration analysis."""
from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt


def plot_ceiling_curves(summary: Dict, out_path: str, strategy: str = "argmax") -> None:
    """Accuracy vs. N for each verifier, with theoretical ceiling overlay."""
    benchmarks = summary.get("benchmarks", {})
    if not benchmarks:
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_bench = len([b for b in benchmarks.values() if "sweep" in b])
    if n_bench == 0:
        return

    fig, axes = plt.subplots(1, n_bench, figsize=(5 * n_bench, 4), squeeze=False)
    ax_idx = 0

    for bench_name, bench_data in benchmarks.items():
        if "sweep" not in bench_data:
            continue
        ax = axes[0, ax_idx]
        sweep = bench_data["sweep"]
        n_values = [int(n) for n in sweep.get("n_values", [])]

        for vname, vdata in sweep.get("verifiers", {}).items():
            empirical = vdata.get("empirical_curves", {}).get(strategy, {})
            theory = vdata.get("theoretical_ceiling", {})
            emp_acc = [empirical.get(str(n), {}).get("accuracy", 0.0) for n in n_values]
            theo_acc = [theory.get(str(n), 0.0) for n in n_values]
            ax.plot(n_values, emp_acc, marker="o", label=f"{vname} (emp)")
            if any(theo_acc):
                ax.plot(n_values, theo_acc, linestyle="--", alpha=0.5, label=f"{vname} (theory)")

        ax.set_xlabel("N (samples per query)")
        ax.set_ylabel("Selection accuracy")
        ax.set_title(f"P2 ceiling — {bench_name}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()