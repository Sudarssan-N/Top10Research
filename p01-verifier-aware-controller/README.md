# P1: Verifier-Aware Learned Controller for Test-Time Compute

**Problem statement summary:**  
Lightweight learned controller that allocates test-time compute per query while accounting for verifier imperfection.

**Compute target:** 1×A100-40GB (~150-300 GPU-hrs, $300-600)  
**Target venues:** NeurIPS / ICLR / COLM  
**Priority:** Highest (start here)  
**Key references:** Snell et al. (ICLR 2025), Damani et al. (ICLR 2025), Re-FORC, Agrawal et al. (imperfect verifier ceiling)

**Research foundation:** [`research.md`](research.md) → [`../common/research/p01-research.md`](../common/research/p01-research.md) (20-paper map, thesis chain, phases). Master index: [`../research.md`](../research.md) §P1.

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] Core implementation (VerifierAwareController + value/trust heads + simple decide_budget)
- [x] Smoke tests fully runnable (no large models)
- [x] **Phase 1 baselines** — best-of-N Pareto, majority/oracle/random/longest, difficulty routing, fixed budgets (`scripts/run_phase1_baselines.py`)
- [x] Benchmark loaders (GSM8K, MATH-500, AIME 2024) + math grading
- [x] **Phase 2 features** — frozen hiddens + value/trust labels + cached `.pt` + train (`scripts/run_phase2.py`)
- [ ] Proper verifier labels for controller trust head
- [x] **Phase 3** — train ablations (verifier-aware vs difficulty-only) + controller eval (`scripts/run_phase3.py`)
- [x] **Colab pipeline** — `colab/P1_Verifier_Aware_Controller.ipynb` + `scripts/colab_run_all.py`
- [ ] Beat Phase 1 Pareto at matched accuracy on full benchmarks
- [ ] Results + analysis on MATH-500 / AIME / GPQA + plots

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p01-verifier-aware-controller/
├── README.md
├── run.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── controller.py          # (or probe, router, etc.)
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── scripts/
│   └── run_experiment.py
├── configs/
│   └── default.yaml
├── data/          # (gitignored, populated at runtime)
├── experiments/
├── results/
└── ...
```

## Quick links

- Full strategy: see top-level `compass_artifact_...markdown.md` and top `README.md`
- Run instructions: `run.md`
