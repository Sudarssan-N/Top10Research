# P2: Calibrated, False-Positive-Bounded Verifiers for Best-of-N

**Problem statement summary:**  
Train/calibrate verifiers explicitly to minimize false positives at chosen operating points and raise the best-of-N ceiling.

**Compute target:** 1-4×A100 (~200 GPU-hrs, $400-800)  
**Target venues:** NeurIPS / ICLR  
**Priority:** Rank 2 (after P1 progress)  
**Key references:** Agrawal et al. (Cut the Overcredit), Stroebl et al. (arXiv:2411.17501), PRM/GenRM literature

**Research foundation:** [`research.md`](research.md) → [`../common/research/p02-research.md`](../common/research/p02-research.md) (20-paper map, thesis chain, phases). Master index: [`../research.md`](../research.md) §P2.

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] Core FPRBoundedVerifier (probe on hidden + score head)
- [x] Asymmetric / focal / standard BCE loss + threshold finder for target FPR
- [x] Calibration utilities (ECE, temperature scaling, precision@τ)
- [x] **Phase 1 ceiling scaffold** — verifier baselines, accuracy-vs-N curves, theoretical ceiling
- [x] Smoke test + train/eval runnable (synthetic data + FPR/precision metrics demo)
- [ ] Real hidden extraction + verifier labels from generations (Phase 2)
- [ ] Train FPRBoundedVerifier on real features vs. standard PRM ablation (Phase 3)
- [ ] Ceiling-shift evaluation at fixed FPR budget + P1 trust label export (Phase 5)

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p02-calibrated-fp-verifiers/
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
