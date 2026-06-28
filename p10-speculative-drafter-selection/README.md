# P10: Dynamic Heterogeneous Drafter Selection for Speculative Decoding

**Problem statement summary:**  
Learned or training-free input-adaptive selector among heterogeneous drafters (small models) with acceptance-rate / speedup guarantees.

**Compute target:** 1-2×A100 (~100 GPU-hrs, $250-500)  
**Target venues:** MLSys / ICLR / NeurIPS  
**Priority:** Medium (needs strong wall-clock baselines)  
**Key references:** arXiv:2604.05417, arXiv:2512.23765 (training-free variants)

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] DrafterSelector (router) + select + est_accept + smoke/train
- [ ] Context features from real drafter runs
- [ ] Learned vs training-free selection
- [ ] Wall-clock speedup + acceptance rate measurement
- [ ] MLSys/ICLR results vs single drafter baselines

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p10-speculative-drafter-selection/
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
