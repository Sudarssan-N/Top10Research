# P8: Step-Wise On-Policy Distillation for Small Reasoning Models (Tool Use)

**Problem statement summary:**  
Improve on naive distillation for ≤4B tool-using reasoners via step-wise, divergence-reweighted on-policy distillation.

**Compute target:** 1-2×A100 (~150 GPU-hrs, $300-600)  
**Target venues:** COLM / ICLR / ACL  
**Priority:** Medium (somewhat crowded but unsettled recipes)  
**Key references:** SOD (arXiv:2605.07725), LIMO/S1K lessons, Phi-4-Mini-Reasoning

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] DistillationReweightProbe + importance weights + smoke/train
- [ ] Step-wise on-policy data collection (tool use traces)
- [ ] Divergence-reweighted loss integration
- [ ] SLM eval (AIME/GPQA/code) vs naive distillation
- [ ] Results for COLM/ICLR/ACL

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p08-slm-distillation-tooluse/
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
