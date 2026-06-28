# P6: Context-Length-Induced Degradation Under Perfect Retrieval

**Problem statement summary:**  
Diagnose attention dilution / other mechanisms for performance drop as context length grows even with perfect retrieval; propose simple mitigations (reorder/calibrate).

**Compute target:** 1×A100 (~80 GPU-hrs, $150-300)  
**Target venues:** ACL / EMNLP / ICLR  
**Priority:** Medium-high (clean analysis)  
**Key references:** arXiv:2510.05381

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] ContextDegradationProbe + reorder suggestion + smoke/train runnable
- [ ] Long-context data injection experiments (perfect retrieval)
- [ ] Attention dilution diagnostics + mitigation (reorder/calibration)
- [ ] Full eval on long-context benchmarks
- [ ] Results + analysis for ACL/EMNLP/ICLR

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p06-long-context-degradation/
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
