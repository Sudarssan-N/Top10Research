# P4: Test-Time Compute vs. Confidence Calibration + Corrective Probe

**Problem statement summary:**  
Systematic study of how reasoning budgets affect calibration; train lightweight recalibration probe, focus on ECE + low-FPR metrics.

**Compute target:** 1×A100 (~80-150 GPU-hrs, $150-350)  
**Target venues:** ICLR / COLM / EMNLP  
**Priority:** Rank 3 hedge / analysis paper (low risk, fast)  
**Key references:** arXiv:2508.15050 (over-reasoning impairs calibration)

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] Core CalibrationProbe + ECE helper usage
- [x] Smoke (ECE raw vs calibrated demo) + train runnable on synthetic
- [ ] Systematic study across reasoning budgets (generate with different max tokens)
- [ ] Collect raw model confidence + hidden + true labels
- [ ] Train probe for recalibration + report ECE + low-FPR recall improvements
- [ ] Analysis: does more test-time compute hurt calibration? (plots vs budget)
- [ ] Results + tables for paper (ICLR/COLM style)

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p04-calibration-probe/
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
