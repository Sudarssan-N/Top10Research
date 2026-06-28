# P5: Batch-Aware MoE Expert Routing at Inference (No Retraining)

**Problem statement summary:**  
Training-free or lightly adapted batch-aware router that mitigates union-of-experts activation and load imbalance for batched decode speedup.

**Compute target:** 2-8×A100 burst (~100 GPU-hrs, $400-900)  
**Target venues:** MLSys / NeurIPS / ICLR  
**Priority:** Medium (systems-heavy, good for MLSys)  
**Key references:** arXiv:2511.02237 (opportunistic MoE activation)

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] BatchAwareMoERouter (expert probs + suggest_batch_mask + est_speedup sim)
- [x] Smoke (mask + speedup demo) + train (MSE on expert dist) runnable
- [ ] Profiling real MoE (Qwen-MoE or similar) batch activation stats
- [ ] Training-free vs lightly trained router comparison
- [ ] Throughput/latency measurement harness (single node)
- [ ] Results vs vLLM/TensorRT baselines + accuracy vs sparsity
- [ ] MLSys-style numbers and ablations

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p05-batch-moe-routing/
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
