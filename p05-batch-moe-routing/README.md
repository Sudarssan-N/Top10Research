# P5: Batch-Aware MoE Expert Routing at Inference (No Retraining)

**Problem statement summary:**  
Training-free or lightly adapted batch-aware router that mitigates union-of-experts activation and load imbalance for batched decode speedup.

**Compute target:** single A100 (offload regime) + vLLM baseline  
**Target venues:** MLSys / NeurIPS / ICLR (systems track)  
**Priority:** LOW — **pivot/deprioritize below P1/P3/P4** (see verdict)  
**Key references:** OEA arXiv:2511.02237, Lynx arXiv:2411.08982, SERE arXiv:2602.07616, XShare arXiv:2602.07265

> **Verdict (2026-06-29, see [`research.md`](research.md)): PIVOT / DEPRIORITIZE.**
> This is the most-scooped problem in the portfolio — the exact idea is OEA, already
> done training-free by Lynx, and the serving-integration gap is being closed now by
> SERE/XShare. Pursue **only** as the narrow "end-to-end vLLM throughput study OEA
> never did" (real wall-clock + peak memory in the single-GPU offload regime), not as
> a new routing idea. Hardest part is the measurement, not the method.

## Goals (if pursued, re-aimed)

- Profile union-of-experts growth vs. batch size on a real open MoE (Qwen-MoE / Mixtral).
- Apply a **training-free** batch policy (opportunistic / similarity / budget) over the
  model's own gating — no trained router.
- Report **end-to-end tokens/s + peak memory vs. vLLM** at matched downstream accuracy
  (the load-bearing metric; FLOP/expert-count proxies do not count).

## Current Status (scaffold — placeholder)

- [x] Folder + basic structure
- [⚠️] `core.py` `BatchAwareMoERouter` is a **placeholder** (trains a router; FLOP-proxy
  speedup) — see PIVOT NOTE in `src/core.py`; replace before any real run
- [ ] Hook a real MoE's gating logits + batch-union profiler
- [ ] Training-free policy + accuracy check
- [ ] vLLM wall-clock / peak-memory harness (adopt MoE-Inference-Bench, arXiv:2508.17467)

See `run.md` for the re-aimed plan.

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
