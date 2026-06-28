# P3: Latent Controller for Adaptive Overthinking Reduction

**Problem statement summary:**  
Per-query learned controller (probe on hidden states) that decides reasoning depth / early termination to combat ~18x token bloat on simple problems.

**Compute target:** 1×A100 (~100-200 GPU-hrs, $200-400)  
**Target venues:** ACL / EMNLP / COLM  
**Priority:** High (parallel track)  
**Key references:** LLMThinkBench, 'Wait, We Don't Need to Wait' (EMNLP 2025), ThinkPrune, L1, Re-FORC

**Research foundation:** [`research.md`](research.md) → [`../common/research/p03-research.md`](../common/research/p03-research.md) (20-paper map, thesis chain, phases). Master index: [`../research.md`](../research.md) §P3.

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] Core OverthinkController (stop_prob probe + should_stop / decide_early_stop_batch)
- [x] BCE training for convergence signal + runnable smoke/train with synthetic
- [x] Demo of early-stop decisions in smoke test
- [ ] Real hidden-state collection during long CoT traces
- [ ] Convergence labels (e.g. from final answer quality vs. length, or self-consistency)
- [ ] Integration with generation loop (stop token generation early)
- [ ] Comparison vs NoWait/ThinkPrune/L1 baselines + token/accuracy curves
- [ ] Full results + analysis on math/code tasks

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p03-overthinking-latent-controller/
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
