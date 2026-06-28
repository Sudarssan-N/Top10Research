# P9: Token-Efficient Agent Memory with Bounded Reasoning Cost

**Problem statement summary:**  
Lightweight memory management policy (what to remember/forget) that bounds per-task token/compute cost while preserving success rate on agent benchmarks.

**Compute target:** 1×A100 + API (~80 GPU-hrs, $200-400)  
**Target venues:** NeurIPS / COLM / EMNLP  
**Priority:** Medium-high (agent cost is painful)  
**Key references:** Memori, A-Mem, LoCoMo, MultiAgentBench, 'Why Do Multi-Agent LLM Systems Fail?'

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] MemoryPolicyProbe + select_memory + smoke/train
- [ ] Agent traces on LoCoMo / agent benchmarks
- [ ] Memory policy training to bound tokens/task
- [ ] Success rate vs token cost curves
- [ ] Results for NeurIPS/COLM/EMNLP

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p09-agent-memory-efficient/
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
