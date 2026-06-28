# P7: Automated Discovery of Novel Biases in LLM-as-Judge

**Problem statement summary:**  
Build contrastive perturbation + causal validation pipeline for automated bias discovery in judges; go beyond known position/verbosity biases.

**Compute target:** Mostly API + 1×A100 (~50 GPU-hrs + API cost, $100-300)  
**Target venues:** ACL / EMNLP / NeurIPS D&B  
**Priority:** High (fresh, low-compute)  
**Key references:** BiasScope (arXiv:2602.09383), position/verbosity bias papers

## Goals for this project

- Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
- Implement the core contribution described in the strategy doc.
- Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
- Produce publication-quality figures, tables, and analysis.

## Current Status (scaffold)

- [x] Folder + basic structure created
- [x] README + run.md + requirements skeleton
- [x] BiasDiscoveryProbe (pair) + bias strength demo + smoke/train
- [ ] Contrastive perturbation pipeline + causal validation
- [ ] Integration with LLM-as-judge (API or local)
- [ ] Discovery on CodeJudgeBench/MTBench + FPR quantification
- [ ] Novel bias findings + paper-ready analysis

See `run.md` for exact next steps and how to execute.

## Directory layout

```
p07-bias-discovery-llm-judge/
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
