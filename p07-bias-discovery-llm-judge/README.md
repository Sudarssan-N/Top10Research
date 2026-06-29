# P7: Automated Discovery of Novel Biases in LLM-as-Judge

**Problem statement summary:**  
Build contrastive perturbation + causal validation pipeline for automated bias discovery in judges; go beyond known position/verbosity biases.

**Compute target:** Mostly API + light GPU ($100-300)  
**Target venues:** ACL / EMNLP / NeurIPS D&B  
**Priority:** Medium — **pivot** (framing scooped; residual is causal + code-judge)  
**Key references:** BiasScope arXiv:2602.09383 (ICLR 2026, REAL), Automated Concept Discovery arXiv:2603.03319, CodeJudgeBench arXiv:2507.10535

> **Verdict (2026-06-29, see [`research.md`](research.md)): PIVOT.**
> Substantially scooped — **BiasScope (arXiv:2602.09383) is real (ICLR 2026)** and already
> does automated discovery + correctness-preserving counterfactual perturbations (48
> validated biases). Defensible residual: **causal** validation (flip-rate ATE + bootstrap
> CIs + mediation) in the **executably-verifiable code-judge domain** (CodeJudgeBench),
> where unit-test pass/fail makes "answer quality held fixed" ground truth rather than an
> approximation. Fast-moving area (Bias-in-the-Loop, arXiv:2604.16790).

## Goals (re-aimed)

- Generate **quality-preserving** perturbations of *code* answers (refactor, rename,
  reformat) where unit tests still pass → quality is provably constant.
- Query the judge on perturbed vs. original; estimate a **flip-rate ATE with bootstrap CIs**.
- **Mediation** analysis to rule out confounds (is the flip caused by the factor, or a
  correlated quality change the tests didn't catch?).
- Differentiate from BiasScope via causal rigor + the code domain, not by re-discovering biases.

## Current Status (scaffold — placeholder)

- [x] Folder + basic structure
- [⚠️] `BiasDiscoveryProbe` (MLP on pair embeddings) is a **placeholder** — see PIVOT NOTE
  in `src/core.py`; the real method is prompt/LLM-driven perturbation + API judge + causal
  estimation, not a trained probe
- [ ] Quality-preserving code-perturbation generator (unit-test gated)
- [ ] Judge-querying harness + flip-rate ATE estimator with bootstrap CIs
- [ ] Mediation / confound checks

See `run.md` for the re-aimed plan.

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
