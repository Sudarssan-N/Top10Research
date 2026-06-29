# How to Run — P7: Automated Discovery of Novel Biases in LLM-as-Judge

> **Read [`research.md`](research.md) first.** Verdict (2026-06-29): **PIVOT**.
> The discovery framing is scooped by BiasScope (ICLR 2026, real). The residual is **causal**
> validation in the **code-judge** domain. `src/core.py`'s `BiasDiscoveryProbe` is a
> placeholder — the real pipeline is prompt/LLM-driven perturbation + API judge + a causal
> estimator. Mostly API cost; minimal GPU.

## 1. Environment

```bash
cd p07-bias-discovery-llm-judge
pip install -r requirements.txt
export OPENAI_API_KEY=...      # or your judge/perturbation provider keys
```

## 2. Smoke test (CPU, no API)

```bash
PYTHONPATH=. python scripts/run_experiment.py --smoke-test
```
Exercises only the placeholder probe — it does not call a judge or validate the method.

## 3. Re-aimed plan — causal bias discovery on code judges

1. **Testbed.** Use CodeJudgeBench (arXiv:2507.10535): pairs of code answers with
   unit-test verdicts, so "answer quality" is *ground truth*, not a judged approximation.
2. **Quality-preserving perturbations.** Use `perturbation_model` to apply a candidate
   factor (rename vars, reformat, add comments, change identifier authority, pad verbosity)
   while keeping all unit tests passing. Discard any perturbation that changes pass/fail —
   this is what makes "quality held fixed" defensible.
3. **Judge query.** Ask `judge_model` to prefer original vs. perturbed (both orders, to net
   out position bias). Record preference flips.
4. **Causal estimate.** Compute the **flip-rate ATE** per factor with **bootstrap CIs**
   (`n_bootstrap`); run **mediation** to rule out confounds (did a hidden quality change,
   not the factor, drive the flip?).
5. **Discovery loop.** Beyond the seed `candidate_bias_factors`, let the perturbation model
   propose new factors; keep those with significant, mediation-robust ATEs → candidate
   *novel* biases. Compare against BiasScope's catalogue to claim genuinely new ones.

## 4. Config

See `configs/default.yaml`: `judge_model`, `perturbation_model`, `testbed`,
`candidate_bias_factors`, `n_perturbations_per_pair`, `n_bootstrap`.

## 5. Success bar

≥1 causally-validated bias (significant flip-rate ATE, mediation-robust, quality provably
fixed by unit tests) that BiasScope did not report — i.e. discovery *with causal rigor in a
domain where quality control is real*, not another catalogue.
