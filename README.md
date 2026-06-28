# Top 10 A*-Tier Problem Statements: Solo ML-Systems Research (2026-2027)

This workspace contains starter implementations and experiment scaffolds for the **Top 10 Problem Statements** from the publication strategy document.

## Overview

The research focuses on:
- Test-time compute allocation and verification
- Reasoning efficiency
- Inference systems for LLMs
- Small models and agents
- Evaluation and interpretability

**Prioritization (from strategy):**
1. **P1** — Verifier-aware learned controller (start here)
2. **P2** — Calibrated false-positive-bounded verifiers
3. **P4** — Calibration probe (hedge/analysis)
4. Others as parallel tracks

All projects target low-to-modest compute: primarily single A100 / Colab Pro+ friendly (or short bursts). Designed for frozen base models (Qwen3 1.7B-8B range) + lightweight controllers/probes.

## Directory Structure

Each problem is a self-contained mini-project (repo-like). **Per-problem research docs** (20-paper maps, thesis chains, implementation phases) live in `common/research/`; each `pXX-*/research.md` symlinks there.

```
common/research/                   # Canonical p01/p02/p03-research.md (+ future P4–P10)
p01-verifier-aware-controller/     # Highest priority starter
p02-calibrated-fp-verifiers/
p03-overthinking-latent-controller/
p04-calibration-probe/
p05-batch-moe-routing/
p06-long-context-degradation/
p07-bias-discovery-llm-judge/
p08-slm-distillation-tooluse/
p09-agent-memory-efficient/
p10-speculative-drafter-selection/
```

Inside each:
- `README.md` — Problem summary, goals, status
- `run.md` — Complete instructions: setup, run, reproduce, expected outputs
- `requirements.txt` — Python deps
- `src/` — Core Python code (trainers, models, evals)
- `scripts/` — Entry-point scripts / data prep
- `configs/` — YAML/JSON experiment configs
- `data/` — Local datasets / cached (gitignored)
- `experiments/` — Run logs, scripts
- `results/` — Metrics, plots, generations

## Getting Started

1. Choose a problem (recommend **p01** first).
2. `cd <problem-dir>`
3. Follow `run.md` (usually: `pip install -r requirements.txt`, download data, run training/eval scripts).
4. Use the provided skeletons and iterate (these are research scaffolds — extend with your experiments).

## Common Environment Notes

- Python 3.10+
- CUDA-capable GPU recommended (A100 / 4090 / RTX 30/40/50 series). Most code falls back to CPU for tiny tests but will be slow.
- Hugging Face Hub access for base models (Qwen2.5/Qwen3, Llama-3, Phi-4 etc.)
- Datasets: MATH, GSM8K, AIME (via huggingface or local), GPQA, etc.
- For full runs you may need ~$200-800 cloud credits depending on problem.

## Status

**Initial scaffolds + deep iteration on P1-P4 + meaningful code for P5-P10 complete:**

- Folder structure + per-problem README/run.md/requirements/src/scripts/configs: ✅
- P1-P4 (top priorities): full custom cores (controllers/probes), asymmetric/specialized losses, synthetic data generators, runnable smoke tests + train loops, FPR/ECE/early-stop/budget demos, updated TODOs.
- P5-P10: customized cores (MoE batch router, context degradation probe + reorder, bias discovery, distillation reweighter, memory policy, drafter selector), smoke demos (speedup, reorder, bias strength, importance, memory select, drafter choice), basic train paths, synthetic loaders.
- All run.mds + README status updated with progress.
- Direct runnable (PYTHONPATH + base torch/numpy) smoke/train verified for all.

See top-level README and individual problem run.md for how to run (start with P1). Next: deepen with real model hidden extraction, real benchmarks (MATH etc), baselines, plots per problem.

Run any problem smoke:
  cd pXX-... && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  python scripts/run_experiment.py --smoke-test


See individual `run.md` for current implementation status and TODOs inside each problem.

## Original Strategy Document

The source of truth is `compass_artifact_wf-8d14bf99-24a1-4aa2-93c0-65bc5148adc7_text_markdown.md` (the full problem statements + citations + compute analysis).

## Next Steps / Roadmap

- Implement and baseline P1 (verifier-aware controller) against best-of-N and simple difficulty controllers.
- Reproduce key numbers from Snell et al., Re-FORC, Damani et al.
- Add proper logging (wandb optional), checkpointing.
- Build shared utilities across projects (e.g., common eval harness for MATH/AIME).
- For systems problems (P5, P10), add latency/throughput measurement harnesses.

Contributions welcome in the spirit of open solo research — PRs that add clean experiments, better baselines, or ablations are great.

**Start with P1!** Open `p01-verifier-aware-controller/run.md` and follow it.
