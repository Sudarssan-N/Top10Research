# How to Run — P3: Latent Controller for Adaptive Overthinking Reduction

This document gives complete, reproducible instructions to set up and run experiments for this problem.

> **Status note:** This is the initial scaffold. Code implements basic structure, mock runs, and the core skeleton. Full research implementation (model training + strong baselines) will be built iteratively. Start by following the "Minimal smoke test" then move to real training.

## 1. Environment Setup

```bash
cd p03-overthinking-latent-controller
python -m venv .venv
source .venv/bin/activate   # or conda / your preferred
pip install --upgrade pip
pip install -r requirements.txt
```

**Recommended hardware:** See top of this doc (usually 1x A100 or equivalent).  
For smoke tests: CPU or small GPU is fine (models will be tiny or mocked at first).

**HF login (for models/datasets):**
```bash
huggingface-cli login
# or set HF_TOKEN env
```

## 2. Data Preparation

Most projects use:
- MATH (hendrycks_math or math dataset on HF)
- GSM8K
- AIME 2024/2025 (usually manual or community splits; scripts will download)
- GPQA (diamond)

The `scripts/prepare_data.py` (or equivalent in src) will cache them under `data/`.

Run once:
```bash
python scripts/prepare_data.py   # if present, or use the logic in evaluate/train
```

## 3. Research & Literature (read first)

Full 20-paper map, thesis chain, and phased build plan:

- **Local:** [`research.md`](research.md) → [`../common/research/p03-research.md`](../common/research/p03-research.md)
- **Master index:** [`../research.md`](../research.md) §P3

**P0 papers to implement first:** LLMThinkBench (#1), NoWait (#2), L1 (#4), Re-FORC (#7), Snell (#20).

**Phase 1 next step:** reproduce token–accuracy Pareto (fixed max, NoWait, budget forcing) before training the stop probe.

## 4. Minimal Smoke Test (always works, even without GPU)

```bash
python -c "
from src.utils import hello
print(hello())
# or
python scripts/run_experiment.py --config configs/default.yaml --smoke-test
"

Expected: prints version info, runs a dummy forward pass or mock controller, writes a tiny result file to results/.
```

## 5. Full Experiment Run (example for this problem)

Typical pattern (will evolve):

```bash
# Train the controller / probe / etc.
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --mode train \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir results/run-001

# Evaluate (generations + metrics)
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --mode eval \
    --checkpoint results/run-001/best_controller.pt \
    --benchmarks math500,aime24,gpqa

# Or combined train+eval script for convenience
bash experiments/run_full.sh
```

## 6. Key Hyperparameters / Config

See `configs/default.yaml`. Common levers:
- base_model
- controller_hidden_size / layers (keep very small: 32-256 dim)
- num_samples_for_best_of_n
- budget_range (tokens or steps)
- verifier_model (or oracle for ceiling studies)
- learning_rate, epochs (small: 1-3 epochs on frozen features often enough)

## 7. Expected Outputs & Metrics

- Accuracy vs. compute (tokens or FLOPs or wall time) curves
- Comparison table vs. best-of-N (fixed N=4,8,16,32), majority vote, difficulty-only router
- For verifier problems: FPR, precision-recall at operating points, ceiling shift
- Plots saved to results/
- JSONL generations + per-example traces

Target bar (from strategy for P1): ≥1.5-2× compute reduction at matched accuracy vs. best-of-N on MATH-500/AIME.

## 8. Reproducing Baselines

The code ships with:
- Naive best-of-N
- Difficulty estimator baseline (e.g. from hidden state entropy or simple probe)
- Random / fixed budget

See `src/evaluate.py` or run with `--baseline_only`.

## 9. Logging & Tracking

- Console + file logs to experiments/
- Optional: set `WANDB_PROJECT=...` and `wandb` will be used if installed.

## 10. Common Issues & Tips

- OOM: lower batch size, use gradient checkpointing on base (but frozen usually), or smaller controller.
- Slow generation: use vLLM if available for eval generations (future extension); start with HF generate.
- Reproducibility: fix seeds everywhere (see utils.set_seed).
- Dataset licenses: respect original dataset terms.

## 11. Iterating / Extending

1. Edit `src/` files.
2. Update `configs/`.
3. Add new benchmarks or ablations in `evaluate.py`.
4. When you have good numbers, update this `run.md` with exact commands + observed metrics.

## Current TODOs (update as you progress)

- [x] OverthinkController probe (stop prob) + batch early-stop decider + min-steps policy
- [x] Smoke test (stop decision demo) + training loop (BCE on convergence target) on synthetic data
- [ ] Hidden state extraction during generation (e.g. save last hidden at each step of CoT)
- [ ] Label construction: "would early stop have been safe?" using final answer correctness or length-ablation
- [ ] Training + inference-time integration (hook into HF generate or vLLM to stop early)
- [ ] Baselines: fixed max, NoWait, ThinkPrune-style, L1 budget
- [ ] Eval: tokens used vs accuracy on MATH/GSM8K/AIME; overthinking reduction %
- [ ] Analysis plots + comparison to cited overthinking papers
- [ ] Reproduce key claims (e.g. ~18x token reduction on simple problems without accuracy loss)

## References (key papers to cite / implement against)

LLMThinkBench, 'Wait, We Don't Need to Wait' (EMNLP 2025), ThinkPrune, L1, Re-FORC

Good luck — this is designed to be high-signal, low-compute research!
