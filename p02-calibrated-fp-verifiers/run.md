# How to Run — P2: Calibrated, False-Positive-Bounded Verifiers for Best-of-N

This document gives complete, reproducible instructions to set up and run experiments for this problem.

> **Status note:** This is the initial scaffold. Code implements basic structure, mock runs, and the core skeleton. Full research implementation (model training + strong baselines) will be built iteratively. Start by following the "Minimal smoke test" then move to real training.

## 1. Environment Setup

```bash
cd p02-calibrated-fp-verifiers
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

## 3. Phase 1 — Ceiling & Verifier Baselines (start here)

Quantify the false-positive ceiling before training the FPR-bounded probe (Stroebl / Agrawal motivation).

```bash
# Fast smoke (mock generator + mock benchmark, ~30s)
python scripts/run_phase1_ceiling.py --smoke-test

# Or via main entry
python scripts/run_experiment.py --mode phase1 --smoke-test

# Real model (downloads HF weights; start small)
python scripts/run_phase1_ceiling.py \
  --config configs/phase1.yaml \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --benchmarks math500 \
  --max-examples 20 \
  --n-values 1,2,4,8,16
```

**Outputs** (under `results/phase1/`):
- `p2_ceiling_summary.json` — accuracy vs. N per verifier + theoretical ceiling
- `p2_ceiling_curves.png` — ceiling shift figure
- `{benchmark}/ceiling_metrics.json` — per-benchmark detail

**Verifiers in Phase 1 (mock proxies → replace with real PRM/GenRM):**
- `noisy_optimistic_prm` — positivity-bias PRM (high FP rate)
- `pessimistic_prm` — precision-first proxy
- `outcome_heuristic` — Cobbe-style outcome baseline
- `oracle` — upper bound

See [`research.md`](research.md) (→ [`../common/research/p02-research.md`](../common/research/p02-research.md)) for the full 20-paper map and phase plan.

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

- [x] Core FPRBoundedVerifier (hidden probe + score) + threshold search for target FPR + metrics (FPR/precision/recall)
- [x] Asymmetric / focal / standard BCE training losses (FP-weighted)
- [x] Calibration harness (ECE, temperature scaling, precision@τ)
- [x] Phase 1 ceiling analysis (`phase1.py`, `ceiling.py`, `verifier.py`, `run_phase1_ceiling.py`)
- [x] Smoke test (forward + low-FPR threshold demo) and train/eval on synthetic labels
- [ ] Data loaders for MATH/AIME + generation of multiple candidates per query
- [ ] Frozen feature extraction (hidden states from base model on candidate answers)
- [ ] Verifier labels (use stronger model / human / execution for "is_correct")
- [ ] Best-of-N simulation with different verifiers + ceiling analysis (Agrawal/Stroebl style)
- [ ] Compare standard accuracy loss vs precision-first vs ROC/pessimistic aggregation
- [ ] Full training, logging, analysis + plots (FPR vs accuracy lift)
- [ ] Reproduce/beat relevant numbers from ceiling papers on open models

## References (key papers to cite / implement against)

Agrawal et al. (Cut the Overcredit), Stroebl et al. (arXiv:2411.17501), PRM/GenRM literature

Good luck — this is designed to be high-signal, low-compute research!
