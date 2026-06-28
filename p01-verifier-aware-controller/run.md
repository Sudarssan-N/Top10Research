# How to Run — P1: Verifier-Aware Learned Controller for Test-Time Compute

This document gives complete, reproducible instructions to set up and run experiments for this problem.

> **Status note:** This is the initial scaffold. Code implements basic structure, mock runs, and the core skeleton. Full research implementation (model training + strong baselines) will be built iteratively. Start by following the "Minimal smoke test" then move to real training.

## 1. Environment Setup

```bash
cd p01-verifier-aware-controller
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

- **Local:** [`research.md`](research.md) → [`../common/research/p01-research.md`](../common/research/p01-research.md)
- **Master index:** [`../research.md`](../research.md) §P1

**P0 papers to implement first:** Snell (#1), Damani (#2), Re-FORC (#3), Stroebl (#4), Agrawal (#5).

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

## 5. Phase 1 — Baseline Reproduction (START HERE after smoke test)

Phase 1 establishes the **compute–accuracy frontier** without the controller: best-of-N, oracle ceiling, difficulty routing, fixed token budgets.

**Smoke (no GPU, no model download):**
```bash
python scripts/run_phase1_baselines.py --smoke-test
# or
python scripts/run_experiment.py --mode phase1 --smoke-test
```

**Cache datasets:**
```bash
python scripts/prepare_data.py --benchmarks gsm8k,math500,aime24
```

**Small real run (GPU, 20 examples, debug):**
```bash
python scripts/run_phase1_baselines.py \
  --config configs/phase1.yaml \
  --max-examples 20 \
  --n-values 1,4,8
```

**Full Phase 1:**
```bash
python scripts/run_phase1_baselines.py --config configs/phase1.yaml --n-values 1,2,4,8,16
```

**Outputs** (under `results/phase1/`):
- `phase1_summary.json` — all metrics
- `phase1_pareto.png` — accuracy vs. avg samples
- `{benchmark}/metrics.json`, `{benchmark}/pareto_rows.json`

**Baselines implemented:**
| Strategy | Description |
|---|---|
| `single_n1` | One sample (pass@1) |
| `best_of_n_majority` | Self-consistency / majority vote |
| `pass_at_n_oracle` | Oracle selection ceiling (any correct sample) |
| `best_of_n_random` | Random pick among N |
| `best_of_n_longest` | Pick longest chain |
| `difficulty_routed_majority` | k∈{1, N/4, N} by prompt-length tertiles |
| `fixed_token_budget` | Single sample at max_new_tokens ∈ {128,256,512,1024} |

## 5. Phase 2 — Feature Extraction + Label Cache

Extract **frozen prompt hiddens** and offline **value/trust labels** from N-sample rollouts.

**Smoke (extract + train, no GPU):**
```bash
python3 scripts/run_phase2.py --smoke-test
```

**Extract only (GPU, 100 examples per benchmark):**
```bash
python3 scripts/run_phase2.py --mode extract --max-examples 100 --label-n 8
```

**Train controller from cache:**
```bash
python3 scripts/run_phase2.py --mode train --config configs/phase2.yaml
```

**Both:**
```bash
python3 scripts/run_phase2.py --mode both --max-examples 100
```

**Cache location:** `data/features/{model_slug}/{benchmark}.pt`

Each record contains:
| Field | Meaning |
|---|---|
| `hidden` | last-token (or mean-pool) prompt embedding |
| `target_value` | normalized compute usefulness ∈ [0,1] |
| `target_trust` | verifier reliability proxy (agreement − FP penalty) |
| `target_optimal_k_norm` | min k for pass / N |
| `verifier_fp_event` | pass@N but majority wrong |

**Outputs:** `results/phase2/phase2_extract_summary.json`, `checkpoints/best_controller.pt`

## 6. Phase 3 — Train Ablations + Controller Eval

Full training with validation, **difficulty-only ablation**, and controller-guided allocation vs fixed-N.

**Colab (recommended for GPU):** open `colab/P1_Verifier_Aware_Controller.ipynb` or:
```bash
python scripts/colab_run_all.py --max-examples 200 --label-n 8
```

**Smoke (local, no GPU):**
```bash
python3 scripts/run_phase3.py --smoke-test
```

**Train both variants:**
```bash
python3 scripts/run_phase3.py --mode train --config configs/phase3.yaml
```

**Eval controllers vs fixed-N:**
```bash
python3 scripts/run_phase3.py --mode eval --max-examples 100
```

**Variants trained:**
| Variant | Description |
|---|---|
| `verifier_aware` | value + trust heads (full P1) |
| `difficulty_only` | value head only (trust=1.0) — required ablation |

**Outputs:** `results/phase3/{variant}/checkpoints/.../best_controller.pt`, `phase3_eval_summary.json`, `phase3_comparison.png`

## 7. Full Experiment Run (Phase 4+ policy tuning)

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

## 8. Key Hyperparameters / Config

See `configs/default.yaml`. Common levers:
- base_model
- controller_hidden_size / layers (keep very small: 32-256 dim)
- num_samples_for_best_of_n
- budget_range (tokens or steps)
- verifier_model (or oracle for ceiling studies)
- learning_rate, epochs (small: 1-3 epochs on frozen features often enough)

## 8. Expected Outputs & Metrics

- Accuracy vs. compute (tokens or FLOPs or wall time) curves
- Comparison table vs. best-of-N (fixed N=4,8,16,32), majority vote, difficulty-only router
- For verifier problems: FPR, precision-recall at operating points, ceiling shift
- Plots saved to results/
- JSONL generations + per-example traces

Target bar (from strategy for P1): ≥1.5-2× compute reduction at matched accuracy vs. best-of-N on MATH-500/AIME.

## 9. Reproducing Baselines

The code ships with:
- Naive best-of-N
- Difficulty estimator baseline (e.g. from hidden state entropy or simple probe)
- Random / fixed budget

See `src/evaluate.py` or run with `--baseline_only`.

## 10. Logging & Tracking

- Console + file logs to experiments/
- Optional: set `WANDB_PROJECT=...` and `wandb` will be used if installed.

## 11. Common Issues & Tips

- OOM: lower batch size, use gradient checkpointing on base (but frozen usually), or smaller controller.
- Slow generation: use vLLM if available for eval generations (future extension); start with HF generate.
- Reproducibility: fix seeds everywhere (see utils.set_seed).
- Dataset licenses: respect original dataset terms.

## 12. Iterating / Extending

1. Edit `src/` files.
2. Update `configs/`.
3. Add new benchmarks or ablations in `evaluate.py`.
4. When you have good numbers, update this `run.md` with exact commands + observed metrics.

## Current TODOs (update as you progress)

- [x] Core controller (VerifierAwareController) with value + verifier_trust heads + decide_budget policy
- [x] Smoke-test runnable end-to-end (controller forward, multi-task loss demo, budget allocation)
- [x] Phase 1 baselines: best-of-N sweep, majority/oracle/random/longest, difficulty routing, fixed budgets
- [x] Benchmark loaders: GSM8K, MATH-500, AIME 2024 (+ mock for smoke)
- [x] Math grading: boxed/GSM8K extraction + sympy numeric check
- [ ] Full-scale Phase 1 runs on Qwen2.5-1.5B/4B (all benchmarks, no max_examples cap)
- [x] Phase 2: frozen hidden extraction (`src/features.py`) + label cache (`src/labels.py`, `src/phase2.py`)
- [x] Phase 2: train controller from cached features (`scripts/run_phase2.py`)
- [ ] Full-scale Phase 2 extract on GSM8K + MATH-500 (no max_examples cap)
- [ ] Implement frozen feature extraction from base model hidden states (last token / mean pool)
- [ ] Synthetic or real verifier labels (use a larger model or oracle for training the controller)
- [ ] Full training loop, checkpointing, logging (wandb optional)
- [ ] Baselines: best-of-N (various N), difficulty-only controller, fixed budget, Re-FORC-style
- [ ] Full evaluation harness: generations, majority / verifier selection, token accounting, accuracy vs compute curves
- [ ] Analysis + plotting scripts + tables
- [ ] Reproduce / beat key numbers (compute efficiency) from cited papers on same base models (Qwen-1.5B/4B/7B)

## References (key papers to cite / implement against)

Snell et al. (ICLR 2025), Damani et al. (ICLR 2025), Re-FORC, Agrawal et al. (imperfect verifier ceiling)

Good luck — this is designed to be high-signal, low-compute research!
