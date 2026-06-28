# How to Run — P9: Token-Efficient Agent Memory with Bounded Reasoning Cost

This document gives complete, reproducible instructions to set up and run experiments for this problem.

> **Status note:** This is the initial scaffold. Code implements basic structure, mock runs, and the core skeleton. Full research implementation (model training + strong baselines) will be built iteratively. Start by following the "Minimal smoke test" then move to real training.

## 1. Environment Setup

```bash
cd p09-agent-memory-efficient
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

## 3. Minimal Smoke Test (always works, even without GPU)

```bash
python -c "
from src.utils import hello
print(hello())
# or
python scripts/run_experiment.py --config configs/default.yaml --smoke-test
"

Expected: prints version info, runs a dummy forward pass or mock controller, writes a tiny result file to results/.
```

## 4. Full Experiment Run (example for this problem)

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

## 5. Key Hyperparameters / Config

See `configs/default.yaml`. Common levers:
- base_model
- controller_hidden_size / layers (keep very small: 32-256 dim)
- num_samples_for_best_of_n
- budget_range (tokens or steps)
- verifier_model (or oracle for ceiling studies)
- learning_rate, epochs (small: 1-3 epochs on frozen features often enough)

## 6. Expected Outputs & Metrics

- Accuracy vs. compute (tokens or FLOPs or wall time) curves
- Comparison table vs. best-of-N (fixed N=4,8,16,32), majority vote, difficulty-only router
- For verifier problems: FPR, precision-recall at operating points, ceiling shift
- Plots saved to results/
- JSONL generations + per-example traces

Target bar (from strategy for P1): ≥1.5-2× compute reduction at matched accuracy vs. best-of-N on MATH-500/AIME.

## 7. Reproducing Baselines

The code ships with:
- Naive best-of-N
- Difficulty estimator baseline (e.g. from hidden state entropy or simple probe)
- Random / fixed budget

See `src/evaluate.py` or run with `--baseline_only`.

## 8. Logging & Tracking

- Console + file logs to experiments/
- Optional: set `WANDB_PROJECT=...` and `wandb` will be used if installed.

## 9. Common Issues & Tips

- OOM: lower batch size, use gradient checkpointing on base (but frozen usually), or smaller controller.
- Slow generation: use vLLM if available for eval generations (future extension); start with HF generate.
- Reproducibility: fix seeds everywhere (see utils.set_seed).
- Dataset licenses: respect original dataset terms.

## 10. Iterating / Extending

1. Edit `src/` files.
2. Update `configs/`.
3. Add new benchmarks or ablations in `evaluate.py`.
4. When you have good numbers, update this `run.md` with exact commands + observed metrics.

## Current TODOs (update as you progress)

- [ ] Implement data loaders for MATH + AIME
- [ ] Implement frozen feature extraction from base model
- [ ] Core controller architecture (per problem)
- [ ] Loss that incorporates verifier reliability estimate
- [ ] Strong best-of-N + oracle verifier baselines
- [ ] Full training loop with early stopping
- [ ] Analysis + plotting scripts
- [ ] Reproduce key numbers from cited papers on same base models

## References (key papers to cite / implement against)

Memori, A-Mem, LoCoMo, MultiAgentBench, 'Why Do Multi-Agent LLM Systems Fail?'

Good luck — this is designed to be high-signal, low-compute research!
