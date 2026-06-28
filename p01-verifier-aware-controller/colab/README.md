# P1 on Google Colab (GPU)

Mac/local CPU is fine for **smoke tests** only. Full runs need a Colab **T4/A100** GPU.

## Quick start

### Option A — Notebook (recommended)

1. Open [`P1_Verifier_Aware_Controller.ipynb`](./P1_Verifier_Aware_Controller.ipynb) in Colab.
2. **Runtime → Change runtime type → T4 GPU** (or A100 if available).
3. Run all cells top to bottom.

### Option B — Upload zip

1. Zip this folder: `p01-verifier-aware-controller/`
2. In Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # upload zip
   !unzip -q p01-verifier-aware-controller.zip -d /content/
   %cd /content/p01-verifier-aware-controller
   ```
3. Run:
   ```bash
   !python scripts/colab_run_all.py --max-examples 200 --label-n 8
   ```

### Option C — Google Drive

1. Copy `p01-verifier-aware-controller` to `MyDrive/Research/`
2. In Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/Research/p01-verifier-aware-controller
   ```

## Hugging Face token

```python
from huggingface_hub import login
login()  # paste token — needed for Qwen weights
```

## Pipeline commands

| Step | Command |
|---|---|
| Smoke (2 min, no HF) | `!python scripts/colab_run_all.py --smoke-test` |
| Full Colab run | `!python scripts/colab_run_all.py --max-examples 200` |
| Phase 1 only | `!python scripts/run_phase1_baselines.py --config configs/colab.yaml --max-examples 200` |
| Phase 2 extract | `!python scripts/run_phase2.py --mode extract --config configs/colab.yaml --max-examples 200 --force` |
| Phase 3 | `!python scripts/run_phase3.py --mode both --config configs/phase3.yaml --max-examples 200` |

## Save results to Drive

```python
!cp -r results/colab_run /content/drive/MyDrive/Research/p01_results_$(date +%Y%m%d)
```

## Compute guide (Colab)

| Setting | ~Time (T4) | VRAM |
|---|---|---|
| `--smoke-test` | < 2 min | minimal |
| `--max-examples 50` | ~30–60 min | ~8–12 GB |
| `--max-examples 200` | ~2–4 hrs | ~12–16 GB |
| Full GSM8K test (1319) | ~8–12 hrs | use A100 |

## Outputs

```
results/colab_run/
  phase1/phase1_summary.json + phase1_pareto.png
  phase2/phase2_extract_summary.json
  data/features/Qwen__Qwen2.5-1.5B-Instruct/*.pt
  phase3/phase3_train_summary.json
  phase3/phase3_eval_summary.json + phase3_comparison.png
```