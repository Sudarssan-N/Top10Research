# P6: Context-Length-Induced Degradation Under Perfect Retrieval

**Problem statement (frozen 2026-09-10):**  
After certified-perfect retrieval and after distractors are removed from the softmax, does remaining accuracy still track softmax support \(n\), RoPE relative distance \(m\), or the query's absolute index — and does the implied training-free fix beat recitation / FitM / PINE / STRING on reasoning tasks?

Full RQ, Du-protocol verification, and five-layout factorial: [`../common/research/p06-problem-statement-2026-09.md`](../common/research/p06-problem-statement-2026-09.md).

**Compute target:** 1×A100 (~80 GPU-hrs, $150-300)  
**Target venues:** ACL / EMNLP / ICLR  
**Priority:** Medium — **pivot/sharpen** to the mechanism (see verdict)  
**Key references:** anchor arXiv:2510.05381, Found-in-the-Middle arXiv:2406.16008, PINE arXiv:2407.01100

> **Verdict (2026-09-10): KEEP, replace the 2×2 language.**
> Existence and recitation are Du et al.\ (EMNLP 2025 Findings). STRING already remaps
> large relative indices (ICLR 2025). Peng lab theory is in NeurIPS 2026 review.
> Du's mask is `[E][MASK][Q]` (few tokens, **large** relative distance). Du's end
> condition is `[WS][E][Q]` (many tokens, **small** relative distance). The unpublished
> cell is `[MASK][E][Q]`. That cell, plus STRING-on-mask, is the paper. Current
> `core.py` KV offset grid is a retrieval control, not the headline.

## Goals (re-aimed, 2026-09-10)

- Replicate Du layouts B (`[E][MASK][Q]`) and C (`[WS][E][Q]`) on open 7–9B models.
- Run **Layout D** `[MASK][E][Q]` — few softmax tokens, small relative distance, large query index.
- Run **Layout E** — STRING remap of Du's mask.
- Preregister which of M1 / M2-rel / M2-abs survives D.
- Bake off the implied fix vs recitation, FitM, PINE, STRING on GSM8K-at-length (accuracy / extra tokens).
- Keep the KV-registry 2×2 only as a NIAH **control**.

## Current Status (pivot implemented, 2026-07-08 — no real model run yet)

- [x] `src/core.py` rewritten: certified-perfect-retrieval KV task generator with the
  orthogonal {total tokens} × {ABSOLUTE evidence offset} 2×2 (the old probe is gone)
- [x] `src/phase1.py`: grid eval, main effects + length×position interaction with
  bootstrap CIs; smoke plants a position-only failure in a mock model and asserts the
  analysis recovers it (effect_position = −1, effect_length = 0)
- [x] `recite` mitigation flag (recitation baseline) wired into the task prompts
- [ ] Real 2×2 on Qwen2.5-1.5B (T4) then 7B + 32K contexts (A100) — `scripts/colab_run_all.py`
- [ ] Dilution-masking follow-up + mechanism-targeted fix that beats recitation

## Directory layout

```
p06-long-context-degradation/
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
