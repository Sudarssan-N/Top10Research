# P6: Context-Length-Induced Degradation Under Perfect Retrieval

**Problem statement summary:**  
Diagnose attention dilution / other mechanisms for performance drop as context length grows even with perfect retrieval; propose simple mitigations (reorder/calibrate).

**Compute target:** 1×A100 (~80 GPU-hrs, $150-300)  
**Target venues:** ACL / EMNLP / ICLR  
**Priority:** Medium — **pivot/sharpen** to the mechanism (see verdict)  
**Key references:** anchor arXiv:2510.05381, Found-in-the-Middle arXiv:2406.16008, PINE arXiv:2407.01100

> **Verdict (2026-06-29, see [`research.md`](research.md)): PIVOT / SHARPEN.**
> The scaffolded version (show length hurts, then reorder/calibrate) is already solved:
> attention re-calibration = Found in the Middle (+15pp), reordering = the standard
> Lost-in-the-Middle remedy, and the anchor already isolates length from retrieval and
> ships a recitation fix. **Real white space = the MECHANISM:** nobody has run the
> orthogonalizing {token count} × {absolute evidence position} 2×2 under certified-perfect
> retrieval. The anchor's own masking result falsifies attention-dilution and points at
> positional/RoPE — a mechanism-targeted fix beating recitation is the contribution.
> **Scoop risk: concentrated** — Hao Peng's lab authored both the anchor and PINE.

## Goals (re-aimed)

- Build a **certified-perfect-retrieval** probe: the gold evidence is always present;
  only token-count and its absolute position vary.
- Run the **{few/many tokens} × {evidence at small/large position}** orthogonalization;
  show count-driven degradation independent of position.
- **Decisive mechanism test:** mask attention dilution out of the softmax — if accuracy
  still drops, the cause is positional/RoPE, not dilution.
- Engineer a training-free fix from the mechanism that **beats recitation**.

## Current Status (scaffold — placeholder)

- [x] Folder + basic structure
- [⚠️] `ContextDegradationProbe` is **demoted to an optional diagnostic** (see PIVOT NOTE
  in `src/core.py`) — it is not the contribution
- [ ] Controlled length×position generator (perfect retrieval guaranteed)
- [ ] Dilution-masking mechanism experiment
- [ ] Mechanism-targeted mitigation vs. recitation baseline

See `run.md` for the re-aimed plan.

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
