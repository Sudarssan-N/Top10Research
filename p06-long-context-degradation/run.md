# How to Run — P6: Context-Length-Induced Degradation Under Perfect Retrieval

> **Read [`research.md`](research.md) first.** Verdict (2026-06-29): **PIVOT / SHARPEN**.
> The reorder/calibrate mitigations are already published; the contribution is the
> **mechanism**. `src/core.py`'s `ContextDegradationProbe` is now only a diagnostic.
> Scoop risk is concentrated in one lab (anchor + PINE) — move fast or pick another problem.

## 1. Environment

```bash
cd p06-long-context-degradation
pip install -r requirements.txt
huggingface-cli login        # if the long-context model is gated
```

**Hardware:** 1×A100. A 7–8B model at 64k context fits with FlashAttention-2; reduce
`context_lengths` for smaller GPUs.

## 2. Smoke test (CPU, no model)

```bash
PYTHONPATH=. python scripts/run_experiment.py --smoke-test
```
Exercises only the (now-diagnostic) probe — it does not validate the real design.

## 3. Re-aimed plan — isolate length from position from retrieval

1. **Certified-perfect-retrieval generator.** Build contexts where the gold evidence is
   *always present and verbatim*; vary only (a) total token count and (b) the gold's
   absolute position. Retrieval is never the failure mode by construction.
2. **The 2×2 (the core experiment).** Cross `{few tokens, many tokens}` ×
   `{evidence at small absolute position, large absolute position}`. Sweep
   `context_lengths` × `evidence_positions` from the config. Show degradation tracks
   token *count* at fixed position (pure length effect).
3. **Decisive mechanism test.** Re-run with attention-dilution masked out of the softmax
   (the anchor's manipulation). If accuracy still drops → mechanism is positional/RoPE,
   not dilution. This is the falsification that justifies the paper.
4. **Mechanism-targeted fix.** Build a training-free intervention from step 3 (e.g.
   position re-mapping à la PINE) and show it **beats the recitation baseline** and the
   Found-in-the-Middle attention-calibration baseline.

## 4. Config

See `configs/default.yaml`: `base_model`, `context_lengths`, `evidence_positions`,
`num_distractors` (keep gold present), `eval_task`, `mitigation ∈ {none, reorder,
attn_calibrate, recitation}`.

## 5. Success bar

A clean count-vs-position disentanglement under guaranteed-perfect retrieval + a decisive
masked-dilution result + a mechanism-targeted fix that beats recitation. Pure replication
of the anchor is not publishable.
