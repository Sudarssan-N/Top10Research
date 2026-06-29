# How to Run — P5: Batch-Aware MoE Expert Routing at Inference

> **Read [`research.md`](research.md) first.** Verdict (2026-06-29): **PIVOT / DEPRIORITIZE**.
> The problem is heavily scooped (OEA, Lynx, SERE, XShare). The only defensible angle is
> the **end-to-end vLLM throughput study OEA never did**. The current `src/core.py` is a
> placeholder (trains a router — contradicts training-free; FLOP-proxy "speedup"). Do not
> invest real compute here until P1 has produced numbers and you accept the scoop window.

## 1. Environment

```bash
cd p05-batch-moe-routing
pip install -r requirements.txt        # + vllm for the serving baseline
huggingface-cli login                  # if the chosen MoE is gated
```

**Hardware:** the MoEs worth testing don't fit fp16 on one 40GB A100 (Mixtral ~90GB,
Qwen3-30B ~60GB). Plan for quantization (AWQ/GPTQ) or CPU-offload — note that this
**changes which regime is bottlenecked**, so report it explicitly.

## 2. Smoke test (CPU, no model)

```bash
PYTHONPATH=. python scripts/run_experiment.py --smoke-test
```
This only exercises the placeholder router. It does **not** validate the real method.

## 3. Re-aimed plan (if pursued)

The measurement is the contribution — not a new routing idea.

1. **Union-of-experts profiling.** Instrument a real open MoE (`Qwen/Qwen1.5-MoE-A2.7B`,
   Mixtral) and record the number of *distinct* experts activated per layer as batch size
   sweeps `[1, 4, 16, 64]`. Confirm union → total as B grows (the core tension).
2. **Training-free policy.** Over the model's own gating logits, apply opportunistic
   (OEA-style piggyback), similarity re-route (SERE-style), or batch budget (XShare-style)
   selection. No trained router.
3. **Accuracy check.** Verify no statistically significant downstream accuracy loss
   (MMLU / GSM8K / a code task — pick what the model is evaluated on, *not* MATH-by-default).
4. **End-to-end wall-clock harness.** Adopt the MoE-Inference-Bench (arXiv:2508.17467)
   vLLM TTFT/throughput methodology. Report **tokens/s + peak memory vs. vLLM** — beware
   that custom routing falls off vLLM's fused grouped-GEMM / CUDA-graph fast path, so fewer
   experts can be *net slower*.
5. **Ablations.** batch-adaptive k₀; per-layer expert budgets (the two open problems OEA names).

## 4. Config

See `configs/default.yaml`: `base_model`, `num_experts`, `top_k_experts`,
`routing_strategy ∈ {baseline, opportunistic, similarity, budget}`, `expert_budget`,
`batch_sizes`, `serving_baseline`.

## 5. Success bar

Real throughput / peak-memory win vs. vLLM on ≥2 open MoEs at matched accuracy — in the
single-GPU offload regime no prior paper measured. FLOP counts and MoE-layer-only latency
do **not** count.
