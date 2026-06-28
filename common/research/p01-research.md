# P1 Research Foundation — Verifier-Aware Learned Controller for Test-Time Compute

**Problem:** Can a lightweight learned controller allocate test-time compute per query while accounting for verifier imperfection?

**Compute:** 1×A100-40GB (~150–300 GPU-hrs)  
**Venues:** NeurIPS / ICLR / COLM  
**Project:** `p01-verifier-aware-controller/`  
**Parent index:** [`../../research.md`](../../research.md) §P1

---


### Problem Statement

> Can a lightweight learned controller allocate test-time compute per query while accounting for verifier imperfection?

**Core claim:** Current adaptive allocators (Snell, Damani, Re-FORC) assume the verifier/reranker is trustworthy. False positives cap best-of-N gains. A controller that estimates **both** expected value-of-compute **and** verifier-reliability per query is novel and directly extends a Meta-Verifier / value-of-verification line of work.

**Target bar:** ≥1.5–2× compute reduction at matched accuracy vs. best-of-N on MATH-500 / AIME; ablation showing the **trust head** beats difficulty-only allocation.

**Venues:** NeurIPS / ICLR / COLM  
**Compute:** 1×A100-40GB, ~150–300 GPU-hrs

---

### Implementation Walkthrough (Current Scaffold → Full System)

#### What exists today (`p01-verifier-aware-controller/`)

| Component | File | Status |
|---|---|---|
| Dual-head controller MLP | `src/core.py` — `VerifierAwareController` | ✅ value + verifier_trust heads, `decide_budget()` heuristic |
| Multi-task training loop | `src/train.py` | ✅ MSE (value) + BCE (trust), checkpointing |
| Synthetic data loader | `src/utils.py` — `make_synthetic_batch()` | ✅ smoke/demo only |
| Eval scaffold | `src/evaluate.py` | ⚠️ stub (no real generations) |
| Entry point | `scripts/run_experiment.py` | ✅ `--smoke-test`, train/eval modes |
| Config | `configs/default.yaml` | ✅ Qwen2.5-1.5B default, budget knobs |

```16:44:p01-verifier-aware-controller/src/core.py
class VerifierAwareController(nn.Module):
    def __init__(self, hidden_size: int = 2048, controller_dim: int = 128, num_layers: int = 2):
        ...
        self.value_head = nn.Linear(controller_dim, 1)          # expected value of extra compute
        self.trust_head = nn.Linear(controller_dim, 1)          # verifier trust / 1-FPR estimate
        ...
    def decide_budget(self, value, trust, base_budget=256, max_budget=2048, trust_threshold=0.6):
        scale = torch.clamp(trust, 0.3, 1.0)
        budget = (base_budget + (value * 1024) * scale).clamp(base_budget, max_budget).int()
        return budget
```

#### Phase 1 — Reproduce baselines (Weeks 1–4)

**Goal:** Establish the compute–accuracy frontier without the controller.

1. **Frozen base model:** Qwen3-4B-Instruct or Qwen2.5-7B-Instruct (match Re-FORC / Damani where possible).
2. **Benchmarks:** MATH-500, GSM8K, AIME 2024/25, GPQA-diamond.
3. **Baselines to implement in `src/evaluate.py`:**
   - **Fixed best-of-N** (N ∈ {1, 4, 8, 16, 32}) + majority vote or verifier selection
   - **Snell-style compute-optimal** (difficulty-conditioned search vs. revision — even a simplified version)
   - **Damani-style adaptive best-of-k** (dynamic k per query from reward curve forecaster)
   - **Difficulty-only controller** (entropy / length / single-head probe — ablation for P1)
   - **Fixed token budget** (256, 512, 1024, 2048)
4. **Metrics:** accuracy, tokens/query, samples/query, wall-clock (optional vLLM later).
5. **Output:** Pareto curves saved to `results/` as JSON + plots.

**Research rationale:** Snell (arXiv:2408.03314) proves allocation effectiveness varies by prompt difficulty; Damani (arXiv:2410.04707) proves learned allocators work. You cannot claim P1 gains without these curves on the same base model.

#### Phase 2 — Feature extraction pipeline (Week 4–5)

**Goal:** Replace synthetic hiddens with real frozen-LM features.

1. Add `src/features.py` (or extend `utils.py`):
   - Load frozen base LM with `transformers` + `accelerate`
   - Extract **last-token hidden** or **mean-pool over prompt tokens** at first forward pass (before generation)
   - Optionally: mid-generation hiddens for sequential allocation (Phase 4 extension)
2. Cache features under `data/features/{model}/{benchmark}.pt` to avoid re-forwarding.
3. Label schema per query:
   - `target_value`: marginal accuracy gain from +Δ compute (estimated offline by sweeping N or tokens)
   - `target_trust`: 1 − empirical FPR of verifier on similar difficulty bucket, or per-query verifier calibration score

**Research rationale:** Re-FORC trains lightweight forecasters on frozen Qwen3 hiddens — same recipe, extra trust head.

#### Phase 3 — Controller training (Weeks 5–7)

**Goal:** Train `VerifierAwareController` on real labels.

1. **Value head targets (choose one, document in paper):**
   - *Regression:* Δaccuracy from N=1→N=k for optimal k
   - *Classification:* bucketed difficulty (easy/medium/hard)
   - *Curve parameter:* fit reward-vs-compute curve per query (Damani formulation)
2. **Trust head targets:**
   - Binary: did verifier give FP on this query's generations?
   - Continuous: verifier ECE, or precision@operating-point from P2-style analysis
   - Proxy: agreement rate between two verifiers (cheap bootstrap)
3. **Loss:** `L = MSE(value) + λ·BCE(trust)` (current); consider focal loss on trust for rare FP events.
4. **Training:** 1–3 epochs on cached features; ≤100M params; AdamW 1e-4.

**Key ablation (required for publication):** difficulty-only (value head only) vs. full verifier-aware (value + trust).

#### Phase 4 — Stopping / allocation policy (Weeks 7–8)

**Goal:** Replace heuristic `decide_budget()` with principled policy.

**Options (implement at least two, compare):**

| Policy | Mechanism | Prior art |
|---|---|---|
| **Heuristic scale** (current) | `budget = f(value, trust)` | Re-FORC-style engineering baseline |
| **Threshold gating** | if trust < τ: reduce N cap | P2 operating-point logic |
| **Gittins / Pandora** | sequential decision under verifier noise | Optimal stopping literature; Snell §sequential |
| **Learned budget head** | direct regression to optimal token count | Damani reward-curve prediction |

When trust is low, the policy should **shrink N** or **switch to conservative aggregation** (min score, pessimistic verifier) — this is the novel coupling P1 adds over Damani/Re-FORC.

#### Phase 5 — Full evaluation harness (Weeks 8–10)

1. Generation loop: controller sets per-query budget → generate → verifier selects answer.
2. Report:
   - Accuracy vs. total tokens (Pareto)
   - Accuracy vs. samples
   - Breakdown by difficulty quartile
   - **Trust ablation:** remove trust head at inference
3. Statistical testing: bootstrap CIs on MATH-500.

#### Phase 6 — Analysis & paper artifacts

- Figure 1: problem — verifier FP ceiling limits best-of-N
- Figure 2: Pareto vs. baselines
- Figure 3: trust head calibration (reliability diagram)
- Table: per-benchmark compute savings at matched accuracy
- Failure cases: where trust head hurts (honest negative results)

---

### Research Foundation for P1

**Thesis chain:**

1. **Test-time compute scales** (Snell; Brown et al.) — more inference can beat bigger models *if allocated well*.
2. **Allocation can be learned** (Damani; Re-FORC; BEST-Route) — small probes on frozen LMs predict how much compute to spend.
3. **Verifiers are imperfect** (Lightman PRMs; GenRM; Stroebl; Agrawal) — false positives create a **ceiling** on best-of-N.
4. **Gap:** Steps 2 and 3 are not integrated. P1 trains a controller whose **trust head** modulates allocation when verifier FPR is high.

**Closest prior work (not equivalent):**

- Damani: predicts reward curve, no verifier uncertainty term
- Re-FORC: early stopping + model/thinking-length selection, assumes forecaster scores are actionable
- Snell: compute-optimal but uses PRM without explicit per-query FPR modeling
- Agrawal / Stroebl: analyze ceiling, do not propose allocation controller

---

### 20 Closest Papers for P1

| # | Paper | ID / Venue | Relevance to P1 |
|---|---|---|---|
| 1 | **Scaling LLM Test-Time Compute Optimally** — Snell, Lee, Xu, Kumar | arXiv:2408.03314, ICLR 2025 | Anchor: compute-optimal per-prompt allocation; PRM search vs. revision |
| 2 | **Learning How Hard to Think** — Damani, Shenfeld, Peng, Bobu, Andreas | arXiv:2410.04707, ICLR 2025 | Learned allocation; adaptive best-of-k; routing — direct baseline |
| 3 | **Re-FORC: Reasoning Feedback On Real Computations** — Zabounidis et al. | arXiv:2511.02130, NeurIPS 2025 ER Workshop | Lightweight forecaster on frozen Qwen3; early stopping — closest implementation recipe |
| 4 | **When Can LLMs Correct Their Own Mistakes?** — Stroebl, Kapoor, Narayanan | arXiv:2411.17501 | Formal imperfect-verifier ceiling; resampling limits |
| 5 | **Cut the Overcredit** — Agrawal et al. | OpenReview:7mVZy4mI1J (ICLR 2026 sub.) | FP ceiling in best-of-N selection; motivates trust head |
| 6 | **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** — Brown et al. | arXiv:2407.21770 | Repeated sampling scaling; best-of-N foundation |
| 7 | **Reasoning on a Budget: A Survey** | arXiv:2507.02076 | Survey framing adaptive test-time compute |
| 8 | **BEST-Route: Adaptive LLM Routing** — Microsoft | arXiv:2506.22716 | Learned routing between inference procedures |
| 9 | **Latency/Token-Aware Routing for LLMs** | arXiv:2509.09864 | Cost-aware allocation adjacent to budget policy |
| 10 | **Let's Verify Step by Step** — Lightman et al. | ICLR 2024 | PRM verifier backbone used in test-time search |
| 11 | **Generative Verifiers: Reward Modeling as Next-Token Prediction** — Zhang et al. | 2024 | GenRM verifier class; positivity bias context |
| 12 | **Training Verifiers to Solve Math Word Problems** — Cobbe et al. | 2021 | Outcome verifier training; best-of-N selection |
| 13 | **Solving Math Word Problems with Process- and Outcome-Based Feedback** — Uesato et al. | 2022 | Process vs. outcome verification |
| 14 | **Self-Consistency Improves Chain of Thought** — Wang et al. | ICLR 2023 | Parallel test-time compute baseline |
| 15 | **LLMs Are Better Reasoners with Self-Verification** — Weng et al. | 2023 | Self-verification; verifier bias precursor |
| 16 | **Scaling Laws for Reward Model Overoptimization** — Gao et al. | 2023 | Verifier reliability under distribution shift |
| 17 | **More Test-Time Compute Can Hurt** | arXiv:2603.15377 | Over-optimistic Q-estimates; motivates trust gating |
| 18 | **Test-Time Compute for LLM Agents** | arXiv:2506.12928 | Extends allocation to agentic settings |
| 19 | **LLMs Cannot Self-Correct Reasoning Yet** — Huang et al. | 2024 | Limits of verifier-guided revision |
| 20 | **ThinkPRM: Process Reward Models with Generative Reasoning** | OpenReview:V727xqBYIW | Data-efficient PRM; verifier architecture option for labels |

**Implementation priority for P1:** Papers 1–3 (baselines), 4–5 (motivation), 10–12 (verifier pipeline), then 6–9 (extended baselines).

---

---

*Last updated: 2026-06-26. Source: `research.md` §P1.*
