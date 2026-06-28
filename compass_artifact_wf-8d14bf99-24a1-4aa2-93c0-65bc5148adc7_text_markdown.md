# A*-Tier Publication Strategy: Top 10 Problem Statements & Top 10 Niches for a Solo ML-Systems Researcher (Late 2026 / Early 2027)

## TL;DR
- **Best bets:** Three problems sit at the intersection of your verification/test-time-compute background and modest compute: (1) a **learned per-query meta-controller that allocates test-time compute under a verifier-imperfection-aware budget**, (2) **calibrated, false-positive-bounded verifiers for best-of-N selection** (the "imperfect verifier ceiling" problem), and (3) **adaptive overthinking reduction via a lightweight latent controller**. All three are doable on a single A100 to an 8×A100 burst (<$1,000) by fine-tuning small controllers/probes on frozen 1.5B–8B open models.
- **Frontier signal:** The field has moved from "more test-time compute is better" to "allocate compute adaptively and verify reliably." Snell et al. (ICLR 2025) showed compute-optimal adaptive allocation improves test-time-compute efficiency by "more than 4x compared to a best-of-N baseline" and "on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14x larger model"; 2025 work (Re-FORC; Damani et al., "Learning How Hard to Think") shows learned controllers "reduce computation by up to 50% at no cost to response quality" and "55% less compute at equal accuracy." Simultaneously, imperfect verifiers are now known to impose an asymptotic ceiling on best-of-N — a wide-open, analysis-friendly gap.
- **Compute reality:** ~14 of the 20 directions are achievable on Colab Pro+/single-A100 or short multi-GPU bursts. The highest publication-probability-per-dollar work is **controller/probe/analysis papers on inference-time methods** — they need no pretraining, reward clever experiments, and play directly to your systems strengths.

## Key Findings

**1. The research frontier has shifted from scaling compute to *allocating* and *verifying* it.** Test-time scaling (Snell et al. 2024/ICLR 2025; Brown et al.) is now the dominant reasoning paradigm, but the open problems are no longer "does more compute help" — they are *how much*, *where*, and *can we trust the verifier*. Survey evidence ("Reasoning on a Budget," arXiv:2507.02076) confirms adaptive/controllable test-time compute is an organizing theme for 2025–2026.

**2. Verifier reliability is the single most exploitable gap for someone with your background.** Multiple 2025 papers document that process reward models (PRMs) and generative verifiers suffer positivity bias, overrate stronger models' outputs, and that false positives impose a hard ceiling on best-of-N gains. This is a quantitative, analysis-heavy area where a solo researcher with a strong experiment beats a big-lab compute advantage.

**3. Efficiency is bifurcating into "reasoning efficiency" (token/overthinking reduction) and "systems efficiency" (KV cache, quantization, MoE routing, speculative decoding).** The reasoning-efficiency side is more accessible to solo researchers because contributions are algorithmic/inference-time, not kernel-level.

**4. Mechanistic interpretability and evaluation methodology are "low-compute, high-prestige" niches.** SAE-based and probe-based papers, and LLM-as-judge reliability audits, are landing at top venues with single-GPU budgets.

**5. Your "FlashAttention character" instinct is correct and timely** — several niches (adaptive compute allocation, verifier calibration, overthinking) are exactly cases where industry optimizes the wrong quantity (raw samples/tokens) instead of value-per-compute.

---

## Details: The Research Landscape (2024–2026 scan)

### Test-time / inference-time scaling
Snell, Lee, Xu & Kumar, "Scaling LLM Test-Time Compute Optimally" (arXiv:2408.03314, ICLR 2025), is the anchor: a compute-optimal adaptive-per-prompt strategy "improve[s] the efficiency of test-time compute scaling by more than 4x compared to a best-of-N baseline," and in a FLOPs-matched comparison, "on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14x larger model" (conditional on non-trivial base success rates and low inference/pretrain token ratios). Methods split into parallel (best-of-N, self-consistency) and sequential (revision/self-correction). 2025 follow-ups extend this to agents (arXiv:2506.12928), latent space (arXiv:2509.26314), and temperature/sampling (arXiv:2510.02611). The clear open problem: principled, *learned* allocation rather than fixed N.

### Adaptive compute allocation (your sweet spot)
Damani, Shenfeld, Peng, Bobu & Andreas, "Learning How Hard to Think" (arXiv:2410.04707, ICLR 2025): "accurate computation-allocation procedures can be learned, and reduce computation by up to 50% at no cost to response quality, or improve quality by up to 10% at a fixed computational budget." Re-FORC (Zabounidis et al., arXiv:2511.02130, accepted at the NeurIPS 2025 Efficient Reasoning Workshop) trains lightweight forecaster adapters on frozen Qwen3 1.7B/4B/8B and reports "early stopping of unpromising reasoning chains, reducing compute by 26% while maintaining accuracy" and "optimized model and thinking length selection that achieves 4% higher accuracy at equal compute and 55% less compute at equal accuracy." BEST-Route (Microsoft, arXiv:2506.22716) and latency/token-aware routing (arXiv:2509.09864) confirm learned routers as a hot, low-compute sub-area. This is the closest extension of your Meta-Verifier Network thread.

### Verification / reward modeling
PRMs (Lightman et al.), GenRMs (Zhang et al. 2024), and critic models are the verifier backbone. Open problems documented in 2025: positivity bias in self-verification, weaker verifiers overrating stronger generations, and the "imperfect verifier ceiling." Agrawal et al. ("Cut the Overcredit," OpenReview id=7mVZy4mI1J, ICLR 2026 under review) formalize that "false positives impose an alignment ceiling in Best of N selection"; Stroebl, Kapoor & Narayanan (arXiv:2411.17501) prove resampling cannot reduce the false-positive probability — a hard ceiling. ThinkPRM (OpenReview V727xqBYIW) shows data-efficient generative PRMs with orders of magnitude fewer labels. "More Test-Time Compute Can Hurt" (arXiv:2603.15377) shows beam search degrades from over-optimistic Q-estimates.

### Reasoning efficiency / overthinking
Reasoning models generate far more tokens for often worse accuracy on simple tasks. LLMThinkBench (arXiv:2507.04023) finds Phi-4-reasoning models generate "~18x more tokens than standard models while achieving worse accuracy" — Phi-4-reasoning-plus averages 6,780.7 tokens at 69.54% accuracy versus standard Phi-4's 378.6 tokens at 78.92%; constrained to 1,024 tokens, Phi-4-reasoning fell "from 72.23% to 53.48% accuracy." "Wait, We Don't Need to 'Wait'" (EMNLP 2025 Findings, arXiv:2506.08343) suppresses thinking tokens to shorten chains without utility loss. Surveys (arXiv:2503.16419) and methods (L1, ThinkPrune, Arora & Zanette) target dynamic budget allocation. This is algorithmic, single-GPU-friendly work.

### LLM inference efficiency (systems)
- **KV cache:** KV cache can consume up to 70% of inference memory (ChunkKV, NeurIPS 2025, OpenReview 20JDhbJqn3). Dozens of methods (SnapKV, PyramidKV, KVzip, H2O, DuoAttention) — crowded, kernel-heavy, harder for solo standout but possible via clever eviction/analysis.
- **Speculative decoding:** Mature (2–3× speedups, native in vLLM/TensorRT-LLM). Open challenge: dynamically selecting among heterogeneous drafters (arXiv:2604.05417); training-free/entropy-based variants (arXiv:2512.23765).
- **Quantization:** W4A4 via rotations (QuaRot, SpinQuant, DuQuant). SVDQuant (ICLR 2025 Spotlight) hits W4A4 on diffusion, runs FLUX 12B on a 4090. Crowded but accessible.
- **MoE routing:** Load imbalance/router collapse; inference-time batch-aware routing without retraining (arXiv:2511.02237) is a fresh, low-compute angle.

### Small language models
Distillation from R1 gives strong SLM reasoning (Qwen-1.5B → 83.9% MATH-500). Phi-4-Mini-Reasoning recipe shows naive distillation hurts (LIMO/S1K degrade Phi-4-Mini). On-policy and step-wise distillation (SOD, arXiv:2605.07725) is emerging. Highly solo-friendly.

### RAG / long context
"Lost in the middle" persists; "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval" (arXiv:2510.05381) shows degradation even with perfect retrieval — a clean analysis target. Distraction-aware retrieval (arXiv:2509.21865) and attention calibration (arXiv:2406.16008) are accessible methods.

### Mechanistic interpretability
"Open Problems in Mechanistic Interpretability" (Sharkey et al., arXiv:2501.16496) is the agenda-setter. SAEs are the dominant tool but the January 2025 update notes they "still underperform simple baselines on safety-relevant tasks." Weight-sparse transformers with interpretable circuits (Gao et al. 2025, arXiv:2511.13653) and causal-generalization critiques (arXiv:2602.16698) are fresh. Single-GPU-friendly with pretrained SAEs (Gemma-Scope).

### LLM-as-judge / evaluation
Position bias, verbosity bias, self-enhancement bias, authority bias all documented (arXiv:2406.07791; survey arXiv:2411.16594). Frontier models exceed 50% error on some bias tests. Automated bias *discovery* (BiasScope, arXiv:2602.09383) is a fresh frontier. Pure-API or single-GPU; extremely solo-friendly.

### Efficient fine-tuning / PEFT
LoRA variants (DoRA, AdaLoRA, VeRA, DeLoRA) are crowded but QLoRA enables 7B fine-tuning on a $1,500 4090. Standout requires a genuinely new insight, not another variant.

### Agentic systems & multi-agent eval
"Why Do Multi-Agent LLM Systems Fail?" (Cemri et al., arXiv:2503.13657) and many benchmarks (MultiAgentBench, AgentBench). Agent cost is a live pain point: unconstrained agents cost $5–8/task; context grows quadratically. Memory efficiency (A-Mem; Memori reports 81.95% LoCoMo accuracy "using only 1,294 tokens per query (~5% of full context)," versus a full-context baseline of ~26,031 tokens/query) is hot and cheap to study.

### Diffusion / flow efficiency
Few-step distillation (DMD, SiD, SANA-Sprint, rectified/flow-matching). SVDQuant Spotlight. More compute-hungry than LLM-inference work; image models fine-tune on a 4090 but video is out of budget.

---

## TOP 10 PROBLEM STATEMENTS

### P1. Can a lightweight learned controller allocate test-time compute per query while accounting for verifier imperfection?
- **(a) Problem/why it matters:** Current adaptive allocation assumes the verifier is trustworthy, but false positives cap best-of-N gains. A controller that estimates *both* expected value-of-compute and verifier-reliability per query directly extends your Meta-Verifier Network.
- **(b) Tractable/novel now:** Frozen base models + a small trained head; Re-FORC and Damani et al. proved the recipe works, but neither integrates verifier-error awareness.
- **(c) Gap:** No existing controller jointly models difficulty and verifier false-positive risk to decide compute.
- **(d) Approach:** Train a small (≤100M-param) controller/probe on hidden states of a frozen Qwen3-4B/8B that predicts the reward-vs-tokens curve and a verifier-trust score; use a Pandora's-box/Gittins-index stopping policy. Eval on MATH-500, AIME 2024/25, GPQA.
- **(e) Compute:** 1×A100-40GB, ~150–300 GPU-hrs, ~$300–600 cloud.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med-high novelty, low risk (incremental on proven base).
- **(h) Fit:** NeurIPS / ICLR / COLM.

### P2. Calibrated, false-positive-bounded verifiers for best-of-N selection.
- **(a) Problem:** False positives impose an asymptotic ceiling on best-of-N (Agrawal et al.; Stroebl et al.). A verifier explicitly trained/calibrated to minimize false positives at a chosen operating point could raise the ceiling.
- **(b) Tractable now:** Verifier calibration is a small-model fine-tuning + analysis task.
- **(c) Gap:** Verifiers are trained for accuracy, not for low-FPR operating points relevant to selection.
- **(d) Approach:** Fine-tune a 7B PRM/GenRM with an asymmetric (precision-first) loss; analyze the resulting best-of-N ceiling shift; compare ROC-aware aggregation (min/pessimistic) vs. mean.
- **(e) Compute:** 1–4×A100, ~200 GPU-hrs, ~$400–800.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** High novelty, med risk.
- **(h) Fit:** NeurIPS / ICLR.

### P3. A latent controller for adaptive overthinking reduction.
- **(a) Problem:** Reasoning models overthink simple problems (~18× more tokens, sometimes lower accuracy). 
- **(b) Tractable now:** Thinking-token suppression and budget methods exist but are static/heuristic.
- **(c) Gap:** No per-query learned controller that decides reasoning depth from internal signals at generation time.
- **(d) Approach:** Train a small probe on hidden states to predict "convergence" and trigger early termination; compare to NoWait, ThinkPrune, L1 on math/code.
- **(e) Compute:** 1×A100, ~100–200 GPU-hrs, ~$200–400.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med novelty, low risk.
- **(h) Fit:** ACL / EMNLP / COLM.

### P4. Does test-time compute improve or degrade confidence calibration, and can a probe fix it?
- **(a) Problem:** "Over-reasoning impairs confidence calibration" (arXiv:2508.15050); reasoning models often stay overconfident.
- **(b) Tractable now:** Calibration analysis is pure inference + small probe.
- **(c) Gap:** Mixed findings on whether reasoning helps calibration; no lightweight corrective probe at low-FPR operating points.
- **(d) Approach:** Systematic calibration study across reasoning budgets on 3–4 open reasoning models; train a confidence-recalibration probe; report ECE and low-FPR recall.
- **(e) Compute:** 1×A100, ~80–150 GPU-hrs, ~$150–350.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med novelty, low risk (strong analysis paper).
- **(h) Fit:** ICLR / COLM / EMNLP.

### P5. Batch-aware MoE expert routing at inference without retraining.
- **(a) Problem:** Batched inference activates the union of experts, negating sparsity; load imbalance creates stragglers.
- **(b) Tractable now:** Opportunistic activation (arXiv:2511.02237) shows training-free routing tweaks work.
- **(c) Gap:** Per-batch routing that trades tiny accuracy for large decode speedup is underexplored on open MoEs (gpt-oss-20b, Qwen MoE).
- **(d) Approach:** Profiling + a lightweight batch-aware router; measure throughput vs. accuracy on a single node.
- **(e) Compute:** 2–8×A100 burst (MoE memory), ~100 GPU-hrs, ~$400–900.
- **(f) Feasibility:** Med.
- **(g) Novelty/risk:** Med-high novelty, med risk (systems baselines demanding).
- **(h) Fit:** MLSys / NeurIPS / ICLR.

### P6. Context-length-induced degradation under perfect retrieval: diagnosis and mitigation.
- **(a) Problem:** Models degrade as context grows even with perfect retrieval (arXiv:2510.05381).
- **(b) Tractable now:** Controlled-injection experiments + a model-agnostic mitigation.
- **(c) Gap:** Retrieval-isolating benchmarks overestimate long-context progress; mechanism underexplored.
- **(d) Approach:** Mechanistic study (attention dilution) + a simple reordering/calibration fix on open long-context models.
- **(e) Compute:** 1×A100, ~80 GPU-hrs, ~$150–300.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med novelty, low risk.
- **(h) Fit:** ACL / EMNLP / ICLR.

### P7. Automated discovery of novel biases in LLM-as-judge.
- **(a) Problem:** Known biases (position, verbosity, authority) are catalogued, but discovery is manual.
- **(b) Tractable now:** BiasScope (arXiv:2602.09383) opened automated discovery; lots of headroom.
- **(c) Gap:** Scalable, generalizable bias-discovery methods with causal validation.
- **(d) Approach:** Build a contrastive perturbation + causal-test pipeline; validate on CodeJudgeBench/MTBench; quantify FPR shifts.
- **(e) Compute:** Mostly API + 1×A100 for open judges, ~50 GPU-hrs, ~$100–300 incl. API.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med-high novelty, low risk.
- **(h) Fit:** ACL / EMNLP / NeurIPS D&B.

### P8. Step-wise on-policy distillation for small reasoning models under tool use.
- **(a) Problem:** Naive distillation degrades SLMs; cascading tool errors amplify divergence (SOD, arXiv:2605.07725).
- **(b) Tractable now:** SLM distillation runs on a single A100; recipes are unsettled.
- **(c) Gap:** Optimal divergence-adaptive distillation recipe for ≤4B tool-integrated reasoners.
- **(d) Approach:** Step-wise divergence-reweighted distillation on Qwen3-1.7B/4B; eval AIME/GPQA/code.
- **(e) Compute:** 1–2×A100, ~150 GPU-hrs, ~$300–600.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med novelty, med risk (crowded).
- **(h) Fit:** COLM / ICLR / ACL.

### P9. Token-efficient agent memory with bounded reasoning cost.
- **(a) Problem:** Agent context grows quadratically; unconstrained agents cost $5–8/task.
- **(b) Tractable now:** Structured memory (Memori: 81.95% LoCoMo accuracy at ~1,294 tokens/query vs. ~26,031 for full context) shows large wins are available.
- **(c) Gap:** Principled controller deciding what to remember/forget to bound per-task compute at fixed task success.
- **(d) Approach:** Train a lightweight memory-management policy; eval on LoCoMo + an agent benchmark; report tokens/task vs. success.
- **(e) Compute:** 1×A100 + API, ~80 GPU-hrs, ~$200–400.
- **(f) Feasibility:** High.
- **(g) Novelty/risk:** Med novelty, med risk.
- **(h) Fit:** NeurIPS / COLM / EMNLP.

### P10. Dynamic heterogeneous drafter selection for speculative decoding.
- **(a) Problem:** Most speculative decoding uses one drafter; dynamically choosing among heterogeneous drafters is open (arXiv:2604.05417).
- **(b) Tractable now:** Training-free entropy/alignment-based selection (arXiv:2512.23765) shows feasibility.
- **(c) Gap:** A learned, input-adaptive drafter selector with acceptance-rate guarantees.
- **(d) Approach:** Small bandit/router selecting drafters per context; measure wall-clock speedup on a single GPU.
- **(e) Compute:** 1–2×A100, ~100 GPU-hrs, ~$250–500.
- **(f) Feasibility:** Med.
- **(g) Novelty/risk:** Med novelty, med risk (rigorous latency baselines needed).
- **(h) Fit:** MLSys / ICLR / NeurIPS.

---

## TOP 10 NICHE AREAS (rising, low-but-growing competition)

1. **Verifier-aware test-time compute allocation** — Hot because allocation now beats brute scaling; contributions landing: learned controllers/routers, value-of-compute estimators. Compute: single-A100.
2. **Verifier/PRM calibration & the imperfect-verifier ceiling** — Hot due to RLVR reliance; contributions: precision-first verifiers, ROC-aware aggregation, ceiling analyses. Compute: 1–4×A100.
3. **Reasoning efficiency / anti-overthinking** — Hot for deployment; contributions: adaptive termination, budget controllers, token-reduction. Compute: single-A100.
4. **Confidence calibration of reasoning models** — Emerging; contributions: calibration-vs-budget studies, recalibration probes. Compute: single-A100.
5. **Automated bias discovery in LLM-as-judge** — Fresh after BiasScope; contributions: discovery pipelines, security/robustness of judges. Compute: API + single-GPU.
6. **Inference-time MoE routing without retraining** — Rising systems niche; contributions: batch-aware routing, replica allocation, plug-and-play balancing. Compute: multi-GPU burst.
7. **Token-efficient agent memory & plan caching** — Hot with agent cost crisis; contributions: structured memory, RL memory managers, plan reuse. Compute: single-GPU + API.
8. **Long-context degradation under perfect retrieval** — Emerging analysis niche; contributions: mechanistic diagnoses, model-agnostic mitigations, better benchmarks. Compute: single-A100.
9. **Small reasoning model distillation recipes (incl. tool use)** — Crowded but rising; contributions: on-policy/step-wise distillation, data-quality recipes. Compute: 1–2×A100.
10. **SAE-based actionable interpretability (steering/debugging)** — Prestige niche; contributions: domain-specific SAEs, causal-generalization critiques, steering for safety/correctness. Compute: single-A100 with pretrained SAEs.

---

## Compute & Cost Analysis

| Problem | GPU profile | Est. GPU-hrs | Est. cost (USD) | Category |
|---|---|---|---|---|
| P1 Verifier-aware controller | 1×A100-40GB | 150–300 | $300–600 | Single-A100 |
| P2 FP-bounded verifiers | 1–4×A100 | 200 | $400–800 | Multi-GPU burst |
| P3 Overthinking controller | 1×A100 | 100–200 | $200–400 | Single-A100 |
| P4 Calibration probe | 1×A100 | 80–150 | $150–350 | Colab-friendly/Single-A100 |
| P5 MoE routing | 2–8×A100 | 100 | $400–900 | Multi-GPU burst |
| P6 Long-context degradation | 1×A100 | 80 | $150–300 | Colab-friendly/Single-A100 |
| P7 Judge bias discovery | API + 1×A100 | 50 | $100–300 | Colab-friendly |
| P8 SLM distillation | 1–2×A100 | 150 | $300–600 | Single-A100 |
| P9 Agent memory | 1×A100 + API | 80 | $200–400 | Single-A100 |
| P10 Drafter selection | 1–2×A100 | 100 | $250–500 | Single-A100 |

**ROI verdict:** Best publication-probability-per-dollar comes from **P1, P3, P4, P6, P7** — all single-A100 or Colab-friendly, analysis/controller-driven, and forgiving of compute limits. P2 is the highest-prestige bet but needs slightly more compute and risk tolerance. Avoid P5/P10 as *first* projects unless you want a systems-venue (MLSys) target, because rigorous latency/throughput baselines against vLLM/TensorRT-LLM are demanding for a solo author.

---

## Recommendations (prioritized shortlist)

**Start now (rank 1): P1 — Verifier-aware learned controller for test-time compute.** This is the maximal-fit extension of your Meta-Verifier Network: it keeps your "value of verification" thesis but reframes it as joint difficulty × verifier-trust allocation, which no existing controller does. Proven recipe (Re-FORC/Damani), single-A100, high acceptance probability at NeurIPS/ICLR/COLM. Benchmark to hit: beat best-of-N by ≥2× compute at equal accuracy on MATH-500/AIME (Snell et al. already demonstrate >4× is attainable with oracle difficulty, so a learned, verifier-aware version clearing 2× is a credible, defensible bar), and show the verifier-trust term adds measurable gains over difficulty-only allocation.

**Rank 2: P2 — False-positive-bounded verifiers.** Highest novelty and prestige; directly attacks the "imperfect verifier ceiling," a hot 2025–2026 theme. Slightly more risk/compute. Pursue this if P1's preliminary results are strong and you want a swing for a top oral/spotlight.

**Rank 3 (hedge): P4 — Calibration vs. test-time compute.** A clean, low-risk analysis paper that shares infrastructure with P1/P2 and can be produced quickly as a parallel/backup submission. Pure analysis papers with a small corrective probe are reliably accepted and de-risk the cycle.

**Staging plan:** (1) Weeks 1–4: reproduce Snell/Re-FORC baselines on Qwen3-4B, single A100. (2) Weeks 5–10: build P1 controller + verifier-trust head; if false-positive analysis is compelling, branch into P2. (3) Run P4 calibration study in parallel using the same generations (near-zero marginal compute). **Threshold to switch focus:** if P1's controller cannot beat best-of-N by ≥1.5× compute-efficiency by week 8, pivot the lead result to P2 (ceiling analysis) or P4 (calibration), both of which need less to clear the bar.

---

## Caveats
- The "14× larger model" Snell result is **conditional** — it holds "on problems where a smaller base model attains somewhat non-trivial success rates" and at low inference/pretrain token ratios; do not over-claim it.
- Re-FORC (arXiv:2511.02130, a NeurIPS 2025 workshop paper) and several cited 2026 papers are **preprints/workshop papers, not yet archival peer-reviewed**; treat their numbers as indicative.
- The Agrawal et al. "Cut the Overcredit" ceiling result is an **under-review submission** (OpenReview id=7mVZy4mI1J) with no confirmed arXiv ID; for a peer-reviewed citation of the ceiling, use Stroebl, Kapoor & Narayanan (arXiv:2411.17501).
- Systems niches (KV cache, quantization, speculative decoding, MoE) are **crowded and kernel-heavy**; a solo standout needs either a clean analysis insight or strong wall-clock results against vLLM/TensorRT-LLM, which raises the bar.
- Acceptance is never guaranteed by topic fit; execution quality, baselines, and a crisp single claim dominate outcomes at A*-tier venues.
- Compute estimates assume ~$2/A100-hr cloud pricing and frozen-base/small-controller training; full fine-tuning of larger models would change categories.