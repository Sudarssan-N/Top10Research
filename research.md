# Base Research & Literature Map — Top 10 Problem Statements

**Purpose:** Master index for all ten problems. Detailed per-problem research (20-paper maps, phases, thesis chains) lives in [`common/research/`](common/research/). This file keeps the landscape overview, cross-problem infrastructure, and next actions.

**Source of truth for problem definitions:** `compass_artifact_wf-8d14bf99-24a1-4aa2-93c0-65bc5148adc7_text_markdown.md`

**Status:** Living document. **P1–P7** research foundations complete (full web-verified docs in `common/research/`); **P8–P10** still use the inline 20-paper maps below. P4–P7 each carry a 2026-06-29 **verdict** — all four are *pivot* recommendations (re-aim before committing compute), with P4 retained as the cheap rank-3 hedge. See each per-problem doc for the critical assessment.

---

## Table of Contents

1. [Research Landscape (2024–2026)](#1-research-landscape-20242026)
   - Per-problem docs: [`common/research/`](common/research/) (P1–P3 complete)
2. [P1 — Verifier-Aware Learned Controller](#p1--verifier-aware-learned-controller-for-test-time-compute)
3. [P2 — Calibrated FP-Bounded Verifiers](#p2--calibrated-false-positive-bounded-verifiers)
4. [P3 — Overthinking Latent Controller](#p3--latent-controller-for-adaptive-overthinking-reduction)
5. [P4 — Calibration vs. Test-Time Compute](#p4--test-time-compute-vs-confidence-calibration)
6. [P5 — Batch-Aware MoE Routing](#p5--batch-aware-moe-expert-routing-at-inference)
7. [P6 — Long-Context Degradation](#p6--context-length-induced-degradation-under-perfect-retrieval)
8. [P7 — Automated Judge Bias Discovery](#p7--automated-discovery-of-novel-biases-in-llm-as-judge)
9. [P8 — SLM Distillation with Tool Use](#p8--step-wise-on-policy-distillation-for-small-reasoning-models)
10. [P9 — Token-Efficient Agent Memory](#p9--token-efficient-agent-memory-with-bounded-reasoning-cost)
11. [P10 — Heterogeneous Drafter Selection](#p10--dynamic-heterogeneous-drafter-selection-for-speculative-decoding)
12. [Cross-Problem Infrastructure](#cross-problem-shared-infrastructure)
13. [Citation Caveats](#citation-caveats)

---

## 1. Research Landscape (2024–2026)

The field has shifted from **“does more test-time compute help?”** to **“how should we allocate and verify it?”**

| Theme | Anchor insight | Open gap our problems attack |
|---|---|---|
| **Adaptive allocation** | Snell et al. and Damani et al. show per-prompt allocation beats fixed best-of-N by 2–4×+ | No controller jointly models **difficulty** and **verifier trust** (P1) |
| **Verifier imperfection** | False positives impose a hard ceiling on best-of-N (Stroebl; Agrawal) | Verifiers trained for accuracy, not low-FPR selection (P2) |
| **Reasoning efficiency** | Reasoning models over-generate tokens on easy inputs (~18×) | No per-query latent stop controller at generation time (P3) |
| **Calibration** | More reasoning can worsen ECE | No systematic budget-vs-calibration study + corrective probe (P4) |
| **Systems inference** | MoE batching, speculative decoding, agent memory are deployment bottlenecks | Training-free / lightweight controllers at inference (P5, P9, P10) |

**Organizing survey:** *Reasoning on a Budget* (arXiv:2507.02076) — adaptive/controllable test-time compute as the 2025–2026 theme.

---

## P1 — Verifier-Aware Learned Controller for Test-Time Compute

> **Full research doc:** [`common/research/p01-research.md`](common/research/p01-research.md)  
> **Project symlink:** `p01-*/research.md`

**Problem:** Allocate test-time compute per query while accounting for verifier imperfection.

**Target bar:** ≥1.5–2× compute reduction at matched accuracy vs. best-of-N; trust-head ablation beats difficulty-only.

**P0 papers:** Snell, Damani, Re-FORC, Stroebl, Agrawal. **Phases:** baselines → features → controller train → policy → eval.

---

## P2 — Calibrated, False-Positive-Bounded Verifiers

> **Full research doc:** [`common/research/p02-research.md`](common/research/p02-research.md)  
> **Project symlink:** `p02-*/research.md`

**Problem:** Train/calibrate verifiers for low-FPR selection operating points; raise best-of-N ceiling.

**Target bar:** Measurable ceiling shift at FPR≤5% on MATH-500/GSM8K; beat standard PRM at matched FPR.

**P0 papers:** Stroebl, Agrawal, Lightman, Cobbe, Dalal, Snell, Guo/Nixon calibration. **Phases:** ceiling baselines → labels → FPR probe → threshold policy → ceiling-shift eval.

---

## P3 — Latent Controller for Adaptive Overthinking Reduction

> **Full research doc:** [`common/research/p03-research.md`](common/research/p03-research.md)  
> **Project symlink:** `p03-*/research.md`

**Problem:** Per-query probe on hidden states for reasoning depth / early termination.

**Target bar:** ≥1.5–2× token reduction at matched accuracy on easy quartile vs. fixed-max CoT.

**P0 papers:** LLMThinkBench, NoWait, L1, Re-FORC, Snell. **Phases:** token Pareto baselines → mid-CoT hiddens → stop probe → generation hook → full eval.

---

## P4 — Test-Time Compute vs. Confidence Calibration

> **Full research doc:** [`common/research/p04-research.md`](common/research/p04-research.md)  
> **Project symlink:** `p04-*/research.md`

**Problem:** Does increasing reasoning budget help or hurt calibration, and can a lightweight *post-hoc* probe fix it without retraining the base model?

**Verdict (2026-06-29):** On track as the rank-3 **hedge**, but **pivot the framing**. The phenomenon (more reasoning → worse calibration; reasoning models stay overconfident) and the hidden-state correctness probe are each already occupied. Defensible white space: a **budget-conditioned, transferable post-hoc recalibration map** for *frozen* reasoning models, judged on **selective risk (AURC / risk–coverage) with a split-conformal coverage guarantee**, under an explicit no-retrain constraint. Cheapest problem to run; reuses P1 generations. Risk: scoop on the descriptive half; budget→ECE may not replicate on math (it is non-monotonic).

**Anchor / P0 papers:** Over-Reasoning Impairs Calibration (arXiv:2508.15050); Reasoning Models Know When They're Right (arXiv:2504.05419); RLCR — the retrain-allowed upper bound (arXiv:2507.16806); Guo temperature scaling (arXiv:1706.04599); Tian "Just Ask" (arXiv:2305.14975). Full web-verified list + citation caveats in the per-problem doc.

---

## P5 — Batch-Aware MoE Expert Routing at Inference

> **Full research doc:** [`common/research/p05-research.md`](common/research/p05-research.md)  
> **Project symlink:** `p05-*/research.md`

**Problem:** Can training-free batch-aware routing reduce union-of-experts activation during batched MoE inference, improving wall-clock throughput/memory without accuracy loss?

**Verdict (2026-06-29):** **Pivot / deprioritize below P1/P3/P4** — most-scooped problem in the portfolio. The exact idea is the anchor OEA (arXiv:2511.02237) and was already done training-free a year earlier by Lynx (arXiv:2411.08982); the serving-integration white space OEA left open is being closed *now* by SERE (arXiv:2602.07616, vLLM-integrated) and XShare (arXiv:2602.07265, vLLM RFC). Only un-taken slice: the **end-to-end vLLM throughput study OEA never did** (offload/single-GPU regime, batch-adaptive k₀, per-layer budgets) — a systems grind with a shrinking window, and the MoEs worth testing don't fit fp16 on one 40GB A100. Current `core.py` is a placeholder (it *trains* a router, contradicting "training-free"; its speedup is a FLOP proxy).

**Anchor / P0 papers:** OEA (arXiv:2511.02237); Lynx (arXiv:2411.08982); SERE (arXiv:2602.07616); XShare (arXiv:2602.07265); MoE-Inference-Bench harness (arXiv:2508.17467). Full web-verified list in the per-problem doc.

---

## P6 — Context-Length-Induced Degradation Under Perfect Retrieval

> **Full research doc:** [`common/research/p06-research.md`](common/research/p06-research.md)  
> **Project symlink:** `p06-*/research.md`

**Problem:** After certified-perfect retrieval, which length still hurts — softmax support \(n\), RoPE relative distance \(m\), or the query's absolute index — and does the implied training-free fix beat recitation / FitM / PINE / STRING on reasoning tasks?

**Verdict (2026-09-10):** **Keep, replace the 2×2 language.** Existence and recitation are Du et al.\ (EMNLP 2025 Findings, arXiv:2510.05381). STRING already remaps large relative indices (ICLR 2025). Peng lab theory is in NeurIPS 2026 review (arXiv:2605.15514). Du's mask is `[E][MASK][Q]` (few \(n\), **large** \(m\)); Du's end condition is `[WS][E][Q]` (many \(n\), **small** \(m\)). The unpublished cell is `[MASK][E][Q]`. Frozen protocol: [`common/research/p06-problem-statement-2026-09.md`](common/research/p06-problem-statement-2026-09.md). Target ICML 2027 / ACL 2027.

**Anchor / P0 papers:** Context Length Alone Hurts (arXiv:2510.05381); STRING (arXiv:2410.18745, ICLR 2025); RoPE Distinguishes Neither (arXiv:2605.15514); Found in the Middle (arXiv:2406.16008); PINE (arXiv:2407.01100); Lost in the Middle (TACL 2024). Full protocol + verified Du layouts in the problem-statement doc.

---

## P7 — Automated Discovery of Novel Biases in LLM-as-Judge

> **Full research doc:** [`common/research/p07-research.md`](common/research/p07-research.md)  
> **Project symlink:** `p07-*/research.md`

**Problem:** Can we automatically *discover* (not just catalogue) evaluation biases in LLM judges, with causal validation?

**Verdict (2026-06-29):** **Pivot** — substantially scooped. **BiasScope (arXiv:2602.09383) is real (ICLR 2026)** — the earlier "maybe speculative" warning was *wrong* — and it already does the proposed automated discovery + correctness-preserving counterfactual perturbations (48 validated biases, JudgeBench-Pro); Automated Concept Discovery (arXiv:2603.03319) is a second scooper. Defensible residual: **causal** validation (flip-rate ATE + bootstrap CIs + mediation to rule out confounds) in the **executably-verifiable code-judge domain** (CodeJudgeBench, arXiv:2507.10535), where unit-test pass/fail makes "answer quality held fixed" ground truth rather than an approximation. Fast-moving (Bias-in-the-Loop, arXiv:2604.16790). Current `core.py` (MLP on pair embeddings) does not match the real prompt/LLM-driven perturbation + API-judge + causal-estimation method.

**Anchor / P0 papers:** BiasScope (arXiv:2602.09383); Automated Concept Discovery (arXiv:2603.03319); CodeJudgeBench (arXiv:2507.10535); self-enhancement bias — Panickssery et al. (arXiv:2404.13076); MT-Bench — Zheng et al. (arXiv:2306.05685). Full web-verified list in the per-problem doc.

---

## P8 — Step-Wise On-Policy Distillation for Small Reasoning Models (Tool Use)

### Problem Statement

> Can divergence-adaptive, step-wise on-policy distillation beat naive distillation for ≤4B tool-using reasoners?

### Implementation Notes (Scaffold)

- `DistillationReweighter` in `p08-slm-distillation-tooluse/src/core.py`
- Qwen3-1.7B/4B; eval on AIME/GPQA + tool-use traces

### 20 Closest Papers for P8

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | SOD — Step-wise On-policy Distillation | arXiv:2605.07725 | Direct method precursor |
| 2 | DeepSeek-R1 Technical Report | 2025 | Reasoning distillation source |
| 3 | Phi-4-Mini-Reasoning Technical Report | 2025 | SLM reasoning recipe |
| 4 | LIMO — Less Is More for Reasoning | 2025 | Data-efficient reasoning |
| 5 | s1 — Simple test-time scaling | 2025 | Budget forcing / data |
| 6 | On-Policy Distillation / GKD | 2024 | Policy matching |
| 7 | Self-Distillation for LLMs | 2024 | Teacher-free variant |
| 8 | Hinton — Knowledge Distillation | 2015 | Foundation |
| 9 | Toolformer — Schick et al. | 2023 | Tool-use training |
| 10 | Gorilla — Patil et al. | 2023 | API tool use |
| 11 | Qwen2.5-Math / Qwen3 reports | 2024–2025 | Base SLM targets |
| 12 | STaR | 2022 | Bootstrapped reasoning data |
| 13 | Rejection Sampling Fine-Tuning | 2023 | On-policy data filter |
| 14 | DPO vs. SFT for reasoning | 2024 | Alignment comparison |
| 15 | OpenR1 / MiniR1 community recipes | 2025 | Open reproduction targets |
| 16 | Process Supervision for reasoning | ICLR 2024 | Step-level labels |
| 17 | AgentFlan / tool-agent training | 2024 | Tool-integrated agents |
| 18 | LIMO/S1K failure on Phi-4-Mini (reported) | 2025 | Negative result — naive distill hurts |
| 19 | RLHF / RLVR for reasoning | 2025 | Alternative to distillation |
| 20 | Cascade tool error amplification studies | 2025 | Motivates step-wise reweighting |

---

## P9 — Token-Efficient Agent Memory with Bounded Reasoning Cost

### Problem Statement

> Can a lightweight memory policy bound per-task token cost while preserving agent success rate?

### Implementation Notes (Scaffold)

- `MemoryPolicy` in `p09-agent-memory-efficient/src/core.py`
- Eval on LoCoMo + agent benchmark; report tokens/query vs. success

### 20 Closest Papers for P9

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | Memori — efficient agent memory | 2025 | 81.95% LoCoMo at ~1,294 tok/query |
| 2 | A-Mem — Agentic Memory | 2025 | Structured agent memory |
| 3 | LoCoMo — Very Long-Term Conversational Memory | 2024 | Primary benchmark |
| 4 | Why Do Multi-Agent LLM Systems Fail? — Cemri et al. | arXiv:2503.13657 | Failure modes / cost |
| 5 | MultiAgentBench | 2025 | Multi-agent eval |
| 6 | AgentBench — Liu et al. | 2023 | Agent capability benchmark |
| 7 | Generative Agents — Park et al. | 2023 | Memory architecture |
| 8 | Reflexion — Shinn et al. | 2023 | Verbal memory for agents |
| 9 | Voyager — Wang et al. | 2023 | Skill library memory |
| 10 | MemGPT / Letta | 2024 | OS-style memory management |
| 11 | RAG for Agents surveys | 2024–2025 | Retrieval memory |
| 12 | SWE-agent | 2024 | Code agent context management |
| 13 | Long-term Memory in LLM Agents survey | 2025 | Field map |
| 14 | MemoryBank | 2023 | Long-term dialog memory |
| 15 | ChatDB — database memory | 2023 | Structured external memory |
| 16 | Context Compression for Agents | 2024 | Token reduction |
| 17 | Plan Caching for LLM Agents | 2025 | Reuse across tasks |
| 18 | Agent cost / pricing analyses | 2024–2025 | $5–8/task motivation |
| 19 | Retrieval-Augmented Agents | 2024 | Hybrid memory |
| 20 | ReAct — Yao et al. | 2023 | Agent loop foundation |

---

## P10 — Dynamic Heterogeneous Drafter Selection for Speculative Decoding

### Problem Statement

> Can an input-adaptive selector choose among heterogeneous drafters to maximize speculative decoding speedup?

### Implementation Notes (Scaffold)

- `DrafterSelector` in `p10-speculative-drafter-selection/src/core.py`
- Bandit/router over {small, medium, large} drafters; measure acceptance rate × wall-clock

### 20 Closest Papers for P10

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | Dynamic Heterogeneous Drafter Selection | arXiv:2604.05417 | Primary open problem citation |
| 2 | Training-Free Drafter Selection | arXiv:2512.23765 | Entropy/alignment heuristic baseline |
| 3 | Fast Inference from Transformers via Speculative Decoding — Leviathan et al. | ICML 2023 | Foundation |
| 4 | Medusa — Cai et al. | 2024 | Multi-head speculative |
| 5 | EAGLE — Li et al. | 2024 | Feature-based drafting |
| 6 | SpecInfer | 2024 | Multi-model speculative |
| 7 | vLLM speculative decoding docs | 2024–2025 | Production baseline |
| 8 | Draft & Verify | 2024 | Alternative framing |
| 9 | Lookahead Decoding | 2024 | Parallel drafting |
| 10 | Self-Speculative Decoding | 2024 | Same-model draft/target |
| 11 | Cascade Speculative Decoding | 2024 | Multi-stage |
| 12 | Blockwise Parallel Decoding | 2023 | Parallelism |
| 13 | Prompt Lookup Decoding | 2023 | Training-free speedup |
| 14 | Adaptive Speculative Decoding | 2024 | Input-dependent acceptance |
| 15 | Online Speculative Decoding | 2024 | Distribution shift |
| 16 | Learned Drafter Models | 2024–2025 | Small LM drafts |
| 17 | Acceptance Rate Theory for SpecDec | 2024 | Analysis tools |
| 18 | TensorRT-LLM speculative mode | 2025 | Systems baseline |
| 19 | Multi-Candidate Speculative Decoding | arXiv:2510.20064 | Heterogeneous candidates |
| 20 | Best-of-N vs. speculative tradeoffs | 2024 | Compute allocation connection to P1 |

---

## Cross-Problem Shared Infrastructure

Build once, reuse across P1–P4 (highest overlap):

| Module | Used by | Description |
|---|---|---|
| `features.py` — frozen LM hidden extraction | P1, P2, P3, P4 | Last-token / mean-pool hiddens |
| `eval_harness.py` — MATH/AIME/GPQA grading | P1–P4, P8 | `sympy` answer extraction |
| `verifier.py` — PRM/GenRM scoring | P1, P2, P4 | Verifier labels for trust head |
| `compute_accounting.py` | P1, P3, P4, P9 | Token/sample FLOPs logging |
| `plots.py` — Pareto curves | All | Accuracy vs. compute figures |

**Recommended staging (from strategy doc):**

1. Weeks 1–4: P1 baselines (Snell/Damani reproduction)
2. Weeks 5–10: P1 controller + trust head
3. Parallel: P4 calibration study on same generations (near-zero marginal cost)
4. Branch to P2 if false-positive analysis is strong

---

## Citation Caveats

1. **Re-FORC** (arXiv:2511.02130) is a NeurIPS 2025 Efficient Reasoning **workshop** paper — treat metrics as indicative until archival.
2. **Agrawal et al. "Cut the Overcredit"** is under review (OpenReview:7mVZy4mI1J) — for peer-reviewed ceiling claims, cite **Stroebl et al.** (arXiv:2411.17501).
3. **Snell "14× larger model"** result is conditional on non-trivial base success rates — do not over-claim.
4. Several **2026 arXiv** papers are preprints — verify venue/status before camera-ready. **Update (2026-06-29):** the P7 anchor **BiasScope (arXiv:2602.09383) is confirmed real** (ICLR 2026) — the earlier "possibly speculative" flag was wrong. The §P4–§P7 inline 20-paper maps that used to live here contained several mis-cites (e.g. self-enhancement bias is arXiv:2404.13076, not 2406.07791; the "Jiang et al. uncertainty" paper is Xiong et al. arXiv:2306.13063) and have been **superseded by the web-verified lists in `common/research/p04–p07-research.md`** — treat those as authoritative.
5. **Memori / A-Mem** numbers come from project reports/preprints — reproduce on LoCoMo locally.

---

## Next Actions

### P1 (in progress)
- [x] Research foundation + 20-paper map in `common/research/p01-research.md`
- [ ] Implement `src/features.py` + MATH-500 loader
- [ ] Reproduce best-of-N Pareto on Qwen2.5-1.5B (smoke → full)
- [ ] Build verifier label pipeline for `target_trust`
- [ ] Replace `decide_budget()` heuristic with trust-gated policy
- [ ] Run difficulty-only ablation
- [ ] Document reproduced numbers in `p01-verifier-aware-controller/run.md`

### P2 (started)
- [x] Research foundation + 20-paper map in `common/research/p02-research.md`
- [x] Phase 1 ceiling scaffold (`phase1.py`, `ceiling.py`, `verifier.py`, `calibration.py`)
- [ ] Reproduce Stroebl ceiling curves on real MATH-500 generations
- [ ] Phase 2: frozen hidden extraction + verifier FP labels
- [ ] Phase 3: train FPRBoundedVerifier vs. standard BCE ablation
- [ ] Export `precision@τ` trust labels for P1 integration
- [ ] Document reproduced numbers in `p02-calibrated-fp-verifiers/run.md`

### P3 (started)
- [x] Research foundation + 20-paper map in `common/research/p02-research.md` + `p03-overthinking-latent-controller/research.md`
- [x] `OverthinkController` scaffold (`stop_head`, `should_stop`, BCE train loop)
- [ ] Phase 1: reproduce LLMThinkBench-style token/accuracy curves + NoWait/fixed/L1 baselines
- [ ] Phase 2: mid-CoT hidden-state extraction + safe-stop labels
- [ ] Phase 3: train probe on real hiddens; difficulty-only ablation
- [ ] Phase 4: generation-loop integration (HF stopping criteria)
- [ ] Phase 5: full Pareto eval vs. Re-FORC / ThinkPrune references
- [ ] Document reproduced numbers in `p03-overthinking-latent-controller/run.md`

### P4–P7 (research foundations complete, 2026-06-29)
- [x] Web-verified literature review + critical "are-we-on-track" verdict in `common/research/p04–p07-research.md`
- [x] Master-doc §P4–§P7 condensed to pointer + verdict; mis-cites corrected
- [x] Scaffold boilerplate (P1 math/best-of-N knobs) stripped from P5/P6/P7 configs/run.md/core docstrings
- [ ] **P4:** build budget-sweep + budget-conditioned recalibration + AURC/risk–coverage on P1 generations (cheap hedge — do in parallel with P1)
- [ ] **P5:** *pivot-or-park* — only if pursuing the end-to-end vLLM throughput study; rewrite `core.py` away from the trained-router placeholder
- [ ] **P6:** *pivot* — design the {token-count} × {absolute-position} 2×2 under certified-perfect retrieval; demote `ContextDegradationProbe` to a diagnostic
- [ ] **P7:** *pivot* — re-aim at causal validation in the code-judge domain; rewrite `core.py` to the LLM-perturbation + API-judge pipeline

---

*Last updated: 2026-06-29. Extend this file as baselines are reproduced and paper lists are validated against your local bibliography manager.*