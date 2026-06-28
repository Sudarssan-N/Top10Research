# Base Research & Literature Map — Top 10 Problem Statements

**Purpose:** Master index for all ten problems. Detailed per-problem research (20-paper maps, phases, thesis chains) lives in [`common/research/`](common/research/). This file keeps the landscape overview, cross-problem infrastructure, and next actions.

**Source of truth for problem definitions:** `compass_artifact_wf-8d14bf99-24a1-4aa2-93c0-65bc5148adc7_text_markdown.md`

**Status:** Living document. **P1**, **P2**, and **P3** research foundations complete; P4–P10 follow the same structure.

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

### Problem Statement

> Does increasing reasoning budget help or hurt calibration, and can a lightweight probe fix it?

### Implementation Notes (Scaffold)

- `CalibrationProbe` in `p04-calibration-probe/src/core.py`
- `compute_ece()` already in P1 `evaluate.py` — share across projects
- Sweep budgets on 3–4 open reasoning models; train recalibration probe

### 20 Closest Papers for P4

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | Over-Reasoning Impairs Confidence Calibration | arXiv:2508.15050 | Direct motivation |
| 2 | Jiang et al. — Can LLMs Express Their Uncertainty? | 2023 | Verbalized confidence |
| 3 | Kuhn et al. — Semantic Uncertainty | 2023 | Uncertainty for long outputs |
| 4 | Lin et al. — Teaching Models to Express Uncertainty in Words | 2022 | Uncertainty language |
| 5 | Desai & Durrett — Calibration of Pre-trained Transformers | 2020 | NLP calibration baseline |
| 6 | Guo et al. — On Calibration of Modern Neural Networks | ICML 2017 | Temperature scaling |
| 7 | Mielke et al. — Reducing Calibration Error | 2022 | Calibration methods |
| 8 | Nixon et al. — Measuring Calibration | 2019 | ECE metrics |
| 9 | Kadavath et al. — Language Models (Mostly) Know What They Know | 2022 | P(True) calibration |
| 10 | Xiong et al. — Approximate Nearest Neighbor Calibration | 2024 | LLM calibration |
| 11 | Snell et al. — test-time compute scaling | arXiv:2408.03314 | Budget-compute axis |
| 12 | LLMThinkBench | arXiv:2507.04023 | Reasoning length vs. quality |
| 13 | DeepSeek-R1 / reasoning model reports | 2025 | Overconfidence in reasoning models |
| 14 | Conformal Prediction for LLMs | 2024 | Distribution-free calibration |
| 15 | Selective Prediction surveys | 2023 | Reject option / abstention |
| 16 | Process Reward Models — calibration at step level | ICLR 2024 | Step-level confidence |
| 17 | Gao et al. — Reward Model Overoptimization | 2023 | Verifier miscalibration |
| 18 | Stroebl et al. — verifier imperfection | arXiv:2411.17501 | FP/FN and selection confidence |
| 19 | Tian et al. — Just Ask for Calibration | 2023 | Prompting for calibration |
| 20 | Zhao et al. — Calibrate Before Use | 2021 | Few-shot calibration effects |

---

## P5 — Batch-Aware MoE Expert Routing at Inference

### Problem Statement

> Can training-free batch-aware routing reduce union-of-experts activation and load imbalance during batched MoE inference?

### Implementation Notes (Scaffold)

- `BatchMoERouter` in `p05-batch-moe-routing/src/core.py`
- Profile expert activation patterns; opportunistic subset selection per batch
- Rigorous wall-clock vs. vLLM baselines required for MLSys credibility

### 20 Closest Papers for P5

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | Opportunistic MoE Activation at Inference | arXiv:2511.02237 | Direct precursor — training-free routing |
| 2 | Switch Transformers — Fedus et al. | JMLR 2022 | MoE routing foundation |
| 3 | GShard — Lepikhin et al. | 2021 | Large-scale MoE |
| 4 | Mixtral of Experts | 2024 | Open MoE reference model |
| 5 | DeepSeek-MoE | 2024 | Fine-grained experts |
| 6 | Qwen-MoE Technical Report | 2025 | Open MoE eval target |
| 7 | BASE Layers — Lewis et al. | 2021 | Load balancing |
| 8 | Expert Choice Routing | 2022 | Alternative routing paradigm |
| 9 | Tutel — Microsoft | 2022 | MoE serving system |
| 10 | DeepSpeed-MoE — Rajbhandari et al. | 2022 | MoE training/serving |
| 11 | MegaBlocks — Gale et al. | 2023 | Efficient MoE kernels |
| 12 | Soft MoE — Puigcerver et al. | 2024 | Differentiable routing |
| 13 | Orca — Yu et al. | OSDI 2022 | Continuous batching |
| 14 | vLLM — PagedAttention | SOSP 2023 | Inference serving baseline |
| 15 | Router Z-Loss — Zoph et al. | 2022 | Router stability |
| 16 | Load Balancing in MoE — analysis papers | 2023–2025 | Imbalance characterization |
| 17 | gpt-oss / open MoE releases | 2025 | Evaluation targets |
| 18 | Expert Parallelism strategies | 2024 | Multi-GPU MoE inference |
| 19 | Batch inference optimization surveys | 2024 | Systems context |
| 20 | Sparsity vs. batching tradeoffs in MoE | 2025 | Core P5 tension |

---

## P6 — Context-Length-Induced Degradation Under Perfect Retrieval

### Problem Statement

> Why do models degrade as context grows even when retrieval is perfect, and can simple mitigations (reorder, calibrate) help?

### Implementation Notes (Scaffold)

- `ContextDegradationProbe` + reorder logic in `p06-long-context-degradation/src/core.py`
- Controlled injection experiments isolating retrieval from length

### 20 Closest Papers for P6

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | Context Length Alone Hurts Despite Perfect Retrieval | arXiv:2510.05381 | Primary motivation |
| 2 | Lost in the Middle — Liu et al. | TACL 2024 | U-shaped performance |
| 3 | RULER — Hsieh et al. | 2024 | Long-context eval suite |
| 4 | Needle-in-a-Haystack evaluations | 2023–2024 | Position sensitivity |
| 5 | Distraction-Aware Retrieval | arXiv:2509.21865 | Mitigation adjacent |
| 6 | Attention Calibration for Long Context | arXiv:2406.16008 | Calibration fix |
| 7 | LongBench — Bai et al. | 2023 | Benchmark suite |
| 8 | InfiniteBench | 2024 | Extreme length eval |
| 9 | NIAH variants (RULER, BABILong) | 2024–2025 | Controlled probes |
| 10 | Retrieval-Augmented Generation surveys | 2024 | RAG context |
| 11 | YaRN — Peng et al. | 2023 | RoPE extension |
| 12 | LongLoRA — Chen et al. | 2023 | Efficient long fine-tune |
| 13 | StreamingLLM — Xiao et al. | 2023 | Attention sink phenomenon |
| 14 | H2O — Heavy Hitter Oracle | 2023 | KV / attention sparsity |
| 15 | Landmark Attention | 2023 | Long-context architecture |
| 16 | Passage Re-ranking for RAG | 2023 | Retrieval quality (control) |
| 17 | FlashAttention-2 — Dao | 2023 | Long-seq efficiency (enables experiments) |
| 18 | Mechanistic interpretability of attention dilution | 2024–2025 | Mechanism diagnosis |
| 19 | Gemma / Llama long-context reports | 2024–2025 | Open models for replication |
| 20 | "Lost in the Middle" follow-ups / mitigations | 2024 | Reordering strategies |

---

## P7 — Automated Discovery of Novel Biases in LLM-as-Judge

### Problem Statement

> Can we automatically discover (not just catalogue) evaluation biases in LLM judges with causal validation?

### Implementation Notes (Scaffold)

- Contrastive perturbation pipeline in `p07-bias-discovery-llm-judge/src/core.py`
- Validate on MT-Bench / CodeJudgeBench; quantify FPR shifts under perturbation

### 20 Closest Papers for P7

| # | Paper | ID / Venue | Relevance |
|---|---|---|---|
| 1 | BiasScope — Automated Bias Discovery | arXiv:2602.09383 | Direct precursor |
| 2 | Judging LLM-as-a-Judge (MT-Bench) — Zheng et al. | 2023 | Judge foundation |
| 3 | LLM Evaluators Recognize and Favor Their Own Generations | arXiv:2406.07791 | Self-enhancement bias |
| 4 | Survey of LLM-as-a-Judge | arXiv:2411.16594 | Bias taxonomy |
| 5 | Position Bias in LLM Evaluators | 2024 | Known bias to beat |
| 6 | Verbosity Bias in LLM Judges | 2024 | Length bias |
| 7 | Authority Bias — model name effects | 2024 | Metadata bias |
| 8 | CodeJudgeBench | 2025 | Code evaluation bias testbed |
| 9 | JudgeBench | 2024 | Judge reliability benchmark |
| 10 | G-Eval — Liu et al. | 2023 | Chain-of-thought judging |
| 11 | Prometheus 2 — Kim et al. | 2024 | Open judge models |
| 12 | AlpacaEval 2 — Dubois et al. | 2024 | Automated eval pipeline |
| 13 | Chatbot Arena methodology — Chiang et al. | 2024 | Human-judge alignment |
| 14 | Constitutional AI — Bai et al. | 2022 | RLAIF judging |
| 15 | Pairwise vs. Pointwise Evaluation | 2024 | Experimental design |
| 16 | CALM — Calibrating LLM Judges | 2024 | Judge calibration |
| 17 | Reward Hacking in LLM Evaluation | 2024 | Gaming metrics |
| 18 | Causal Inference for NLP — Keith et al. | 2020 | Causal validation methods |
| 19 | CheckEval — checklist evaluation | 2024 | Structured judging |
| 20 | Multi-attribute LLM judging | 2024 | Factorized bias discovery |

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
4. Several **2026 arXiv** papers (e.g., arXiv:2603.15377, arXiv:2604.05417) are preprints — verify venue/status before camera-ready.
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

---

*Last updated: 2026-06-26. Extend this file as baselines are reproduced and paper lists are validated against your local bibliography manager.*