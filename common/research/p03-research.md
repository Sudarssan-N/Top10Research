# P3 Research Foundation — Latent Controller for Adaptive Overthinking Reduction

**Problem:** Can a per-query learned probe on hidden states decide reasoning depth / early termination, reducing token bloat on easy problems?

**Compute:** 1×A100 (~100–200 GPU-hrs)  
**Venues:** ACL / EMNLP / COLM  
**Project:** `p03-overthinking-latent-controller/`  
**Parent index:** [`../../research.md`](../../research.md) §P3

---

## Thesis Chain

1. **Long chain-of-thought (CoT) improves hard reasoning** (Wei et al. 2022; DeepSeek-R1; OpenAI o1) — RL-trained reasoning models default to verbose, self-reflective traces.
2. **Overthinking is real and measurable** (LLMThinkBench) — on basic math, reasoning models generate ~18× more tokens than standard models while sometimes achieving *lower* accuracy; the accuracy–verbosity curve is non-monotonic.
3. **Current efficiency methods are coarse** (NoWait, L1, ThinkPrune, s1, TokenSkip) — they suppress tokens, fix budgets, or retrain the base model; none use a **frozen hidden-state probe** for per-step, per-query stop decisions at generation time.
4. **Internal states encode progress** (Re-FORC; Quiet-STaR; Damani) — lightweight adapters on hiddens predict reward-vs-length curves and difficulty without full model fine-tuning.
5. **Gap → P3:** Train `OverthinkController` — a small BCE probe on mid-CoT hiddens predicting safe convergence — and hook it into generation for adaptive early termination.

---

## Key Gaps P3 Fills vs. Prior Work

| Prior approach | Limitation | P3 contribution |
|---|---|---|
| NoWait / TokenSkip | Surface token heuristics; no internal convergence signal | Hidden-state **stop probability** per reasoning step |
| L1 / Arora & Zanette | RL **retrains** the reasoning model for length control | **Inference-only** probe on frozen LM (~50M params) |
| ThinkPrune / R1-Compress | Training-time or post-hoc compression | **Online** early termination during generation |
| Re-FORC | Reward-curve forecasting (regression over length) | Binary **safe-stop** / convergence classification |
| Damani / Snell | Per-query sample/search budget, not within-trace stop | **Intra-trace** termination when probe fires |
| Fixed max tokens | Same cap for easy and hard problems | **Adaptive** depth keyed to per-query hidden signals |

---

## 20 Closest Papers

| # | Citation | Key finding (P3) | Implement for P3 | Tier |
|---|---|---|---|---|
| 1 | Srivastava et al. *Do LLMs Overthink Basic Math Reasoning? (LLMThinkBench)* — ACL 2026 Findings, arXiv:2507.04023 | ~18× token bloat on easy math; Overthinking Score; non-monotonic budget scaling | Token/accuracy Pareto + Overthinking Score eval harness | **P0** |
| 2 | Wang et al. *Wait, We Don't Need to "Wait"! (NoWait)* — EMNLP 2025 Findings (exp.), arXiv:2506.08343 | Banning "Wait"/"Hmm" tokens cuts CoT 27–51% without utility loss | Logit-bias / banned-token decode baseline | **P0** |
| 3 | Sui et al. *Stop Overthinking: A Survey on Efficient Reasoning* — TMLR 2025, arXiv:2503.16419 | Taxonomy of model/output/prompt efficiency methods | Related-work map; position P3 in output-based + probe class | **P2** |
| 4 | Aggarwal & Welleck *L1: Controlling How Long A Reasoning Model Thinks With RL* — COLM 2025, arXiv:2503.04697 | LCPO enforces user-specified CoT length; SRM short-reasoning variants | Length-conditioned prompt / L1-model baseline | **P0** |
| 5 | Hou et al. *ThinkPrune: Pruning Long CoT via RL* — arXiv:2504.01296 | Iterative RL length caps → half tokens, ~2% AIME drop | Reference RL-pruned model on same backbone | **P1** |
| 6 | Arora & Zanette *Training Language Models to Reason Efficiently* — NeurIPS 2025, arXiv:2502.04463 | RL incentivizes dynamic CoT length by task complexity | Compare inference-only probe vs. RL-efficient model | **P1** |
| 7 | Zabounidis et al. *Re-FORC: Adaptive Reward Prediction for Efficient CoT* — NeurIPS 2025 ER Workshop, arXiv:2511.02130 | Adapter predicts reward vs. future thinking tokens; 26% compute cut via early stop | Closest cousin — reproduce forecaster + compare to stop probe | **P0** |
| 8 | Damani et al. *Learning How Hard to Think* — arXiv:2410.04707 | Learned allocator −50% compute or +10% acc at fixed budget | Difficulty-only tertile baseline (prompt hidden → fixed cap) | **P1** |
| 9 | DeepSeek-AI et al. *DeepSeek-R1* — Nature 2025, arXiv:2501.12948 | RL elicits verification-heavy long CoT | Primary R1-Distill eval backbone | **P1** |
| 10 | OpenAI *Learning to Reason with LLMs (o1)* — System card, Sept 2024 | Scaled RL reasoning paradigm anchor | Citation only unless API budget for length sweep | **P2** |
| 11 | Zelikman et al. *Quiet-STaR* — arXiv:2403.09629 | Latent thought tokens at each step improve hard-token prediction | Mid-step hidden extraction precedent | **P2** |
| 12 | Zelikman et al. *STaR* — arXiv:2203.14465 | Bootstrapped rationales improve reasoning iteratively | Foundational CoT training context | **P2** |
| 13 | Wei et al. *Chain-of-Thought Prompting* — NeurIPS 2022, arXiv:2201.11903 | CoT unlocks multi-step reasoning in large LMs | Default eval prompting format | **P2** |
| 14 | Madaan et al. *Self-Refine* — arXiv:2303.17651 | Iterative self-feedback ~+20% but multiplies cost | Motivation for inference-cost analysis | **P2** |
| 15 | Muennighoff et al. *s1: Simple Test-Time Scaling* — arXiv:2501.19393 | Budget forcing extends/truncates thinking via "Wait" injection | Budget forcing baseline | **P1** |
| 16 | Wang et al. *Learning to Refine: Self-Refinement of Parallel Reasoning* — arXiv:2509.00084 *(proxy)* | Parallel refinement reduces wasted candidates | Cite with caveat — see below | **P2** |
| 17 | Xia et al. *TokenSkip: Controllable CoT Compression* — EMNLP 2025, arXiv:2502.12067 | Skips low-importance tokens; 40% cut, <0.4% GSM8K drop | Token-skip importance baseline | **P1** |
| 18 | Wang et al. *R1-Compress: Long CoT Compression via Chunk Search* — arXiv:2505.16838 *(proxy)* | Chunk compression ~20% tokens, ~0.6% MATH drop | Post-hoc compression baseline | **P1** |
| 19 | Abdin et al. *Phi-4 Technical Report* — arXiv:2412.08905; *Phi-4-reasoning* — arXiv:2504.21318 | Standard vs. reasoning variant token/accuracy gap | LLMThinkBench reproduction subset | **P1** |
| 20 | Snell et al. *Scaling LLM Test-Time Compute Optimally* — ICLR 2025, arXiv:2408.03314 | Optimal compute allocation varies by difficulty; 4×+ over best-of-N | Difficulty-conditioned stop policy; shared P1 infra | **P0** |

**P0 stack:** #1, #2, #4, #7, #20 — reproduce first.  
**P1 baselines:** #5, #6, #8, #9, #15, #17, #18, #19.

---

## Recommended Implementation Phases (Weeks 1–10)

| Phase | Weeks | Goal | Key outputs |
|---|---|---|---|
| **1 — Baselines** | 1–3 | Token–accuracy frontier without controller | NoWait, fixed max, s1 budget forcing, L1-length prompts; Pareto JSON |
| **2 — Features** | 3–5 | Mid-CoT hidden extraction + labels | `features.py`; `safe_stop` / counterfactual labels per step |
| **3 — Training** | 5–7 | Train `OverthinkController` on real hiddens | BCE probe; difficulty-only ablation |
| **4 — Integration** | 7–8 | Hook probe into `generate()` loop | Threshold + hysteresis policies; `min_steps` guard |
| **5 — Evaluation** | 8–10 | Full Pareto vs. baselines + Re-FORC comparison | MATH-500, GSM8K, AIME; easy/hard quartile breakdown |
| **6 — Paper** | 10 | Figures + failure analysis | LLMThinkBench repro fig; calibration plot |

---

## Shared Infrastructure (P1 / P2 / P3)

| Module | P1 | P2 | P3 | Notes |
|---|---|---|---|---|
| `features.py` — frozen LM hiddens | ✅ | ✅ | ✅ | P3: per-CoT-step hiddens |
| `eval_harness.py` — MATH/AIME/GPQA | ✅ | ✅ | ✅ | Shared grading |
| `compute_accounting.py` | ✅ | — | ✅ | Tokens/query, FLOPs |
| `plots.py` — Pareto curves | ✅ | ✅ | ✅ | Accuracy vs. tokens |
| `verifier.py` | ✅ | ✅ | — | Optional: answer check for safe-stop labels |
| Snell / Damani baselines | ✅ | — | ✅ | Difficulty allocation framing |
| Re-FORC forecaster recipe | ✅ | — | ✅ | Hidden probe training pattern |

**Parallel track:** Run P3 on same R1-Distill generations as P1 Phase 1 baselines — marginal cost is hidden caching + probe training only.

---

## Citation Caveats (Unresolved Titles)

| Strategy doc entry | Resolution |
|---|---|
| **L1 — Controllable Reasoning Budget (2024)** | → Aggarwal & Welleck, *L1: Controlling How Long A Reasoning Model Thinks With RL*, COLM 2025, arXiv:2503.04697 |
| **Arora & Zanette — Efficient Reasoning (2025)** | → *Training Language Models to Reason Efficiently*, NeurIPS 2025, arXiv:2502.04463 |
| **Less Is More: Learning to Refine for Efficient Reasoning (2025)** | **Not found** on arXiv; proxy: *Learning to Refine: Self-Refinement of Parallel Reasoning*, arXiv:2509.00084 |
| **Compression of Chain-of-Thought (2025)** | **Not found** as exact title; proxy: *R1-Compress*, arXiv:2505.16838 (alt: *Making Slow Thinking Faster*, arXiv:2508.03346) |
| **NoWait EMNLP 2025** | arXiv:2506.08343; venue per strategy doc — verify before camera-ready |
| **Re-FORC** | NeurIPS 2025 Efficient Reasoning **workshop** — indicative metrics only |
| **o1 System Card** | OpenAI blog/system card, not peer-reviewed |

---

## Scaffold Status

- [x] `OverthinkController` — `stop_prob` head, `should_stop()`, `min_steps` guard
- [x] BCE training loop on synthetic convergence labels
- [ ] Real mid-CoT hidden-state collection
- [ ] Safe-stop label pipeline (counterfactual truncation)
- [ ] Generation-loop integration
- [ ] NoWait / L1 / Re-FORC baselines
- [ ] Full Pareto results on MATH-500 / GSM8K / AIME

See `run.md` for execution commands.

*Last updated: 2026-06-26.*