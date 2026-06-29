# P5 Research Foundation — Batch-Aware MoE Expert Routing at Inference (Training-Free)

**Problem:** Can a training-free, batch-aware routing policy reduce the union of activated experts (and load imbalance) during *batched* MoE inference, improving wall-clock throughput/memory without accuracy loss?

**Compute:** 2–8×A100 burst / single A100 + offload (~100 GPU-hrs, $400–900)
**Venues:** MLSys / NeurIPS / ICLR (systems track)
**Project:** `p05-batch-moe-routing/`
**Parent index:** [`../../research.md`](../../research.md) §P5

---

### TL;DR Verdict

**PIVOT / NARROW — do not pursue the headline framing as a flagship; it is already scooped.** The exact problem ("union of activated experts grows toward total under batching; reduce it training-free for faster decode") is the literal subject of the verified anchor **Opportunistic Expert Activation (OEA, arXiv:2511.02237, Tri Dao et al., Nov 2025)** — and it was *already* solved in workload-agnostic, training-free form a year earlier by **Lynx (arXiv:2411.08982, Nov 2024)**. Worse for novelty, the white space OEA explicitly left open (end-to-end serving integration, batch-adaptive k₀, layer heterogeneity) is being closed *right now* by **SERE (ICLR 2026, arXiv:2602.07616)** — which integrates into vLLM with no core-pipeline changes — and **XShare (arXiv:2602.07265)**, which already has a vLLM upstreaming RFC (issue #35550). "Training-free with negligible accuracy loss" is **realistic** (Lynx ≤1.55×, OEA halves active experts, XShare −30% activation, all "no statistically significant accuracy loss"), so the *idea* is sound — that is precisely why four groups converged on it. The decisive obstacle is the one the task flags: **a credible end-to-end wall-clock harness vs. vLLM on solo compute is genuinely hard**, because (a) the MoEs worth testing do not fit in fp16 on one 40GB A100 (Mixtral ~90GB, Qwen3-30B ~60GB), forcing quantization/offload that *changes which regime is bottlenecked*; (b) custom routing breaks vLLM's fused grouped-GEMM / CUDA-graph fast path, so fewer experts can still be *slower*; (c) OEA itself dodged this by reporting **MoE-layer latency only, never end-to-end, never vs. vLLM**. Net: there is one defensible, un-taken slice (rigorous *end-to-end* throughput on open MoEs in the *offloading/single-GPU* regime, with batch-adaptive + layer-wise k₀), but it is a systems-engineering grind with a shrinking window and high scoop risk. **Recommendation: deprioritize behind P1/P3/P4; if pursued, reframe narrowly as "the end-to-end serving study OEA/Lynx never did," not as a new routing idea.**

---

### Problem Statement & Framing

> Can training-free, batch-aware routing reduce union-of-experts activation and load imbalance during batched MoE inference, improving wall-clock throughput/memory without accuracy loss?

**The core tension (well-posed, and now textbook):** per-token top-k routing keeps each token sparse, but the *batch* must materialize the **union** of every token's expert set. At batch B with E experts, top-k, the expected number of distinct activated experts approaches E quickly — so decode becomes **memory-bound on the number of *unique* experts loaded**, not on per-token FLOPs. MoE decode latency is therefore roughly linear in |union of experts|, and batching erodes the very sparsity MoE was built for. This is stated almost identically in OEA, Lynx, XShare, and SERE.

**The lever:** at inference you already have the router's gating logits. You can (i) *piggyback* low-ranked experts onto experts already loaded for other tokens in the batch (OEA), (ii) *re-route* a token's secondary experts to a similar primary expert that is already active (SERE), (iii) *remap/skip* low-importance experts batch-wide using only in-model signals (Lynx), or (iv) maximize total retained gating score under a batch-wide active-expert budget (XShare). All are **training-free** and trade a tiny gating-score / accuracy loss for a large reduction in unique experts.

**Target bar (for MLSys credibility):** real wall-clock throughput / tokens-per-second and peak memory vs. a **vLLM** serving baseline on ≥2 open MoEs, at matched accuracy on downstream tasks — *not* FLOP counts and *not* MoE-layer-only latency. This bar is exactly what the existing literature has mostly avoided, and exactly what is hardest solo.

**Scaffold reality check:** the current `p05-batch-moe-routing/src/core.py` implements the *wrong mechanism for the wrong measurement*. It (a) trains a **new** router MLP with MSE on a synthetic "expert distribution" — contradicting "training-free"; (b) `suggest_batch_mask` does top-k + threshold over that MLP's softmax, never touching a real MoE model's own router; (c) `estimate_speedup` returns `baseline_union / active_experts` — a **FLOP/count proxy**, precisely what the task says is insufficient. The config (`configs/default.yaml`) points at a **dense** model (`Qwen2.5-1.5B-Instruct`) with math/best-of-N knobs copied from the P1 test-time-compute scaffold — none of it is MoE-relevant. Treat `core.py` as an empty placeholder, not a viable mechanism. (Note: class is named `BatchAwareMoERouter`; the master doc/prompt call it `BatchMoERouter`.)

---

### Research Landscape (2024–2026)

The niche the compass doc bet on ("inference-time MoE routing without retraining") went from *fresh* to *crowded* in ~18 months. Three clusters now exist:

1. **Batch-aware activation reduction (the P5 core — now occupied):** Lynx (11/2024) → OEA (11/2025) → XShare, SERE (02/2026) → Expert Streaming (03/2026). All training-free, all reducing the per-batch expert union, all reporting negligible accuracy loss. This is no longer white space.

2. **Expert offloading / single-GPU serving (the regime where solo compute actually lives):** MoE-Infinity (01/2024), HOBBIT (11/2024), fMoE (02/2025), MoE-SpeQ (11/2025), OD-MoE (12/2025), Dynamic Expert Quantization (11/2025). These keep a subset of experts resident and predict/prefetch the rest — the regime where reducing the expert union most directly buys *real* speed, because each avoided unique expert is an avoided PCIe transfer.

3. **Routing / load-balancing foundations & serving systems:** Switch, GShard, BASE Layers, Expert Choice, ST-MoE (z-loss); vLLM/PagedAttention, DeepSpeed-MoE, Tutel, MegaBlocks; expert-parallelism (Wide-EP in vLLM/llm-d). These define the baselines you must beat and the kernels you must not break.

**Structural shift that matters for P5:** modern open MoEs are **fine-grained** (DeepSeekMoE, Qwen3-MoE 128 experts/top-8, gpt-oss 32–128 experts/top-4). Fine-grained + shared experts changes the union math — more experts but each cheaper, and shared experts always load — so batch-aware gains and the right baseline differ from the Mixtral-8×7B/top-2 era the original framing implicitly assumes.

---

### Deep Paper Reviews (15)

> "Verified" = I confirmed title + arXiv ID via web search/fetch this session.

**1. Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining** — Oncescu, Wu, Chung, R. Wu, Gopal, J. Wang, Tri Dao, Athiwaratkun. arXiv:2511.02237, Nov 2025. **Verified: YES (fetched).**
*The anchor and the direct precursor — this IS P5.* Two-phase, training-free: Phase 1 each token takes top-k₀ by router score; Phase 2 tokens "piggyback" additional lower-ranked experts already being loaded for other tokens in the batch, minimizing the unique-expert union. On Qwen3-30B/235B at B=16: 39% / 15% **MoE-layer** decode-latency reduction, no statistically significant accuracy loss; with k₀=3 the active-expert count is halved. **Borrow:** the piggyback mechanism, the "latency ≈ |unique experts|" model, the no-retrain framing. **Beat:** it reports **MoE-layer latency only — never end-to-end, never vs. vLLM**, only B=16, single-machine, and explicitly lists batch-adaptive k₀ and per-layer hyperparameters as *open problems*. That gap is the only credible P5 contribution left.

**2. Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection** — Gupta, Sinha, Gavrilovska, Iyer (Georgia Tech). arXiv:2411.08982, Nov 2024. **Verified: YES.**
*Predates OEA and already does the training-free batch-level version.* Run-time dynamic expert remapping using only in-model information; reduces total experts invoked per batch; up to **1.55× latency reduction** with negligible accuracy loss on code-gen and math reasoning. **Borrow:** "expert importance varies across tokens/phases" insight; their workload-agnostic remap. **Beat:** Lynx is a strong baseline you must cite as prior art — it weakens any "first training-free batch-aware router" claim to near zero. Reproduce it as a baseline, don't re-invent it.

**3. SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models** — (ICLR 2026; OpenReview 98IxaUQtMY). arXiv:2602.07616, Feb 2026. **Verified: YES.**
Input-aware reduction of active experts by re-routing tokens from secondary experts to their most-similar *primary* (already-active) expert; uses similarity to preserve critical experts and avoid capability loss; dynamic skipping by batch-level redundancy. **Crucially: model-agnostic and integrates into vLLM with no core-pipeline changes, with reported real speedups.** **Beat:** this is the closest competitor that *already crossed the vLLM-integration bar P5 set as its differentiator.* Its existence is the strongest argument to pivot.

**4. XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference** — Vankov, Ivkin, Ulrich, Song, Khetan, Karypis (Amazon). arXiv:2602.07265, Feb 2026. **Verified: YES.**
No-retrain; per batch maximizes total gating score of selected experts under a shared-expert budget; hierarchical, correlation-aware selection. Reports −30% expert activation under standard batching, up to **3× lower peak GPU load in expert-parallel** deployment, and **+14% throughput in speculative decoding**. **Has a vLLM upstreaming RFC (GitHub vllm-project/vllm #35550).** **Beat:** covers EP + spec-decode angles and is being productized — the niche is going from research to infrastructure. Borrow the "maximize retained gating mass under budget" objective as a clean baseline.

**5. Expert Streaming: Accelerating Low-Batch MoE Inference via Multi-chiplet Architecture and Dynamic Expert Trajectory Scheduling** — arXiv:2603.27624, Mar 2026. **Verified: YES (search).**
Low-batch regime, hardware/architecture co-design (multi-chiplet) + dynamic expert-trajectory scheduling. **Relevance:** shows the frontier is moving toward hardware co-design and the *low-batch* regime; mostly orthogonal to a pure-software solo project but signals the space's maturation. **Don't chase** — requires hardware modeling out of scope for one A100.

**6. MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading** — arXiv:2511.14102, Nov 2025. **Verified: YES (search).**
Combines speculative decoding with proactive expert prefetch + offload and quantization. **Relevance:** the speculative-decode × MoE intersection (ties to P10) and confirms prefetch/offload as the lever in memory-bound serving. **Borrow:** prefetch scheduling ideas if P5 is reframed into the offload regime.

**7. HOBBIT: A Mixed-Precision Expert Offloading System for Fast MoE Inference** — arXiv:2411.01433, Nov 2024. **Verified: YES.**
Dynamically replaces less-critical cache-miss experts with low-precision versions to cut expert-loading latency while preserving accuracy. **Relevance:** in the single-GPU offload regime, *fewer unique experts* (P5's lever) directly = fewer cache misses, so batch-aware routing composes naturally with HOBBIT-style caching. **Borrow:** mixed-precision fallback for "must-load but low-score" experts.

**8. fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving** — arXiv:2502.05370, Feb 2025. **Verified: YES.**
Semantic matching between historical prompts and the current input to raise expert cache-hit rate. **Relevance:** cache-hit maximization is the offload-regime sibling of union-minimization; the natural baseline if P5 pivots to "single-GPU offloaded MoE." **Borrow:** the cache-management evaluation methodology (hit-rate vs. latency).

**9. MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache** — arXiv:2401.14361, Jan 2024. **Verified: YES.**
Sparsity-aware expert caching for single personal/consumer GPUs serving >100GB MoEs via host-memory offload. **Relevance:** defines the realistic solo-compute deployment target and a concrete, reproducible offload baseline on one GPU. **Borrow:** their setup is the most honest harness for a solo author who lacks 8×A100.

**10. MoE-Inference-Bench: Performance Evaluation of MoE Large Language and Vision Models** — arXiv:2508.17467, SC'25 Workshops, Aug 2025. **Verified: YES.**
Systematic inference benchmarking across Mixtral/DeepSeek/Qwen/Phi/OLMoE (2B–70B), sweeping batch size, sequence length, OOM boundaries; evaluates quantization, intra/inter-expert pruning, speculative decoding, Fused MoE; all on 4×H100 with **vLLM**, reporting TTFT/throughput. **Borrow heavily:** this is the closest thing to an off-the-shelf harness + baseline protocol — adopt its metric set (TTFT, throughput, OOM frontier) verbatim so P5's numbers are comparable. **Beat:** it benchmarks existing techniques; it does not propose batch-aware routing, so it is a measurement ally, not a competitor.

**11. Mixtral of Experts** — Jiang et al. (Mistral AI). arXiv:2401.04088, Jan 2024. **Verified: YES.**
8 experts/layer, top-2; 47B total / 13B active; the canonical open MoE; vLLM integrates MegaBlocks CUDA kernels for it. **Relevance:** classic eval target, but **coarse-grained/top-2** — the *easy* case for union growth and **does not fit fp16 on a 40GB A100**. **Use** as one eval target but know the regime differs from fine-grained models.

**12. DeepSeekMoE: Towards Ultimate Expert Specialization** — Dai et al. arXiv:2401.06066, Jan 2024. **Verified: YES.**
Fine-grained expert segmentation (mN experts, activate mK) + isolated shared experts. **Relevance:** fine-grained + shared experts *changes the union problem* — more, cheaper experts and always-on shared experts shift where batch-aware gains come from. Any P5 result must control for fine- vs. coarse-grained, or it won't generalize across modern MoEs.

**13. Mixture-of-Experts with Expert Choice Routing** — Zhou et al. arXiv:2202.09368, 2022. **Verified: YES.**
Each expert picks its top-c tokens → perfect per-batch load balance by construction, no aux loss. **Relevance:** the conceptual ancestor of batch-level (vs. token-level) optimization; "optimize the batch, not the token" is exactly P5's framing. **Borrow:** the batch-as-unit objective. **Caveat:** Expert Choice is a *training-time* paradigm; P5 must achieve the analogous balance *post-hoc at inference*.

**14. Switch Transformers: Scaling to Trillion-Parameter Models with Simple and Efficient Sparsity** — Fedus, Zoph, Shazeer. arXiv:2101.03961, JMLR 2022. **Verified: YES.**
Top-1 routing, capacity factors, load-balancing aux loss — the foundation of routing + imbalance. **Relevance:** defines "load imbalance / stragglers," the second half of P5's problem; capacity-factor logic is the lever for the load-balance sub-claim. **Borrow:** capacity/drop framing for the imbalance metric.

**15. ST-MoE: Designing Stable and Transferable Sparse Expert Models (router z-loss)** — Zoph et al. arXiv:2202.08906, 2022. **Verified: YES.**
Introduces router z-loss for training stability. **Relevance:** establishes that router logits are fragile/peaky — important because P5 perturbs routing at inference; z-loss-trained routers may tolerate piggyback/re-route differently. **Borrow:** as background for "why training-free routing perturbation is safe-ish."

**Foundational / serving context (verified where noted; cite as background, do not reproduce):**

| Paper | ID / Venue | Verified | Role for P5 |
|---|---|---|---|
| vLLM: Efficient Memory Management (PagedAttention) | arXiv:2309.06180, SOSP 2023 | ID not re-checked this session (well-known) | **The serving baseline** P5 must beat end-to-end |
| gpt-oss-120b & gpt-oss-20b Model Card | arXiv:2508.10925, 2025 | YES | Eval target (32/128 experts, top-4, Apache-2.0) |
| GShard | arXiv:2006.16668, 2020/2021 | YES (search) | Large-scale MoE + capacity foundation |
| BASE Layers (Lewis et al.) | arXiv:2103.16716 (not re-verified) | NO (ID from memory) | Assignment-as-optimization load balancing |
| Soft MoE (Puigcerver et al.) | arXiv:2308.00951, **2023** (master doc says 2024) | partial | Differentiable routing contrast |
| DeepSpeed-MoE | arXiv:2201.05596 (not re-verified) | NO (ID from memory) | MoE serving system baseline |
| Tutel | arXiv:2206.03382 (not re-verified) | NO (ID from memory) | Adaptive MoE parallelism/kernels |
| MegaBlocks | arXiv:2211.15841 (not re-verified) | NO (ID from memory) | Block-sparse MoE kernels (used by vLLM/Mixtral) |
| MoE-Infinity / Wide-EP (vLLM+llm-d) | see #9 / Red Hat blog 2025 | partial | Single-GPU offload + expert-parallel context |

---

### Gaps & Novelty — What Does NOT Yet Exist

**Already exists (do not claim as novel):**
- Training-free batch-aware reduction of the expert union → **Lynx, OEA, XShare, SERE** (4 independent works).
- Piggybacking / re-routing secondary experts to active ones → **OEA, SERE**.
- Batch-wide gating-mass-budget selection → **XShare**.
- vLLM integration of batch-aware routing → **SERE (no core changes), XShare (RFC #35550)**.
- Single-GPU offload with sparsity-aware caching → **MoE-Infinity, HOBBIT, fMoE**.
- Systematic MoE inference benchmarking on vLLM → **MoE-Inference-Bench**.

**Genuine remaining white space (narrow, shrinking, defensible):**
1. **The end-to-end study OEA never did.** OEA reports MoE-*layer* latency at B=16 with no vLLM comparison. A rigorous **end-to-end tokens/sec + peak-memory + accuracy** sweep across batch sizes, on ≥2 open MoEs, vs. a real vLLM baseline, is *not yet published as such*. This is the one slice consistent with P5's stated MLSys bar — but note it is engineering, not a new idea, and SERE/XShare are racing into it.
2. **Batch-adaptive k₀ as an automatic policy.** OEA names "batch-size-dependent k₀ choice" an open problem. A cheap closed-loop controller (k₀ as a function of batch size, sequence position, and observed gating-entropy) is unclaimed — and is the *one place a lightweight learned/heuristic controller (your scaffold's intended spirit) could add value*.
3. **Per-layer heterogeneous routing budgets.** OEA uses uniform hyperparameters across layers and flags per-layer adaptation as open. Early vs. late layers have very different gating entropy; a layer-wise budget allocator is unclaimed.
4. **Union-minimization × offload composition with a measured cache-miss model.** Nobody has shown that batch-aware union reduction translates to *fewer PCIe expert transfers* in a HOBBIT/fMoE/MoE-Infinity offload setup with an end-to-end number. This is the most solo-compute-friendly framing (one A100 + host RAM) and the most honest fit for the budget.

**Honest assessment of the white space:** items 2–4 are *incremental refinements explicitly enumerated as future work by the precursor.* They are publishable only with strong execution and real wall-clock numbers, and the window is months, not years (Feb–Mar 2026 papers are already encroaching). Item 1+4 combined (end-to-end, offload regime, batch-adaptive + layer-wise) is the strongest single story but is a systems grind.

---

### Critical Assessment

**Well-posedness:** Strong. The problem is crisply defined, the bottleneck (latency ≈ |unique experts| under batching) is real and now well-characterized, and the lever (re-route/piggyback using existing gating logits) is concrete. No conceptual risk.

**Novelty vs. the precursor:** **Weak and deteriorating.** The headline claim is fully occupied by OEA, and Lynx predates it. Three more works (XShare, SERE, Expert Streaming) landed within ~4 months. "Batch-aware training-free MoE routing" as a *new contribution* is no longer available. Only the enumerated future-work refinements (batch-adaptive k₀, per-layer budgets) and the *missing end-to-end/offload measurement* remain — and those are being actively closed. **Scoop risk is the highest of any problem in this portfolio.**

**Is training-free + lossless realistic?** Yes — empirically. OEA halves active experts, Lynx hits 1.55×, XShare −30% activation, SERE preserves accuracy, all reporting negligible/no significant accuracy loss. The idea works; that is the problem (everyone found it).

**Feasibility — the wall-clock measurement risk (the decisive factor):** This is where P5 is genuinely hard for a solo author, and the compass doc already warned "avoid P5/P10 as first projects." Specifics:
- **Memory:** The MoEs worth testing don't fit in fp16 on one A100-40GB (Mixtral ~90GB; Qwen3-30B ~60GB; gpt-oss-20B ~40GB at MXFP4 only). You're forced into quantization or host-offload — which **moves the bottleneck** and means your "speedup" is measured in a different regime than OEA's (HBM-resident) numbers, complicating comparison.
- **Kernel fast-path breakage:** vLLM's MoE throughput comes from **fused grouped-GEMM + CUDA graphs + autotuned kernels** that assume static per-expert token counts. Injecting dynamic batch-aware routing can fall off the fast path and be **net slower despite touching fewer experts** — the classic "fewer FLOPs, worse wall-clock" trap. SERE's selling point is precisely that it integrates *without* breaking the pipeline; matching that is non-trivial engineering.
- **Why OEA reported layer-only latency:** almost certainly because end-to-end vs. a tuned baseline is hard and easy to lose. Reproducing that pitfall is the default failure mode.
- **Multi-GPU EP** (where XShare shows 3× peak-load wins) needs 2–8 GPUs you may not have on Colab.

**Net feasibility:** *Medium-low* as a flagship; *medium* if narrowed to the single-A100 offload regime with MoE-Infinity/HOBBIT-style baselines and end-to-end numbers, accepting that you are doing the measurement study, not inventing the mechanism.

**Verdict:** **PIVOT.** Either (a) reframe as a focused systems/benchmark contribution — "end-to-end batch-aware routing on open MoEs in the single-GPU offload regime, with batch-adaptive + per-layer k₀, vs. vLLM and vs. Lynx/OEA reproduced" — accepting incremental novelty and racing a closing window; or (b) **drop P5 to lowest priority** and let P1/P3/P4 (analysis/controller, single-A100, low scoop risk) carry the portfolio, as the compass ROI verdict already recommends.

---

### Recommended Implementation Phases

If pursued, restructure entirely — the current scaffold is not on the path. Tie each phase to concrete files.

**Phase 0 — Replace the scaffold's mechanism (Week 1).**
- Delete the MLP-router + MSE-training path in `src/core.py`; it contradicts "training-free." Rename to match the master doc or fix the master doc (currently `BatchAwareMoERouter` vs `BatchMoERouter`).
- New `core.py`: a *training-free* batch policy operating on a **real MoE model's own router logits** — implement (i) piggyback (OEA), (ii) similarity re-route (SERE), (iii) gating-mass-budget (XShare) as three switchable policies.
- Replace `estimate_speedup` (the `union/active` FLOP proxy) with a real timing hook. Fix `configs/default.yaml` (dense Qwen2.5-1.5B, math/best-of-N knobs) → MoE models + serving knobs.

**Phase 1 — Profiling + measurement harness (Weeks 1–3). THE critical phase.**
- Instrument a real MoE (start small: **Qwen1.5-MoE-A2.7B** or **OLMoE-1B-7B**, which fit on one A100) in HF/transformers; log per-layer, per-batch **|unique experts|** vs. batch size and sequence position. This reproduces the core "union grows with B" curve — the figure that motivates everything.
- Build the **end-to-end harness now**: tokens/sec, TTFT, peak memory, downstream accuracy. Adopt **MoE-Inference-Bench** (arXiv:2508.17467) metrics verbatim. Establish the **vLLM baseline** first; if you cannot beat or match vLLM's fused path, the project is dead — know this by Week 3.

**Phase 2 — Reproduce the precursors as baselines (Weeks 3–5).**
- Reproduce **OEA piggyback** and **Lynx remap** on the small MoE; confirm halved active experts / negligible accuracy loss. You cannot claim improvement without these on your hardware.
- Decide regime: **HBM-resident** (needs ≥Qwen3-30B, multi-GPU or quantized) vs. **single-GPU offload** (Mixtral/Qwen3-30B in host RAM + MoE-Infinity/HOBBIT-style cache). The offload regime is the recommended solo path — there, |union| reduction = fewer PCIe transfers = honest end-to-end wins.

**Phase 3 — The actual contribution (Weeks 5–8).**
- Implement **batch-adaptive k₀** (OEA's open problem): k₀ as a cheap function of batch size + gating entropy; close the loop online. This is where the scaffold's "lightweight controller" spirit legitimately fits.
- Implement **per-layer budgets** (OEA's other open problem): allocate the active-expert budget across layers by measured gating entropy.
- In the offload regime, **compose with a cache** and show union-minimization lowers cache-miss/PCIe traffic with an end-to-end number.

**Phase 4 — Evaluation vs. vLLM + accuracy (Weeks 8–10).**
- End-to-end throughput + peak memory across batch sizes vs. vLLM, on ≥2 MoEs; accuracy on MMLU/GSM8K/HumanEval (not the P1 math-only set).
- Ablations: piggyback vs. re-route vs. budget; fixed vs. adaptive k₀; uniform vs. per-layer; coarse (Mixtral) vs. fine-grained (DeepSeek/Qwen3-MoE).
- Statistical CIs on throughput (multiple runs; CUDA-graph warmup; report median + IQR). **Honest negatives:** batch sizes/regimes where it loses to vLLM's fused path.

**Phase 5 — Artifacts.** Figure: |union| vs. B (motivation). Figure: throughput–accuracy Pareto vs. vLLM/OEA/Lynx. Table: per-model end-to-end speedup + peak mem at matched accuracy. Figure: adaptive vs. fixed k₀.

---

### Compute Budget Estimate

| Item | Hardware | Est. GPU-hrs | Notes |
|---|---|---|---|
| Phase 1 profiling + harness (small MoE) | 1×A100-40GB / Colab | 15–25 | Qwen1.5-MoE / OLMoE fit natively |
| Phase 2 reproduce OEA + Lynx | 1×A100 (+host RAM offload) | 20–30 | Offload regime avoids multi-GPU |
| Phase 3 adaptive/per-layer policy | 1×A100 | 20–30 | Mostly inference loops |
| Phase 4 vLLM end-to-end sweep, ≥2 MoEs | 2×A100 burst (or 1×A100 offload) | 30–50 | EP experiments need 2–8 GPUs |
| **Total** | mixed | **~85–135** | Matches compass ~100 GPU-hrs / $400–900 |

**Caveat:** the compass "2–8×A100 burst" applies to the HBM-resident, EP regime (XShare-style). The **single-A100 + host-offload** path keeps it Colab-Pro/one-A100 feasible at the cost of measuring a different (offload) regime — which is a legitimate, honest reframing, not a shortcut.

---

### Citation Caveats / Unverified IDs

**Verified this session (title + ID confirmed via web):** OEA 2511.02237; Lynx 2411.08982; SERE 2602.07616 (ICLR 2026); XShare 2602.07265; Expert Streaming 2603.27624; MoE-SpeQ 2511.14102; HOBBIT 2411.01433; fMoE 2502.05370; MoE-Infinity 2401.14361; MoE-Inference-Bench 2508.17467; Mixtral 2401.04088; DeepSeekMoE 2401.06066; Expert Choice 2202.09368; Switch 2101.03961; GShard 2006.16668; ST-MoE/z-loss 2202.08906; gpt-oss model card 2508.10925.

**NOT re-verified this session — IDs from prior knowledge, confirm before camera-ready:** BASE Layers (arXiv:2103.16716), DeepSpeed-MoE (arXiv:2201.05596), Tutel (arXiv:2206.03382), MegaBlocks (arXiv:2211.15841), vLLM/PagedAttention (arXiv:2309.06180), Soft MoE (arXiv:2308.00951). These are all real, well-known works; only the exact arXiv numbers are unconfirmed here.

**Errors / fixes for the master `research.md` §P5 20-paper map:**
1. **Title fix:** #1 "Opportunistic MoE Activation at Inference" → actual title **"Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining"** (ID 2511.02237 correct; authors incl. Tri Dao). The framing-only title undersells that it solves P5 outright.
2. **Critical omissions:** the map is missing the **three most important direct competitors** — **Lynx (2411.08982)**, **XShare (2602.07265)**, **SERE (2602.07616, ICLR 2026)**. These are the actual scoop risks and must be added; the map reads as if OEA were the only precursor.
3. **Date error:** #12 "Soft MoE — Puigcerver — 2024" is **2023** (arXiv:2308.00951).
4. **Vague placeholders:** entries #16 "Load Balancing in MoE — analysis papers," #18 "Expert Parallelism strategies," #19 "Batch inference optimization surveys," #20 "Sparsity vs. batching tradeoffs in MoE" have no concrete citations — replace with the offload/serving cluster (MoE-Infinity, HOBBIT, fMoE, MoE-Inference-Bench, MoE-SpeQ) and Wide-EP (vLLM+llm-d).
5. **No fabricated IDs detected** in §P5 — unlike some other sections, all listed P5 entries are real works (just under-specified). The anchor ID 2511.02237 is genuine and correctly attributed.

**Scaffold caveats:** `core.py` mechanism (trained MLP router, MSE) and `configs/default.yaml` (dense model, math/BoN) are inherited from the P1 test-time-compute scaffold and are **not MoE-relevant**; `estimate_speedup` is a FLOP proxy, not wall-clock. Class name `BatchAwareMoERouter` ≠ master doc's `BatchMoERouter`.

---

### Next-Actions Checklist

- [ ] **Decision gate:** pivot vs. kill. Default recommendation = drop to lowest priority behind P1/P3/P4; pursue only if you specifically want an MLSys-track systems paper and accept the scoop window.
- [ ] If pursuing: rewrite `src/core.py` as a **training-free** policy over a real MoE's router logits (piggyback / re-route / gating-budget); delete MLP+MSE path.
- [ ] Fix `configs/` to MoE models (Qwen1.5-MoE-A2.7B / OLMoE first) + serving knobs.
- [ ] Build the **end-to-end harness + vLLM baseline by Week 3** (kill-criterion: cannot match vLLM fused path).
- [ ] Reproduce |union| vs. batch-size profiling figure on a small MoE.
- [ ] Reproduce **OEA** and **Lynx** as baselines on your hardware.
- [ ] Choose regime: single-A100 **offload** (recommended) vs. HBM-resident/EP.
- [ ] Implement the two un-taken slices: **batch-adaptive k₀** and **per-layer budgets**.
- [ ] Update master `research.md` §P5: add Lynx/XShare/SERE, fix OEA title, fix Soft MoE date, replace vague entries.
- [ ] Monitor arXiv weekly — this niche is adding a paper roughly every 1–2 months; re-check scoop status before any submission.

---

*Last updated: 2026-06-28. Source: `research.md` §P5 + web-verified 2024–2026 literature. Verdict: pivot/deprioritize — problem is sound but scooped (OEA/Lynx) and actively being closed (SERE/XShare); only remaining slice is the end-to-end/offload measurement study OEA never did, which is a high-effort systems grind on solo compute.*
