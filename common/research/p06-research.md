# P6 Research Foundation — Context-Length-Induced Degradation Under Perfect Retrieval

**Problem:** Why do LLMs degrade as context length grows *even when the needed information is retrieved perfectly* (i.e., distractor length, not retrieval failure, is the cause), and can simple, training-free mitigations recover accuracy?

**Compute:** 1×A100-40/80GB or Colab (~80–120 GPU-hrs, ~$150–300)
**Venues:** ACL / EMNLP / COLM / ICLR (analysis track)
**Project:** `p06-long-context-degradation/`
**Parent index:** [`../../research.md`](../../research.md) §P6

---

### TL;DR Verdict

**PIVOT (sharpen), do not proceed as scaffolded, do not kill.** The problem as written in the scaffold — "diagnose that length hurts under perfect retrieval, then apply simple reorder/calibration fixes" — is **already substantially solved in the published 2024–2026 literature, and by multiple groups.** The named anchor, *Context Length Alone Hurts LLM Performance Despite Perfect Retrieval* (arXiv:2510.05381, **verified**), already (a) isolates pure length from retrieval via whitespace replacement **and** attention masking, (b) shows 13.9–85% degradation, and (c) ships a training-free mitigation (recite-evidence-before-solving). Both mitigations P6 proposes are also already published: **attention re-calibration** is *Found in the Middle* (arXiv:2406.16008, verified, +15pp) and PINE (arXiv:2407.01100, verified); **reordering** is the standard Lost-in-the-Middle remedy. Running P6 as scaffolded = replication + a marginal third mitigation → **high incrementalism/scoop risk, likely reject.**

**But there is one specific, defensible white space the anchor explicitly leaves open: the mechanism.** The anchor's own masking experiment is self-undercutting in a productive way — if you mask the irrelevant tokens so the model attends *only* to evidence+question, you have **removed attention dilution from the softmax**, yet degradation persists. That rules out the field's dominant mechanistic story (softmax entropy/attention-fading, arXiv:2506.16640, 2602.15028) and points at a *positional/representational* cause (RoPE long-term decay, hidden-state norm growth). **Nobody has cleanly run the orthogonalizing 2×2: {few vs. many tokens in attention} × {relevant span at small vs. large absolute position}, with perfect retrieval held fixed.** That causal disentanglement — plus a *mechanism-targeted* training-free fix (position re-mapping / attention-temperature recalibration) benchmarked head-to-head against recitation and Found-in-the-Middle — is the contribution worth pursuing. Novelty: **medium and conditional on the mechanism framing**; feasibility: **high** (8B open models, RULER/BABILong, attention hooks all fit one A100/Colab); biggest risk: **incrementalism vs. the anchor and a possible v2 from the same lab (Hao Peng's group is actively on this line).** Verdict: pivot from "diagnose + mitigate" to **"resolve WHICH mechanism, then beat recitation with a fix derived from it."**

---

### Problem Statement & Framing

> Why do LLMs degrade as context length grows even when the needed information is retrieved perfectly, and can simple, training-free mitigations recover accuracy?

**Core claim (original scaffold):** Retrieval-isolating benchmarks overestimate long-context progress; if you hold retrieval perfect and only grow the *amount* of (irrelevant) context, performance still drops — implicating an internal mechanism (attention dilution / position bias) rather than retrieval failure. A model-agnostic, training-free fix (reorder relevant content, recalibrate attention) should recover much of the loss.

**Why this needs sharpening:** As of mid-2026 the *existence* claim is settled (anchor 2510.05381; reinforced by 2601.15300 "critical threshold," 2602.15028 "long context, less focus"), and the *generic* training-free fixes are published (2406.16008, 2407.01100, recitation). So the original framing is no longer a research question — it is a reproduction.

**Reframed core claim (recommended):** The literature offers two competing mechanisms for length-induced degradation under perfect retrieval — **(M1) attention dilution** (softmax denominator/entropy grows ~log n, flattening attention; 2506.16640, 2602.15028) and **(M2) positional/representational effects** (RoPE long-term decay at large absolute positions; hidden-state norm drift; 2405.14591, 2407.01100). The anchor's masking result is *inconsistent with M1 alone*. P6's contribution is a **controlled causal study that orthogonalizes token-count from absolute-position (and from distraction), pins the dominant mechanism, and derives a training-free intervention that targets it and beats recitation/Found-in-the-Middle at matched compute.**

**Target bar:** (1) A 2×2(×distractor) controlled benchmark where retrieval is provably perfect (evidence extractable verbatim) that shows degradation tracks **position more than count** (or vice-versa) with bootstrap CIs on ≥3 open models (Llama-3.1-8B, Qwen2.5/3-7B, Gemma-2-9B). (2) A training-free fix that recovers ≥ the recitation baseline's accuracy at **lower token overhead**, with an ablation isolating *why* it works. Anything less than (1)+(2) is incremental.

**Venues:** ACL / EMNLP / COLM (analysis); ICLR if the mechanism result is strong.
**Compute:** 1×A100, ~80–120 GPU-hrs.

---

### Research Landscape 2024–2026

The space has three clusters, all crowded:

**1. Diagnosis that length-alone hurts (the "existence" question — DONE).**
- Anchor 2510.05381 (Oct 2025): pure length hurts despite perfect retrieval, even under masking; 13.9–85% drop.
- 2601.15300 (Jan 2026): "critical threshold" — Qwen2.5-7B collapses (F1 0.55→0.30) at 40–50% of max context, using *natural* (un-padded) lengths as causal evidence.
- 2602.15028 (2026): "long context, less focus" — degradation in personalization/privacy with length, theoretically attributed to attention dilution; ships PAPerBench.
- Foundational position effect: Lost in the Middle (TACL 2024) U-shape; RULER (COLM 2024) and BABILong (NeurIPS 2024) show near-perfect vanilla NIAH but large drops on harder/longer tasks → "effective context ≪ claimed context."

**2. Mechanism (the OPEN question — partially addressed, contested).**
- **Attention dilution / softmax entropy:** 2506.16640 (α-entmax, ICLR 2026) and 2602.15028 formalize that softmax entropy grows ~log n, flattening attention ("attention fading"); these motivate architectural fixes (sparse/entmax attention) that require *training*.
- **Positional / RoPE:** "Base of RoPE Bounds Context Length" (2405.14591, NeurIPS 2024) — RoPE long-term decay; PINE (2407.01100) attributes position bias mechanistically to causal attention + relative position encodings.
- **Mech-interp tooling:** Stream / Sparse Tracing (2510.19875) makes per-head attention analysis tractable at 100k+ tokens (O(T log T)) — directly usable as P6's instrumentation.
- **Gap:** none of these *orthogonalize* count vs. absolute position under guaranteed-perfect retrieval; the anchor's masking experiment hints the answer is M2, but does not test it.

**3. Mitigations (mostly DONE; this is where scoop risk is highest).**
- **Attention re-calibration (training-free):** Found in the Middle (2406.16008) subtracts positional bias from attention, +15pp RAG; PINE (2407.01100) bidirectional inter-document attention to erase order bias — both training-free.
- **Recitation (training-free):** anchor 2510.05381 itself.
- **Reordering:** standard Lost-in-the-Middle remedy (put evidence at the ends).
- **Retrieval-side:** LDAR (2509.21865) learns distraction-aware retrieval (but it's a *trained* retriever, not training-free internals); END (2502.18915) early-drops noisy context.
- **Training-side denoising:** CDT (2510.05862) trains attention onto critical tokens via Integrated-Gradient noise scores.
- **Efficiency-adjacent (mechanism context, not mitigation targets):** StreamingLLM attention sinks (ICLR 2024), H2O heavy-hitter KV (NeurIPS 2023) — explain *where* attention mass concentrates, useful priors for M1/M2.

**Net:** the only uncrowded corridor is **mechanism-resolution + a fix engineered from the resolved mechanism**, evaluated against the three existing training-free fixes.

---

### Deep Paper Reviews (15)

Legend: **Verified?** = directly confirmed via web (arXiv/venue) during this review.

**1. Context Length Alone Hurts LLM Performance Despite Perfect Retrieval — Du, Tian, Ronanki, Rongali, Bodapati, Galstyan, Wells, Schwartz, Huerta, Peng. arXiv:2510.05381 (Oct 2025). Verified: YES.**
The anchor. Across 5 open/closed LLMs (Llama-3.1-8B, Mistral-7B-v0.3, GPT-4o, Claude-3.5-Sonnet, Gemini-2.0) on math/QA/coding (RULER + synthetic), performance drops 13.9–85% with length even when (a) irrelevant tokens are replaced with whitespace and (b) irrelevant tokens are *masked* so the model attends only to evidence+question. Mitigation = recite evidence before solving (+~4% RULER QA, +31% Mistral GSM8K). **Relevance:** this *is* P6's stated problem, already executed. **Borrow:** the whitespace/masking isolation protocol, the model set, recitation as a baseline. **Beat:** it never resolves the mechanism and never disentangles absolute position from token count under masking — the exact gap to attack; and recitation's gains are small and add tokens.

**2. Lost in the Middle: How Language Models Use Long Contexts — Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. TACL 2024 (arXiv:2307.03172). Verified: YES (TACL anthology).**
Canonical U-shaped position bias on multi-doc QA and key-value retrieval: accuracy highest when evidence is at the start/end, worst in the middle. **Relevance:** the positional-effect prior (M2) and the origin of "reordering" mitigations. **Borrow:** position-sweep evaluation design; the reorder-to-ends baseline. **Beat:** it studies *position at fixed length*; P6 must additionally vary *length at fixed position* to separate the two.

**3. RULER: What's the Real Context Size of Your Long-Context LMs? — Hsieh, Sun, et al. (NVIDIA). arXiv:2404.06654, COLM 2024. Verified: YES.**
Synthetic, length-configurable benchmark; 13 tasks/4 categories (retrieval, multi-hop, aggregation, QA). Of 17 models claiming ≥32K, ~half fail at that length; near-perfect vanilla NIAH but big drops on harder tasks. **Relevance:** the standard controlled long-context harness; anchor uses it. **Borrow:** RULER as the backbone generator for the controlled 2×2. **Beat:** RULER mixes retrieval difficulty with length; P6 needs a *perfect-retrieval-guaranteed* variant.

**4. Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization — Hsieh et al. (Snorkel/CMU). arXiv:2406.16008 (2024). Verified: YES.**
Training-free, inference-time calibration that *subtracts the model's positional attention bias* from attention weights, recovering up to **15pp** RAG accuracy. **Relevance:** this is essentially P6's "attention re-calibration" mitigation, already published and stronger than recitation. **Borrow:** the bias-estimation/subtraction technique as both a baseline and a starting point for a mechanism-targeted variant. **Beat:** it targets U-shaped *position* bias, not pure *length*; under the anchor's masking setup (dilution removed) its calibration may not be the right lever — test whether it helps when count, not position, is the manipulated axis.

**5. Eliminating Position Bias of Language Models: A Mechanistic Approach (PINE) — Wang, Zhang, Li, Huang, Han, Ji, Kakade, Peng, Ji. arXiv:2407.01100 (2024). Verified: YES.**
Attributes position bias mechanistically to causal attention + relative position encodings; PINE makes inter-document attention bidirectional and re-orders documents by attention value — training-free, zero-shot. **Relevance:** the mechanistic + training-free precedent, and **shares author Hao Peng with the anchor** — i.e., the same lab owns both the diagnosis and the position-bias fix. **Borrow:** the mechanistic framing and the bidirectional-attention intervention. **Beat:** PINE removes *order* bias among documents; it does not address degradation from raw token *count* under fixed order — P6's axis.

**6. BABILong: Testing the Limits of LLMs with Long-Context Reasoning-in-a-Haystack — Kuratov, Bulatov, et al. arXiv:2406.10149, NeurIPS 2024 D&B. Verified: YES.**
20 bAbI reasoning tasks embedded in long distractor text; even 128K-claimed models degrade beyond ~10% of capacity; RAG gives no lift on reasoning. **Relevance:** reasoning-flavored controlled NIAH; good secondary benchmark for the count axis. **Borrow:** reasoning-in-haystack construction with controllable filler. **Beat:** filler is natural text (distraction confound); P6's whitespace/masked variants remove that confound.

**7. Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval (LDAR) — Shim, Kim, Cho, Lee. arXiv:2509.21865 (Sep 2025, rev. Feb 2026). Verified: YES.**
Learns a retriever that accounts for the LLM's susceptibility to distraction, beating full-long-context at lower tokens. **Relevance:** retrieval-side mitigation of the same degradation. **Borrow:** the "distraction amplifies with capacity" framing; token-efficiency metric. **Beat:** it's a *trained* retriever and operates outside the model; P6 is training-free and internal — orthogonal, so cite as complementary, not competing.

**8. How Is LLM Reasoning Distracted by Irrelevant Context? (GSM-DC) — Yang, Huang, Zhang, Surdeanu, Wang, Pan. arXiv:2505.18761, EMNLP 2025. Verified: YES.**
Symbolic reasoning graphs with precisely injected distractors; LLMs are sensitive to irrelevant context in both path selection and arithmetic; training with strong distractors + PRM tree search helps. **Relevance:** the gold standard for *controlled distractor injection*. **Borrow:** the controlled-injection methodology and reproducible graph generator. **Beat:** GSM-DC studies *distraction*; P6's novelty is the regime where distraction is *removed* (whitespace/masked) and only length remains.

**9. StreamingLLM: Efficient Streaming LMs with Attention Sinks — Xiao, Tian, et al. arXiv:2309.17453, ICLR 2024. Verified: YES.**
Initial tokens act as "attention sinks" absorbing excess attention mass; keeping them enables stable streaming to 4M tokens. **Relevance:** explains where attention mass goes as n grows — a key prior for the dilution (M1) story and for any attention-temperature fix. **Borrow:** attention-sink diagnostics as a mechanism probe. **Beat:** it's an efficiency method, not an accuracy-under-perfect-retrieval study.

**10. H2O: Heavy-Hitter Oracle for Efficient Generative Inference — Zhang, Sheng, et al. arXiv:2306.14048, NeurIPS 2023. Verified: YES.**
A small set of "heavy-hitter" tokens carries most attention value; evicting the rest preserves quality. **Relevance:** evidence that attention is highly concentrated, informing whether dilution (M1) is plausible once distractors are masked. **Borrow:** heavy-hitter attribution as a mechanism diagnostic. **Beat:** efficiency framing; not a length-degradation analysis.

**11. Long-Context Generalization with Sparse Attention (α-entmax) — (authors per arXiv). arXiv:2506.16640, ICLR 2026. Verified: YES.**
Formalizes that softmax attention disperses as n grows (entropy ~log n; signal dilution), and that α-entmax keeps bounded entropy → better length generalization. **Relevance:** the cleanest statement of mechanism M1 (attention dilution). **Borrow:** the entropy/dispersion metrics for the mechanism analysis. **Beat:** it's a *training/architecture* fix; and crucially M1 is exactly what the anchor's masking experiment seems to *rule out* — P6 can use this paper's own metrics to show dilution is insufficient.

**12. Long Context, Less Focus: A Scaling Gap Revealed through Privacy and Personalization (PAPerBench) — (authors per arXiv). arXiv:2602.15028 (2026). Verified: YES.**
Benchmark (≈29k instances, 1K–256K) showing personalization/privacy degrade with length; theoretical analysis attributing it to attention dilution in fixed-capacity transformers. **Relevance:** the most recent "dilution explains length degradation" claim; a direct intellectual competitor on mechanism. **Borrow:** the theoretical dilution model as the M1 hypothesis to test/falsify. **Beat:** it asserts dilution without the masking control that would isolate it; P6's masked regime is the discriminating experiment.

**13. Intelligence Degradation in Long-Context LLMs: Critical Threshold via Natural Length Distribution — Wang, Min, Zou. arXiv:2601.15300 (Jan 2026). Verified: YES.**
Uses each sample's natural (un-truncated) length to argue degradation is caused by length itself; finds a "critical threshold" (~40–50% of max context for Qwen2.5-7B) where F1 collapses 45.5%. **Relevance:** a 2026 competitor making a stronger *causal* claim than the anchor. **Borrow:** the natural-length design as additional causal evidence; threshold-detection methodology. **Beat:** it locates *when* but not *why*; no mechanism, no fix — leaves P6's mechanism+intervention lane open.

**14. Revisiting Long-Context Modeling from a Context-Denoising Perspective (CDT) — (authors per arXiv). arXiv:2510.05862 (2026). Verified: YES.**
Integrated-Gradient "noise score" detects irrelevant tokens that mislead attention; even simple noise mitigation boosts attention on critical tokens; proposes Context Denoising Training. **Relevance:** shows attention-on-critical-tokens is recoverable. **Borrow:** the IG noise-attribution metric as a diagnostic. **Beat:** CDT requires *training*; P6's lane is training-free and in the *no-distractor* (masked/whitespace) regime where "denoising" does not apply.

**15. Stream: Scaling Mechanistic Interpretability to Long Context via Sparse Attention — (authors per arXiv). arXiv:2510.19875 (2026). Verified: YES.**
Sparse Tracing + hierarchical pruning estimate per-head attention masks in near-linear time/space, enabling one-pass interpretability at 100k+ tokens; on RULER it discards 90–96% of interactions while preserving retrieval paths. **Relevance:** the enabling *tool* for P6's mechanistic analysis on a single A100. **Borrow:** Stream as the instrumentation to measure where attention/signal is lost as length grows. **Beat:** N/A — it's tooling; adopt it.

**Secondary / context (verified by name in search, used as supporting cites, not deep-reviewed):** InfiniteBench (arXiv:2402.13718), HELMET (arXiv:2410.02694), A Controlled Study on Long-Context Extension & Generalization (arXiv:2409.12181), Base of RoPE Bounds Context Length (arXiv:2405.14591), END: Early Noise Dropping (arXiv:2502.18915), YaRN (arXiv:2309.00071) / Position Interpolation (arXiv:2306.15595) for RoPE-extension priors.

---

### Gaps & Novelty — What Does NOT Yet Exist

**Already exists (do NOT claim as contribution):**
- Existence of length-induced degradation under perfect retrieval — anchor 2510.05381; reinforced by 2601.15300, 2602.15028.
- Training-free **attention re-calibration** mitigation — Found in the Middle 2406.16008 (+15pp), PINE 2407.01100.
- Training-free **recitation** mitigation — anchor 2510.05381.
- **Reordering** evidence to the ends — Lost in the Middle and its follow-ups.
- The **attention-dilution** mechanism narrative (M1) — 2506.16640, 2602.15028.
- Controlled **distractor** injection — GSM-DC 2505.18761, BABILong, RULER.
- Mech-interp **tooling** for long context — Stream 2510.19875.

**Genuine white space (defensible novelty):**
1. **Orthogonalization of token-count vs. absolute-position under guaranteed-perfect retrieval.** The 2×2 {few/many tokens in attention} × {evidence at small/large absolute position}, with whitespace/masked variants, has not been run. This directly tests whether the anchor's "length" effect is really a *position/RoPE* effect (M2) once dilution (M1) is masked away. **This is the headline experiment.**
2. **A falsification of M1 in the masked regime.** Use the dilution metrics from 2506.16640/2602.15028 to show that when distractors are masked out of the softmax, dilution is near-constant yet accuracy still drops — i.e., dilution is *insufficient*, mechanism is positional/representational. No one has done this discriminating test.
3. **A mechanism-targeted, training-free fix** (e.g., position re-mapping / RoPE-index compression of the relevant span, or per-head attention-temperature recalibration conditioned on the diagnosed mechanism) that **beats recitation and Found-in-the-Middle at lower token overhead** in the no-distractor regime. The existing fixes target *order/position bias* (FitM, PINE) or *add tokens* (recitation); a fix targeting raw-count/absolute-position would be new.
4. **A perfect-retrieval-certified benchmark variant** (RULER/BABILong with verbatim-extractable evidence and an explicit retrieval-success check per example) so that, unlike RULER/BABILong, length and retrieval are provably decoupled at the instance level.

**Honest novelty ceiling:** items 1–2 are the strongest (clean negative/causal results are publishable at ACL/EMNLP analysis tracks). Item 3 is higher-reward but higher-risk (may end up marginal vs. recitation/FitM — the classic "mitigation marginal" trap). Item 4 is a useful artifact but secondary.

---

### Critical Assessment

**Well-posedness:** Strong *after* reframing. "Does length hurt under perfect retrieval?" is no longer well-posed (answered yes). "*Which* mechanism causes it once dilution and distraction are controlled, and can a fix derived from that beat recitation?" is sharp, falsifiable, and has a clear discriminating experiment (the masked 2×2). Keep the perfect-retrieval isolation as the methodological core — it is the part the scaffold gets right in spirit.

**Novelty vs. the anchor paper:** This is the crux. The anchor owns the existence claim and a mitigation; **its weakness is precisely P6's opportunity** — it never separates absolute position from count and never names the mechanism. So P6 is *not* redundant *if* it leads with mechanism resolution + a mechanism-derived fix. Run as scaffolded (probe + reorder + "show length hurts"), it is redundant and will read as a late replication.

**Feasibility (solo, 1×A100/Colab):** High. 7–9B open models (Llama-3.1-8B, Qwen2.5/3-7B, Gemma-2-9B) at up to ~32–64K context fit on one A100-40/80GB with attention hooks; RULER/BABILong generators are open; Stream (2510.19875) makes attention attribution tractable. No training required for the core result. Total ≈80–120 GPU-hrs.

**Risk of incrementalism (the dominant risk):** Real and high if the project drifts back to "another mitigation." The mitigation lane is saturated (recitation, FitM, PINE, LDAR, CDT, END). Mitigate by *front-loading the mechanism result* — a clean falsification of attention-dilution-as-cause in the masked regime is publishable even if the new fix only ties recitation.

**Scoop risk:** Elevated and concentrated. Hao Peng's group co-authored *both* the anchor (2510.05381) and PINE (2407.01100); a v2 of the anchor adding the mechanism is plausible. Two 2026 papers (2601.15300, 2602.15028) are already circling the same target. **Move fast on the orthogonalization experiment; it is the one thing not yet published.** If a competitor publishes the count-vs-position disentanglement first, P6's novelty largely evaporates — treat this as the kill-trigger to monitor.

**Verdict:** **PIVOT.** Keep the perfect-retrieval-isolation methodology and the open-model/single-A100 plan; discard the "show length hurts + generic reorder/calibrate" framing; lead with mechanism disentanglement and a mechanism-targeted fix.

---

### Recommended Implementation Phases (tied to the scaffold)

The current `src/core.py` `ContextDegradationProbe` (an MLP mapping one hidden vector → a synthetic degradation label, plus a sort-by-degradation reorder) does **not** reflect the real contribution and should be demoted to an optional diagnostic. The real system is a **controlled-data generator + attention/representation instrumentation + training-free interventions**. Note also: `README.md`/`run.md`/`config.py` carry **P1 boilerplate** (best-of-N, MATH-500/AIME target "1.5–2× compute reduction") that is wrong for P6 — fix these first.

**Phase 0 — De-P1 the scaffold (Week 1).** Rewrite `config.py` (drop `n_samples_best_of_n`, `budget_max_tokens`; add `context_lengths`, `position_bins`, `filler_mode∈{natural,whitespace,masked}`, `models`), `README.md`, and `run.md` to long-context terms. Set models = {Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it}.

**Phase 1 — Perfect-retrieval-certified controlled generator (Weeks 1–3).** Build `src/data_gen.py`: RULER/BABILong-style items where evidence is verbatim-extractable, plus a per-example **retrieval check** (model must reproduce the evidence) so retrieval success is certified, not assumed. Implement the three filler modes (natural distractor / whitespace / attention-masked) and the **2×2 axes**: token-count ∈ {short…long} crossed with relative absolute-position of evidence ∈ {early, mid, late}. This operationalizes the length-vs-retrieval *and* length-vs-position isolation.

**Phase 2 — Degradation map + mechanism instrumentation (Weeks 3–5).** Run the 2×2×filler grid; record accuracy with bootstrap CIs. Add `src/mechanism.py` extracting: attention entropy/dispersion vs. n (test M1 per 2506.16640/2602.15028), attention-sink mass (StreamingLLM), heavy-hitter concentration (H2O), hidden-state norms, and RoPE-distance attention decay (M2 per 2405.14591). Use Stream (2510.19875) for scalable per-head attribution. **Decisive test:** in the masked regime, is dilution flat while accuracy drops? If yes → M2 dominates.

**Phase 3 — Mechanism-targeted training-free fix (Weeks 5–7).** Depending on Phase 2: if M2, implement **position re-mapping** (re-index the relevant span to small absolute positions / compress RoPE indices of evidence) and/or **per-head attention-temperature recalibration**; reproduce **recitation** (anchor) and **Found-in-the-Middle** as baselines. Repurpose the `ContextDegradationProbe` as an *optional* per-example degradation predictor from internal signals (entropy, norms) to decide when to apply the fix.

**Phase 4 — Head-to-head evaluation (Weeks 7–9).** Compare {no-fix, recitation, Found-in-the-Middle, PINE-style, P6 fix} on accuracy vs. **token overhead** across models, lengths, positions, filler modes. Primary metric: accuracy recovered per extra token. Statistical testing: bootstrap CIs; report on RULER, BABILong subset, and the certified generator.

**Phase 5 — Analysis & artifacts (Weeks 9–10).** Figures: (1) accuracy vs. length factored by position (the disentanglement); (2) dilution-flat-yet-accuracy-drops in masked regime (the falsification); (3) fix vs. baselines on accuracy/token Pareto; (4) per-example degradation predictability. Honest negatives: regimes where the fix ties or loses to recitation.

---

### Compute Budget Estimate

| Item | Detail | GPU-hrs |
|---|---|---|
| Phase 1 generator + retrieval-certification runs | 3 models × small grid | ~8 |
| Phase 2 degradation map + attention/representation extraction | 3 models × 2×2 × 3 filler × lengths, with hooks | ~45 |
| Phase 3 fix development + baselines (recitation, FitM, PINE) | iterative | ~25 |
| Phase 4 head-to-head eval (RULER + BABILong subset + generator) | 3 models × 5 methods | ~25 |
| Buffer / re-runs | — | ~15 |
| **Total** | **1×A100-40/80GB or Colab** | **~80–120 GPU-hrs (~$150–300)** |

No training needed for the core result (all interventions training-free); fits Colab A100 sessions if checkpointed. Attention-hook extraction at 32–64K is the memory-heaviest step — use Stream/sparse tracing and FlashAttention to stay within 40GB; cap at 32K if on 40GB.

---

### Citation Caveats / Unverified IDs

- **All 15 deeply-reviewed papers were web-verified** (arXiv abstract pages / venue anthologies) during this review, including every 2026-dated ID used: 2601.15300, 2602.15028, 2510.05862, 2510.19875, 2506.16640 (ICLR 2026). None appear speculative.
- **Master `research.md` §P6 table — status:** the three entries that carry explicit arXiv IDs are all **correct and verified**: 2510.05381 (anchor), 2509.21865 (LDAR), 2406.16008 (Found in the Middle). The remaining P6 table rows are **generic placeholders without IDs** ("Needle-in-a-Haystack evaluations," "Retrieval-Augmented Generation surveys," "Mechanistic interpretability of attention dilution," "Gemma/Llama long-context reports," "Lost in the Middle follow-ups") — not wrong, but not citable as-is; this document supplies concrete replacements (RULER 2404.06654, BABILong 2406.10149, PINE 2407.01100, StreamingLLM 2309.17453, H2O 2306.14048, α-entmax 2506.16640, etc.).
- **Compass artifact §P6** cites 2510.05381, 2509.21865, 2406.16008 — **all verified correct.**
- **Scaffold bug (not a citation issue, but flag):** `p06-long-context-degradation/README.md` and `run.md` state a target bar of "≥1.5–2× compute reduction vs. best-of-N on MATH-500/AIME" and reference `arXiv:2510.05381` as the only key ref — the target-bar text is **copied from P1 and is wrong for P6** (P6 is not a test-time-compute/best-of-N problem). `config.py` defaults (`benchmarks=["math500","gsm8k"]`, `n_samples_best_of_n`, `budget_max_tokens`) are likewise P1 leftovers.
- **IDs cited from strong prior knowledge but not re-fetched line-by-line this session** (treat as high-confidence, verify before camera-ready): Lost in the Middle arXiv:2307.03172 (TACL 2024 anthology entry confirmed), InfiniteBench 2402.13718 (confirmed via GitHub result), HELMET 2410.02694, A Controlled Study 2409.12181, Base of RoPE 2405.14591, END 2502.18915, YaRN 2309.00071, Position Interpolation 2306.15595. Author lists for some 2026 papers (2506.16640, 2602.15028, 2510.05862, 2510.19875) were not all captured — fill from arXiv before submission. **No citation was fabricated.**

---

### Next-Actions Checklist

- [ ] **Decide go/no-go on the pivot** (mechanism-first) vs. the scaffold's diagnose+mitigate framing. Recommended: pivot.
- [ ] **Monitor scoop risk weekly** (arXiv: Hao Peng group; "context length" + "mechanism"/"position vs length"). Kill-trigger: someone publishes the count-vs-position disentanglement first.
- [ ] **Phase 0:** strip P1 boilerplate from `README.md`, `run.md`, `config.py`; rewrite to long-context terms.
- [ ] **Phase 1:** build perfect-retrieval-*certified* generator with the 2×2 (count × position) × 3 filler modes (natural/whitespace/masked).
- [ ] **Phase 2 (highest priority experiment):** run the masked-regime test — is attention dilution flat while accuracy still drops? This is the headline falsification.
- [ ] Stand up baselines early: **recitation** (anchor), **Found-in-the-Middle** (2406.16008), **PINE** (2407.01100).
- [ ] Integrate **Stream** (2510.19875) for scalable attention attribution; add entropy/sink/heavy-hitter/RoPE-decay probes.
- [ ] **Phase 3:** implement the mechanism-targeted training-free fix (position re-mapping / attention-temperature) only after Phase 2 names the mechanism.
- [ ] Repurpose `ContextDegradationProbe` as an optional internal-signal degradation predictor (not the centerpiece).
- [ ] Fill missing author lists / verify the high-confidence IDs above before any submission.

---

*Last updated: 2026-06-28. Source: `research.md` §P6 + web verification (2024–2026 literature). Verdict: PIVOT to mechanism-first.*
