# P7 Research Foundation — Automated Discovery of Novel Biases in LLM-as-Judge

**Problem:** Can we *automatically discover* (not merely catalogue) evaluation biases in LLM-as-judge systems, with *causal validation* — generate contrastive perturbations that hold answer quality fixed while varying a candidate bias factor, and measure the judge's preference shift (flip-rate / FPR)?

**Compute:** API judges + 1×A100 (~50 GPU-hrs, ~$100–300)
**Venues:** ACL / EMNLP / NeurIPS D&B
**Project:** `p07-bias-discovery-llm-judge/`
**Parent index:** [`../../research.md`](../../research.md) §P7

---

### TL;DR verdict — **PIVOT (do not run the original framing; do not kill).**

The original P7 thesis — "automated *discovery* of unknown judge biases via quality-preserving contrastive perturbations, measured by a preference-shift/flip-rate metric" — **has been substantially scooped**. The master doc's anchor, **BiasScope (arXiv:2602.09383), is REAL** — it is a published **ICLR 2026** paper (code at `github.com/sustech-nlp/BiasScope`), not a speculative 2026 ID. BiasScope does almost exactly what P7 proposed: an LLM-driven framework that (a) automatically discovers *unknown* biases beyond the known catalogue, (b) generates **counterfactual perturbations that preserve answer correctness** (it reports only 8.5% quality leakage), (c) validates each candidate bias by an **error-rate-increase test** on held-out data (its flip-rate analogue), and (d) reports **48 validated biases** including genuinely novel ones (Novelty, Exact-Match, Confirmation, Completeness, Complexity, Moral-Licensing). Worse, a **second** 2026 paper, *Automated Concept Discovery for LLM-as-a-Judge* (arXiv:2603.03319), attacks the same "discover-without-a-predefined-taxonomy" goal from the input side using SAEs. The headline novelty of P7 as written is therefore gone.

**But there is a defensible residual gap.** BiasScope's validation is *associational* (does error rate go up under perturbation?), **not causal** — it admits quality leakage, applies multi-attribute rewrites that confound the bias factor, and validates on a single text-preference benchmark. Nobody has yet done **rigorous causal isolation of a single bias factor with answer quality held *provably* fixed** — which is only credible in a domain where quality is *machine-checkable*. That domain is **code**: CodeJudgeBench / LiveCodeBench give binary, execution-verified correctness, so "hold quality fixed" stops being an approximation and becomes ground truth. **Recommended pivot: reframe P7 as "causally-validated bias discovery in *executably-verifiable* code-judges"** — use BiasScope/Concept-Discovery as the cited discovery front-end, and contribute (1) a causal-inference estimation layer (matched counterfactuals + average-treatment-effect flip-rate with CIs + mediation to separate the factor from confounds), (2) the code domain where quality-invariance is verifiable, (3) a causally-validated code-judge bias catalogue + stress benchmark. **Scoop risk remains high** (≥4 directly-adjacent papers landed Feb–Apr 2026; "Bias in the Loop", arXiv:2604.16790, already does semantics-preserving SE-judge perturbations). Novelty is now medium and *methodological* (rigor + domain), not "first to discover". Proceed only with the pivot.

---

### Problem Statement & Framing

> Can we automatically discover (not just catalogue) evaluation biases in LLM judges with causal validation?

**Original core claim (now largely pre-empted):** Prior work catalogues *known* biases (position, verbosity, self-enhancement, authority); an automated pipeline that *discovers* unknown ones and *causally validates* them via quality-preserving contrastive perturbations would be novel.

**Why the framing must change.** As of mid-2026 the "automated discovery vs. cataloguing" distinction — which is the conceptual spine of P7 — is *exactly* the framing BiasScope (ICLR 2026) uses in its own abstract ("transforming bias discovery from a passive process relying on manual effort and predefined bias lists into an active and comprehensive automated exploration"). The distinction is no longer white space; it is the established frame of the most recent prior work.

**Revised core claim (defensible):** Existing discovery methods validate biases *associationally* and on *subjectively-scored text*, where "answer quality held fixed" can only be approximated. In a domain with *objective, executable* quality (code with unit tests), one can (i) construct counterfactual response pairs whose correctness is *identical and verified*, (ii) estimate the *causal* effect of a candidate bias factor on judge preference (a flip-rate / ATE with confidence intervals), and (iii) use mediation analysis to confirm the effect is not carried by a confounded attribute (e.g., length). This converts P7 from a (scooped) discovery claim into a (open) **causal-rigor + code-domain** claim.

**Target bar (revised):**
- A reproducible pipeline that, on CodeJudgeBench / LiveCodeBench-derived pairs, measures **causally-isolated flip-rate** for ≥6 candidate bias factors on ≥4 judges (2 API + 2 open), with bootstrap CIs and a mediation check separating each factor from length/format confounds.
- Show ≥2 bias factors that are **causally significant in code-judges but absent/weaker in text-judges** (or vice-versa) — a domain-transfer finding BiasScope cannot claim.
- Release a small "stress" split (a code analogue of BiasScope's JudgeBench-Pro) of causally-validated adversarial perturbations on which strong judges flip ≥X% of the time.

---

### Research Landscape 2024–2026 — *cataloguing* vs. *discovery*

The central organizing axis for P7 is **cataloguing known biases** → **automated discovery of unknown biases** → (newest frontier) **causal validation / robustness / security of judges**.

| Layer | What it does | Representative work (verified) | Status for P7 |
|---|---|---|---|
| **Foundations** | Establish LLM-as-judge; first name position/verbosity/self-enhancement | MT-Bench (2306.05685), G-Eval (2303.16634), Prometheus 2 (2405.01535) | Background |
| **Cataloguing known biases** | Quantify a *named* bias (position, length, self-preference, authority) | Position-bias study (2406.07791), Self-recognition (2404.13076), Self-preference (2410.21819, 2604.22891), Length (2404.04475, 2407.01085), Scoring bias (2506.22316), Silent/shortcut bias (2509.26072) | The thing P7 wanted to "beat" — now well-trodden |
| **Judge benchmarks** | Objective-correctness testbeds | JudgeBench (2410.12784), CodeJudgeBench (2507.10535), robustness assessment (2506.09443) | Testbeds for P7 |
| **Surveys / taxonomies** | Organize the bias zoo | From Generation to Judgment (2411.16594), A Survey on LLM-as-a-Judge (2411.15594), Security SoK (2603.29403) | Map of the territory |
| **AUTOMATED DISCOVERY** | Find *unknown* biases without a predefined list | **BiasScope (2602.09383)**, **Automated Concept Discovery (2603.03319)** | **This is P7's claimed white space — already occupied** |
| **Causal / mitigation methods** | Isolate a factor causally; debias | CausaLM (2005.13407), Causal Mediation (2004.12265), Dubois LC-AlpacaEval (2404.04475), bias-mitigation eval (2604.23178), OffsetBias (2407.06551) | Methods to **borrow** for the pivot |
| **Code/SE judge domain** | Judges over code artifacts | CodeJudgeBench (2507.10535), Bias-in-the-Loop SE audit (2604.16790) | **Residual white space (causal discovery here is open)** |

**Key takeaway:** the field crossed the "cataloguing → discovery" line *between* the master-doc snapshot and now. Two independent automated-discovery frameworks exist (BiasScope perturbs the *response*; Concept-Discovery analyzes *input/embedding* concepts). What *no one* has nailed is **causal** validation with **provably** fixed quality — and the only domain where that is achievable cleanly (code) has discovery work that is auditing-only, not automated-discovery.

---

### Deep Paper Reviews (15)

> "Verified?" = confirmed to exist via web search on the live arXiv/venue listing during this review (mid-2026).

**1. BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation — Lai, Ou, Wang, Wang, Yang, Chen, Chen.**
arXiv:2602.09383 · ICLR 2026 · **Verified? YES** (arXiv abs+HTML, ICLR poster, OpenReview QGOw6AU8Lp, GitHub `sustech-nlp/BiasScope`).
*Summary:* An LLM-driven, seed-and-expand pipeline. A teacher (Qwen2.5-72B) perturbs *rejected* responses with known biases, extracts the target judge's misjudgments, "cascades" deeper self-explanations, and runs `IdentifyBias()` to propose *new* candidate biases. Each candidate is retained iff it raises held-out error rate (Err(perturbed) > Err(original)); reports +6.9% avg error inflation, 48 validated biases, and a "Longer-Length-Is-Not-The-Key" decomposition. Ships **JudgeBench-Pro** (1,178 samples) on which even GPT-4o/DeepSeek-V3/Kimi-K2 exceed 50% error.
*Relevance:* This is the **direct precursor / scooper** of P7's entire original framing (automated discovery + quality-preserving counterfactual perturbation + preference-shift metric + novel biases + JudgeBench).
*Borrow:* the perturbation taxonomy, the JudgeBench validation setup, the length-control decomposition. *Beat:* its validation is associational error-inflation, not causal ATE; admits 8.5% quality leakage; multi-attribute rewrites confound factors; single benchmark (its own stated limitation); no code domain; no statistical CIs/mediation.

**2. Automated Concept Discovery for LLM-as-a-Judge Preference Analysis — Wedgwood, Yadav, Smith.**
arXiv:2603.03319 · 2026 · **Verified? YES** (arXiv abs).
*Summary:* Discovers preference drivers **without a predefined taxonomy** via embedding-level concept extraction; finds **sparse-autoencoder** concepts are the most interpretable while still predicting judge decisions. Validated on 27k+ paired responses from human-preference datasets across three LLM judges.
*Relevance:* The *second* automated-discovery scooper, attacking the same "no predefined list" goal from the **input/representation** side rather than response perturbation.
*Borrow:* SAE-based concept surfacing as a *complementary* discovery front-end. *Beat:* it is correlational concept analysis, no causal perturbation test, no code domain.

**3. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — Zheng et al.**
arXiv:2306.05685 · NeurIPS 2023 · **Verified? YES.**
*Summary:* Foundational LLM-as-judge paper; first to name and measure **position, verbosity, and self-enhancement** biases and propose mitigations; introduces MT-Bench + Chatbot Arena.
*Relevance:* The canonical "cataloguing of known biases" baseline and a P7 testbed. *Borrow:* MT-Bench prompts, the position-swap protocol. *Beat:* it catalogues by hand — the antithesis P7 is defined against.

**4. LLM Evaluators Recognize and Favor Their Own Generations — Panickssery, Bowman, Feng.**
arXiv:**2404.13076** · NeurIPS 2024 · **Verified? YES.** *(Master doc cited this as 2406.07791 — that ID is WRONG; see caveats.)*
*Summary:* Shows GPT-4/Llama-2 judges recognize their own outputs and that **self-recognition linearly predicts self-preference** strength.
*Relevance:* The definitive *self-enhancement / own-generation* bias paper P7 must situate against. *Borrow:* the controlled self-vs-other pairing. *Beat:* single bias, no automated discovery.

**5. Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge.**
arXiv:2406.07791 · 2024 · **Verified? YES** (this is the *actual* paper at that ID — position bias, **not** self-enhancement).
*Summary:* Systematic investigation of **position bias** in pairwise LLM judging (repetition stability, position consistency/fairness metrics).
*Relevance:* Strong known-bias baseline + a clean metric vocabulary (consistency/fairness) reusable for the flip-rate definition. *Borrow:* metrics. *Beat:* cataloguing only.

**6. JudgeBench: A Benchmark for Evaluating LLM-based Judges — Tan et al.**
arXiv:2410.12784 · ICLR 2025 · **Verified? YES.**
*Summary:* Objective-correctness benchmark — each item is one objectively correct vs. one objectively incorrect response (knowledge/reasoning/math/code), labeled by ground-truth + dual-LLM verification, avoiding subjective preference.
*Relevance:* P7's core *text-side* testbed; the substrate BiasScope validates on; basis for JudgeBench-Pro. *Borrow:* the objective-correctness pairing — essential for "quality held fixed."

**7. CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks.**
arXiv:2507.10535 · 2025 · **Verified? YES.**
*Summary:* Built on LiveCodeBench-v6 (1,055 problems, May 2023–Apr 2025); (instruction, good, bad) triplets; benchmarks 26 judges. Finds **response-order substantially changes accuracy** and judge variance across the code author model — i.e., position bias in code judging.
*Relevance:* **The pivot's primary testbed** — code correctness is execution-verifiable, enabling provable quality-invariance. *Borrow:* triplet construction, thinking-vs-nonthinking judge split. *Beat:* catalogues order effects only; no automated discovery, no causal isolation.

**8. Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering — Zhao, Esmaeili, Fard (UBC).**
arXiv:2604.16790 · Apr 2026 · **Verified? YES.**
*Summary:* Audits SE/code judges for reliability and bias; shows repeated evaluations disagree, small prompt edits swing verdicts, and **semantics-preserving, human-equivalent perturbations elicit divergent verdicts**.
*Relevance:* The **closest competitor to the pivot** — it already does semantics-preserving code perturbations. *Borrow:* its perturbation operators. *Beat:* it is an *audit of perturbation sensitivity*, not *automated discovery* nor *causal-ATE estimation with mediation*; check on release whether it forecloses the causal angle.

**9. Quantifying and Mitigating Self-Preference Bias of LLM Judges — Yang et al.**
arXiv:2604.22891 · 2026 · **Verified? YES.**
*Summary:* Fully automated framework that **constructs equal-quality response pairs with negligible quality difference** to isolate self-preference bias, then mitigates it.
*Relevance:* Directly relevant — "equal-quality pairs" is the same quality-invariance idea P7 needs, applied to one bias. *Borrow:* its equal-quality-pair construction and metric. *Beat:* single bias (self-preference), text domain.

**10. Self-Preference Bias in LLM-as-a-Judge.**
arXiv:2410.21819 · 2024 · **Verified? YES.**
*Summary:* Quantifies systematic self-favoring directional deviation in LLM judges.
*Relevance:* Known-bias baseline for the self-preference family. *Borrow:* metric. *Beat:* cataloguing.

**11. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators — Dubois et al.**
arXiv:2404.04475 · 2024 · **Verified? YES.**
*Summary:* Uses a **causal-inference (regression) framework** to estimate the win-rate "if all outputs had the baseline length," lifting Spearman vs. Chatbot Arena from 0.94→0.98.
*Relevance:* The cleanest existing *causal-control* template for a judge bias (length). **Methodological keystone to borrow** for P7's mediation/confound-control layer. *Beat:* single factor (length), regression-only.

**12. From Generation to Judgment: Opportunities and Challenges of LLM-as-a-Judge — Li et al.**
arXiv:2411.16594 · 2024 · **Verified? YES** (master-doc ID correct).
*Summary:* Comprehensive survey; taxonomy of what/how/benchmark, with task-agnostic vs. judgment-specific bias classes.
*Relevance:* The taxonomy to position P7's discovered factors against (so reviewers see they are *not* re-discoveries). *Borrow:* taxonomy scaffolding.

**13. CausaLM: Causal Model Explanation Through Counterfactual Language Models — Feder, Oved, Shalit, Reichart.**
arXiv:2005.13407 · CL 2021 · **Verified? YES.**
*Summary:* Estimates the **causal effect of a concept** on model predictions by training counterfactual representations and comparing predictions with/without the concept.
*Relevance:* The methodological backbone for "isolate the bias factor causally," importable to judge logits/preferences. *Borrow:* the causal-concept-effect estimator. *Beat (extend):* apply to judge preference rather than classifier output.

**14. Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias — Vig et al.**
arXiv:2004.12265 · NeurIPS 2020 (Spotlight) · **Verified? YES.**
*Summary:* Mediation analysis decomposes a bias effect into direct vs. indirect (mediated) components inside a network.
*Relevance:* Lets P7 show a discovered factor's effect is *not* mediated by a confound (e.g., length/format) — the rigor BiasScope lacks. *Borrow:* mediation decomposition for flip-rate attribution.

**15. Discovering Language Model Behaviors with Model-Written Evaluations — Perez et al.**
arXiv:2212.09251 · ACL Findings 2023 · **Verified? YES.**
*Summary:* Uses LLMs to *automatically generate* evaluation datasets that surface model behaviors/failure modes at scale.
*Relevance:* The intellectual ancestor of LLM-driven automated discovery (and of BiasScope's teacher loop). *Borrow:* model-written generation of perturbations/probes. *Beat:* not about judges or causal validation.

**Supporting (verified, cite as needed):** A Survey on LLM-as-a-Judge (2411.15594); Security in LLM-as-a-Judge — SoK (2603.29403); LLMs Cannot Reliably Judge (Yet?) (2506.09443); Judging the Judges — Bias-Mitigation Strategies (2604.23178); OffsetBias (2407.06551); Explaining Length Bias (2407.01085); Evaluating Scoring Bias (2506.22316); Silent/shortcut bias (2509.26072).

---

### Gaps & Novelty — what does NOT yet exist

Split deliberately into the two layers, because P7's value now lives entirely in the second.

**A. Known biases / cataloguing — CLOSED.**
Position, verbosity/length, self-enhancement/self-preference, authority/style, scoring, shortcut/format biases are all individually catalogued and quantified (MT-Bench, 2406.07791, 2404.13076, 2410.21819, 2604.22891, 2404.04475, 2506.22316, 2509.26072). Re-measuring any of these is *not* a contribution.

**B. Automated discovery — MOSTLY CLOSED (this is the scoop).**
- *Response-perturbation discovery:* **BiasScope (2602.09383)** — done, ICLR 2026.
- *Input/representation concept discovery:* **Automated Concept Discovery (2603.03319)** — done, SAE-based.
- *Equal-quality-pair isolation for one bias:* **2604.22891** — done for self-preference.
Building "yet another automated discovery pipeline" is now a re-implementation, not research.

**C. The remaining WHITE SPACE (defensible P7 contribution):**
1. **Causal vs. associational validation.** Every discovery paper validates by *error inflation / correlation*. None estimates a **causal flip-rate (ATE)** with (i) **provably** quality-invariant counterfactuals, (ii) bootstrap **CIs / significance**, and (iii) **mediation analysis** ruling out confounds (length, format, token-overlap). Dubois (length) and CausaLM/Vig (concepts) provide the tools but were never combined for judge-bias discovery.
2. **Executably-verifiable quality domain.** "Hold quality fixed" is only *provable* where quality is machine-checkable. **Code** (CodeJudgeBench / LiveCodeBench, unit-test pass/fail) is the cleanest such domain and is *unexploited for automated causal discovery* — existing code work (2507.10535, 2604.16790) is cataloguing/auditing only.
3. **Cross-domain bias transfer.** No one shows which discovered biases are domain-specific (code-only) vs. universal (text↔code). This is a clean, reviewer-legible finding.
4. **A causally-validated code stress benchmark** ("CodeJudge-Pro" analogue of JudgeBench-Pro) — a NeurIPS D&B-shaped artifact.

**Net:** P7 survives only as **"causally-validated bias discovery where answer quality is execution-verifiable (code judges)."** The novelty is *methodological rigor + domain*, medium-strength, defensible.

---

### Critical Assessment

**Well-posedness.** The revised question is sharp and falsifiable: for factor *f*, does randomizing *f* across quality-identical (unit-test-equivalent) code-response pairs causally shift judge preference (flip-rate ATE ≠ 0, CI excludes 0, effect not mediated by length/format)? This is cleaner than the original because code makes "quality held fixed" *verifiable* rather than asserted.

**Novelty vs. real precursors.** Against BiasScope the discovery novelty is **gone**; against BiasScope + Concept-Discovery + 2604.22891 + Bias-in-the-Loop the residual novelty is the *combination* {causal ATE + mediation + provable code-quality-invariance + cross-domain transfer}. That combination is, as of this review, unclaimed — but it is an *incremental* novelty, sellable to ACL/EMNLP/D&B, not a flagship.

**Feasibility (solo, low-compute).** High. API judges (GPT-4o-class, DeepSeek, Claude) + 2 open judges (Qwen2.5/3, Llama) on a single A100; perturbations generated by an open teacher; correctness from a sandboxed unit-test runner (LiveCodeBench harness exists). Causal estimation is cheap (regression/bootstrap/mediation on CPU). The expensive part is API judge calls, controllable to the $100–300 budget.

**Biggest risks.**
1. **Scoop velocity (highest).** ≥4 directly-adjacent papers landed Feb–Apr 2026; Bias-in-the-Loop (2604.16790) already does semantics-preserving code-judge perturbations. A causal-discovery code paper could appear mid-execution. *Mitigation:* move fast, lead with the *causal-mediation + execution-verified-quality* angle that audits lack, and frame as benchmark+method.
2. **"Discovered biases are trivial/known."** The pivot sidesteps this by making the contribution *causal rigor*, not novelty of the bias — even re-confirming length/self-preference *causally and in code* is publishable if the method is the point. BiasScope already shows non-trivial biases exist, de-risking the "nothing new" outcome.
3. **Quality-invariance still imperfect even in code.** Two programs can both pass tests yet differ in latent quality (style, complexity). *Mitigation:* that *is* the experiment — condition on pass/fail AND complexity, use mediation to separate.
4. **Scaffold mismatch.** The current `src/core.py` (`BiasDiscoveryProbe`, a tiny MLP on concatenated pair embeddings) does **not** match the real method, which is prompt/LLM-driven perturbation + API judging + causal estimation — not a trained probe. The scaffold needs a near-complete rewrite (see phases). The existing `compute_ece()` in `evaluate.py` is reusable; little else is.

**Verdict: PIVOT.** Kill the "first automated discovery" claim; build the causal + code-domain reframing, or fold this effort into a smaller "causal-validation add-on that audits BiasScope's discovered biases in code" if time-boxed.

---

### Recommended Implementation Phases (tied to the scaffold)

**Phase 0 — Reframe + scaffold rewrite (Week 1).**
Replace `BiasDiscoveryProbe` with the real pipeline objects: `JudgeClient` (API + open, with position-swap controls), `PerturbationGenerator` (teacher-LLM rewrites + rule-based code transforms), `CausalEstimator` (flip-rate ATE, bootstrap CI, mediation). Keep `compute_ece()`. Update `run.md` (currently inherited boilerplate referencing MATH/AIME — wrong for P7).

**Phase 1 — Reproduce the cataloguing baselines (Weeks 1–3).**
On JudgeBench (2410.12784) and CodeJudgeBench (2507.10535): reproduce **position-swap flip-rate** and **length/verbosity preference** for ≥4 judges. This anchors the field and is required to claim anything causal later. Output: baseline flip-rate tables to `results/`.

**Phase 2 — Quality-invariant counterfactual construction (Weeks 3–5).**
Code-side (primary): from LiveCodeBench solutions, build pairs that are **both unit-test-passing** (quality verified identical) but differ on one candidate factor (comment density, identifier verbosity, novelty/idiom, exact-match-to-prompt, complexity). Text-side (secondary, for transfer): reuse BiasScope/JudgeBench equal-quality pairs (2604.22891 method). Log a leakage check (analogue of BiasScope's 8.5%).

**Phase 3 — Causal estimation + mediation (Weeks 5–7).**
For each factor: estimate flip-rate **ATE** with bootstrap CIs (Dubois-style regression control, 2404.04475), then **mediation analysis** (Vig, 2004.12265 / CausaLM, 2005.13407) to confirm the effect is not carried by length/format. A factor is "causally validated" only if CI excludes 0 *and* direct effect survives mediation.

**Phase 4 — Automated discovery layer (Weeks 7–9, optional/secondary).**
*Build on, don't reinvent:* run BiasScope's `IdentifyBias` loop (its code is public) and Concept-Discovery's SAE surfacing to *propose* code-specific candidate factors, then pass them through the Phase-3 causal filter. The contribution is the *causal filter on top of existing discovery*, not the discovery itself.

**Phase 5 — Cross-domain transfer + stress benchmark (Weeks 9–11).**
Compare causally-validated factors text vs. code; ship a small **CodeJudge-Pro** split of causally-validated adversarial perturbations with judge flip-rates. Bootstrap CIs, ≥4 judges (2 API + 2 open).

**Phase 6 — Analysis & artifacts (Weeks 11–12).**
Fig 1: cataloguing→discovery→causal framing. Fig 2: ATE flip-rate per factor with CIs, code vs. text. Fig 3: mediation decomposition (direct vs. length-mediated). Table: causally-validated biases by judge. Honest negatives: factors that are *associationally* significant (BiasScope-style) but *vanish under causal control* — itself a finding.

---

### Compute Budget Estimate

| Item | Detail | Est. cost |
|---|---|---|
| API judge calls | ~4 judges × ~6 factors × ~500 pairs × 2 orders × few repeats ≈ 100–200k calls | $80–250 |
| Open judges (Qwen2.5/3, Llama) | inference on 1×A100 | ~20–40 GPU-hrs |
| Teacher perturbation generation | open 70B-class on A100 or cheap API | ~10–20 GPU-hrs or ~$30 |
| Unit-test execution | sandboxed CPU (LiveCodeBench harness) | negligible |
| Causal estimation | CPU (bootstrap/mediation) | negligible |
| **Total** | **~50 GPU-hrs + API** | **~$100–300** |

Matches the master-doc envelope; Colab-friendly. The only real lever is API call volume — cap pairs/factor and cache aggressively.

---

### Citation Caveats / Unverified IDs

1. **BiasScope (arXiv:2602.09383) is REAL — not speculative.** Contrary to the master doc's warning, it is a verified **ICLR 2026** paper (arXiv abs + HTML + ICLR poster + OpenReview `QGOw6AU8Lp` + GitHub `sustech-nlp/BiasScope`). It is a **direct scooper** of P7's original framing. Treat it as P0 prior work, not a target to claim priority over.
2. **Self-enhancement citation ID in the master doc is WRONG.** Master `research.md` §P7 row 3 lists *"LLM Evaluators Recognize and Favor Their Own Generations — arXiv:2406.07791."* The correct ID is **arXiv:2404.13076** (Panickssery, Bowman, Feng, NeurIPS 2024). The ID **2406.07791** actually points to a *different* paper, *"Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge."* The master doc conflated two papers — fix both rows.
3. **Survey ID is correct.** arXiv:2411.16594 = *From Generation to Judgment* (Li et al.) — verified. (Note a *separate* survey, arXiv:2411.15594 *A Survey on LLM-as-a-Judge* (Gu et al.), also exists — do not confuse the two adjacent IDs.)
4. **Master-doc placeholder rows without IDs** (Position Bias #5, Verbosity #6, Authority #7, CALM #16, Reward-Hacking #17, CheckEval #19, Multi-attribute #20) were generic descriptors. Verified concrete substitutes are listed above (2406.07791, 2404.04475/2407.01085, 2410.21819/2604.22891, 2604.23178, 2506.22316, 2509.26072). The "Causal Inference for NLP — Keith et al. 2020" row (#18) is real as a concept but the more directly usable causal tools are CausaLM (2005.13407) and Vig (2004.12265).
5. **All 2026 IDs cited here were live-verified** (2602.09383, 2603.03319, 2604.22891, 2604.16790, 2603.29403, 2604.23178). They remain *preprints/recent* — re-confirm venue/status before any camera-ready.
6. **No citation in this document was fabricated;** every arXiv ID above was checked against the live listing during this review.

---

### Next-Actions Checklist

- [x] Verify BiasScope anchor — **REAL (ICLR 2026), scoops original framing.**
- [x] Verify/repair master-doc IDs — **2406.07791 mislabeled; correct self-enhancement ID is 2404.13076.**
- [x] Map cataloguing vs. discovery vs. causal layers; identify residual white space.
- [ ] **Decide pivot vs. fold:** commit to "causal + code-domain" reframing (or descope to a causal-audit add-on on BiasScope's biases).
- [ ] Rewrite `src/core.py` from `BiasDiscoveryProbe` MLP → `JudgeClient` + `PerturbationGenerator` + `CausalEstimator`; fix `run.md` (drop MATH/AIME boilerplate).
- [ ] Phase 1: reproduce position/length flip-rate baselines on JudgeBench + CodeJudgeBench (≥4 judges).
- [ ] Phase 2: build unit-test-equivalent code counterfactual pairs (LiveCodeBench harness) + leakage check.
- [ ] Phase 3: flip-rate ATE + bootstrap CI + mediation (Dubois/Vig/CausaLM) — the core differentiator.
- [ ] Phase 4 (optional): run public BiasScope/Concept-Discovery to *propose* code factors, then causal-filter them.
- [ ] Phase 5: cross-domain transfer table + CodeJudge-Pro stress split.
- [ ] Monitor arXiv weekly for code-judge causal-discovery scoops (esp. follow-ups to 2604.16790).

---

*Last updated: 2026-06-28. Source: `research.md` §P7 + live web verification (mid-2026). BiasScope (2602.09383) confirmed real; original P7 framing assessed as scooped → pivot to causal-validation in the executably-verifiable code-judge domain.*
