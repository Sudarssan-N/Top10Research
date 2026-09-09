# A* Opportunity Assessment — Three Interest Topics

**Date:** 2026-09-10  
**Mode:** Deep research (literature scan + gap analysis + FINER problem statements)  
**Topics:** (1) Long-Context Degradation Mechanism, (2) Latent Overthinking Controller, (3) Batch-Aware MoE Routing  
**Question:** Which of these, if any, still supports an A* conference paper (ICLR / ICML / NeurIPS / ACL / EMNLP / MLSys), and what is the problem statement that is actually still open?

**AI disclosure:** Literature search, source screening, and synthesis were AI-assisted (Grok 4.6, 2026-09-10). Every arXiv ID cited below was checked against an arXiv abstract page or a venue page (ACL Anthology, OpenReview, CVPR Open Access). Unverified titles were omitted. Human review of the original PDFs is still required before any submission.

**Search window:** 2024–2026, with extra weight on papers after the portfolio’s last literature pass (2026-06-26/28) through 2026-09-10.

---

## Executive verdict (read this first)

All three topics **already exist in this repo** as P6, P3, and P5. Pipelines are scaffolded. **No real-model experiment has been run** (smoke/mock only). That matters because **ICLR 2027 abstracts are due 18 Sep 2026 and papers 25 Sep 2026** — eight to fifteen days from this assessment. A new ICLR 2027 paper is not feasible.

| Interest topic | Repo project | Local work | Original idea still A*? | Remaining A* slice | Recommendation |
|---|---|---|---|---|---|
| Long-context degradation mechanism | `p06-long-context-degradation/` | Scaffold + 2×2 generator; **mock only** | Existence claim: **no**. Recitation/reorder: **no**. | **Yes, narrow:** certified 2×2 `{few/many attended tokens} × {small/large RoPE index}` under perfect retrieval, plus a remap ablation vs STRING/FitM/PINE | **Best of the three.** Pivot, do not kill. Target **ICML 2027 / ACL 2027**. |
| Latent overthinking controller | `p03-overthinking-latent-controller/` | Probe + phases 1–2; **mock only** | **No.** ROM (Mar–Aug 2026) is the exact method. | **Yes, sharp:** hidden-state detector of *harmful* (correct→wrong) continuation, plus a halt that actually terminates | **Second.** Rewrite the RQ. Do not ship “stop probe on hiddens.” Target **ICML 2027 / ICLR 2028**. |
| Batch-aware MoE routing | `p05-batch-moe-routing/` | Training-free policies + hooks; **smoke only** | **No.** Lynx, OEA, SERE (ICLR 2026), XShare. | **Thin systems remainder:** batch-adaptive \(k_0\) + union-min × measured PCIe/offload | **Deprioritize.** Not NeurIPS/ICLR-main. Only if you want an MLSys grind. |

**Portfolio call:** Among these three, pursue **P6 (mechanism 2×2)** as the flagship and **P3 (harmful-overthinking probe)** as the parallel track. Do not pursue P5 as a first paper.

---

## 1. Do these topics exist here, and have we started?

Yes. They are three of the ten problems in `research.md`.

### 1.1 Mapping

| User wording | Repo | Original problem (scaffold) |
|---|---|---|
| Long-Context Degradation Mechanism | `p06-long-context-degradation/` | Why accuracy drops as context grows even with perfect retrieval; training-free mitigations |
| Latent Overthinking Controller | `p03-overthinking-latent-controller/` | Per-query hidden-state probe that early-stops CoT on easy problems |
| Batch-Aware MoE Routing | `p05-batch-moe-routing/` | Training-free batch policy that shrinks the union of activated experts |

### 1.2 What has actually been run

Honest status as of 2026-09-10:

| Project | Code | Real model run | What results exist |
|---|---|---|---|
| P3 | Phase 1 budget curve + Phase 2 stop probe; `experimentation.md` says “not yet run on real models” | **No** | `results/*/phase*_smoke/` and `colab_run/*_smoke/` — mock JSON + `stop_probe.pt` |
| P5 | Training-free `budget`/`coverage` policies over real gating logits (`moe_hooks.py`); README says “no real MoE run yet” | **No** | smoke JSON only; leftover `test_train_p5` checkpoints from the *old* trained-router scaffold |
| P6 | Certified-perfect-retrieval 2×2 grid (`{length} × {absolute evidence offset}`) + mock recovery test | **No** | smoke `grid_rows.jsonl` / `grid_summary.json`; leftover `test_p6` checkpoints from the old MLP probe |

**Implication:** the literature has moved ~10 weeks since the June 2026 research docs, and the experiments have not. Any A* plan must assume **from-scratch empirical work**, not “we already have numbers.”

---

## 2. Research questions (FINER)

Primary question for this report:

> As of 10 September 2026, which of the three interest topics still contains a problem statement that is novel, feasible on one A100, and competitive at an A* ML/NLP venue?

### 2.1 Candidate problem statements (the ones worth scoring)

**RQ-P6 (recommended).** Under certified-perfect retrieval, does length-induced degradation track **absolute RoPE index** after attention dilution is removed from the softmax, and does **index remapping of the evidence span** recover accuracy better than recitation, Found-in-the-Middle, and STRING at matched decode tokens?

**RQ-P3 (recommended pivot).** Can a frozen-LRM hidden-state probe detect the onset of **harmful** overthinking (a correct prefix that will flip) as distinct from **verbose** overthinking, and can a halt actuator that actually terminates (not a raw `</think>` inject) reduce flip-rate at matched length?

**RQ-P5 (only if systems-track).** Does a **batch-adaptive \(k_0\)** policy, composed with union-minimization on an **instrumented single-GPU offload** runtime, reduce PCIe expert transfers and end-to-end TPOT versus SERE/Lynx/OEA with fixed \(k_0\) at matched accuracy?

### FINER scores

| RQ | F | I | N | E | R | Avg | Notes |
|---|---|---|---|---|---|---|---|
| RQ-P6 | 5 | 4 | 4 | 5 | 4 | **4.4** | Feasible on 7–9B / 32K. Novelty is the *missing cell*, not “length hurts.” Scoop risk: Hao Peng lab. |
| RQ-P3 | 4 | 5 | 4 | 5 | 5 | **4.6** | Labels for “will flip” are expensive (counterfactual prefixes). Novelty is the *failure mode*, not “stop from hiddens.” |
| RQ-P5 | 2 | 3 | 2 | 5 | 3 | **3.0** | Measurement is the hard part; novelty is compositional. Below the usual A* bar unless MLSys-shaped execution is exceptional. |
| Original P3 (“stop probe”) | 4 | 3 | **1** | 5 | 3 | 3.2 | Occupied by ROM. |
| Original P5 (“batch-aware union”) | 3 | 3 | **1** | 5 | 3 | 3.0 | Occupied by Lynx/OEA/SERE/XShare. |
| Original P6 (“show length hurts + recitation”) | 5 | 2 | **1** | 5 | 3 | 3.2 | Occupied by Du et al. 2025 + STRING/FitM. |

**Scope of this report**

- **In:** 2024–2026 literature on the three topics; remaining white space; A* venue fit; local repo status; recommended RQs.
- **Out:** New experiments; paper writing; the other seven portfolio problems except as a one-line alternative.

---

## 3. Search strategy

**Databases / surfaces:** arXiv abs/html, OpenReview (ICLR 2026), ACL Anthology, CVPR Open Access, GitHub (SERE, vLLM RFC #35550), Hao Peng publications page.

**Date range:** 2024-01-01 to 2026-09-10. Extra queries restricted to submissions after 2026-06-01.

**Inclusion:** peer-reviewed or arXiv papers that (a) diagnose or mitigate overthinking / CoT early-exit, (b) change MoE expert activation at batched inference, or (c) isolate length vs position vs retrieval in long-context LLMs.

**Exclusion:** blogs without an arXiv/venue ID; papers whose IDs could not be opened; adjacent work that only shares a keyword (e.g. “Union of Experts” as an architecture rename).

**PRISMA-style counts (approximate, three parallel searches):** identified ~120 unique candidate records; ~70 included after title/abstract screen; ~55 verified at the abs page and used below.

**Distributional skew (advisory):** the corpus is ≥70% 2025–2026 arXiv CS.CL/CS.LG/CS.AI, computational, US/China labs. That matches the RQ (fast-moving ML) and is not a defect.

---

## 4. Topic A — Latent Overthinking Controller (P3)

### 4.1 Landscape

Overthinking of long-CoT models is settled as a phenomenon (Chen et al., 2025; LLMThinkBench, ACL 2026 Findings, arXiv:2507.04023). Mitigation split into three crowded families:

1. **Retrain the reasoner to be shorter:** L1 (COLM 2025), ThinkPrune, Arora & Zanette (NeurIPS 2025), JET (ICLR 2026), Thinkless (NeurIPS 2025), SAGE-RL (arXiv:2602.08354).
2. **Training-free token / trial-answer heuristics:** NoWait, DEER (arXiv:2504.15895), Dynasor/Certaindex (NeurIPS 2025), REFRAIN (ACL 2026 long), ThinkBrake, RCPD, DTSR (ACL 2026).
3. **Internal-state controllers on a frozen model** — this *was* P3’s claimed gap. It is now occupied.

### 4.2 The scoop: ROM is P3

Wang, Liu, Pei, and Xiao (2026), *ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention* (arXiv:2603.22016, v3 10 Aug 2026) is the full conjunction P3 proposed:

- frozen LRM;
- lightweight hidden-state detector (~0.1% of backbone parameters);
- streaming intervention at reasoning boundaries;
- **no** intermediate-answer extraction, **no** extra probe decoding, **no** backbone updates;
- 28–77% length cut (mean 45%), Pareto-best against ten baselines, 46.5% wall-clock drop, transfer across families.

Near-isomers:

| Paper | ID | What it is | Why it is not a free lane |
|---|---|---|---|
| Zhang et al., *Reasoning Models Know When They’re Right* | 2504.05419, **COLM 2025** | Linear probe on frozen hiddens verifies intermediate answers; −24% tokens | Existence proof that “when I’m right” is linearly readable |
| LYNX (Akgül et al.) | 2512.05325 | Hidden-state probe **at cue tokens** + split conformal | Same ingredients, cue-gated, with coverage |
| ASAG | 2606.15070, **ICML 2026 Spotlight** | Training-free **attention-state** adaptive generation; +3.2% acc, ~−40% tokens | Strongest *venue* internal-state stop |
| NEAT | 2602.02010 | Training-free **neuron** tracking | Internal signal, no classifier |
| Re-FORC | 2511.02130, **now ICML 2026** | Adapter predicts reward vs future thinking tokens | Closest 2025 cousin; reward-curve, not sufficiency |
| ReProbe | 2511.06209, **ACL 2026 Main** | Hidden-state probe for step *verification*, not halt | Occupies “probe internals of frozen LLMs” |

A paper titled “latent controller that stops overthinking from hidden states” in 2027 will be read as a ROM replica.

### 4.3 Remaining white space (ordered)

1. **Harmful vs verbose overthinking.** Caldarella et al. (2026), *Thinking Past the Answer* (arXiv:2606.02835): stopping at the first-correct prefix **raises** accuracy up to 21%; standard early-stop cuts verbose tokens ~50% **and fails to stop harmful flips**. ROM/LYNX/Zhang optimize length × accuracy, not “do not leave a correct prefix.”
2. **A halt that actually terminates.** Koh et al. (2026), *</think> Doesn't Stop Reasoning* (arXiv:2609.03633, 3 Sep 2026): injecting `</think>` produces **spurious CoT termination** (reasoning continues in the answer phase). Controller + actuator is open.
3. **Per-token distribution-free halt risk.** LYNX has conformal guarantees only at cue tokens; Statistical Early Stopping (arXiv:2602.13935) has finite-sample bounds on **keywords**. A per-token hidden-state rule with a real coverage guarantee is absent.
4. **Not white space:** “early-stop CoT,” “entropy/Wait/trial-answer,” “RL to think less,” “models know when they’re right.”

### 4.4 Recommended problem statement (P3)

> **Can a frozen-LRM hidden-state probe distinguish productive self-correction from harmful overthinking (a correct prefix that will flip), and can a halt actuator that actually terminates (attention-biased EoT, not a raw `</think>` inject) reduce flip-rate at matched length versus ROM/DEER/REFRAIN?**

**Target bar:** flip-rate ↓ at matched tokens on MATH-500 / AIME / GPQA; ablate vs ROM-style FCS stop; report the spurious-termination rate from Koh et al.; easy/hard quartile breakdown.

**Venues:** ICML 2027, ACL 2027, ICLR 2028. Not ICLR 2027 (no time).

**Local code:** keep hidden-state extraction; **throw away** the “BCE stop_prob = enough tokens” label. Relabel with counterfactual prefix correctness (first-correct vs later-wrong).

---

## 5. Topic B — Batch-Aware MoE Routing (P5)

### 5.1 Landscape

The bottleneck is real and now textbook: per-token top-\(k\) stays sparse, but batched decode materializes the **union** of experts, so latency tracks unique experts, not per-token FLOPs.

That idea was independently solved four times:

| Paper | ID | Venue | What it did |
|---|---|---|---|
| Lynx | 2411.08982 (v3 May 2026) | preprint | Training-free AffinityBinning; up to 1.30×; <1 pp accuracy |
| OEA | 2511.02237 | preprint (Tri Dao et al.) | Piggyback already-loaded experts; 39%/15% **layer** latency on Qwen3-30B/235B |
| SERE | 2602.07616 | **ICLR 2026** | Similarity re-route; **vLLM CUDA kernel**, up to 2.0× decode |
| XShare | 2602.07265 | preprint (Amazon) | Batch gating-mass budget; −30% activation; vLLM RFC #35550 **closed, not planned** |

February–September 2026 then closed most of OEA’s listed future work:

| Slice (June 2026 leftover) | Status now |
|---|---|
| (a) End-to-end vLLM of batch-aware routing | **Mostly closed by SERE.** Residual is a bake-off, not an idea. |
| (b) Batch-adaptive \(k_0\) | **Still open** (narrowest algorithmic leftover). XShare uses *fixed* \(k_0\). |
| (c) Per-layer budgets | **Closed.** Alloc-MoE, ACL 2026 main (arXiv:2604.08133). |
| (d) Union-min × offload/PCIe | **Open as a composition.** HOBBIT, FineMoE, ExpertFlow, APEX, CAEE, Cache-Aware Routers (arXiv:2609.04895, 4 Sep 2026) exist; none attach Lynx/OEA/SERE union reduction to a measured miss/PCIe curve. |

ACE (arXiv:2609.05228, 4 Sep 2026) and MoDES (CVPR 2026) further crowd **token-adaptive expert skipping**. That is a different axis from batch-union, but it means “training-free MoE inference routing” is no longer a story.

### 5.2 Recommended problem statement (P5) — only if pursued

> **Does a closed-loop batch-adaptive \(k_0(B, \text{gating entropy})\) composed with union-minimization on an instrumented single-GPU offload runtime reduce PCIe expert transfers and end-to-end TPOT versus SERE/Lynx/OEA with fixed \(k_0\), at matched accuracy?**

**Target bar:** tokens/s + peak memory + PCIe bytes vs vLLM fused path **and** vs FineMoE/HOBBIT; honest negatives where dynamic routing falls off CUDA graphs.

**Venues:** MLSys 2027. Not ICLR/NeurIPS main.

**Local code:** the rewritten training-free `core.py` is on the right *mechanism* axis; it is not a contribution until it beats SERE on wall-clock.

### 5.3 Why this is the weakest of the three

Scoop risk is the highest in the portfolio. The remaining package is systems engineering. The MoEs worth testing do not fit fp16 on one 40GB A100. Custom routing often **loses** to vLLM’s fused grouped-GEMM even with fewer experts — the failure mode OEA avoided by reporting layer latency only.

---

## 6. Topic C — Long-Context Degradation Mechanism (P6)

### 6.1 Landscape

Three questions, three different occupancy levels.

**Existence (“does length hurt under perfect retrieval?”).** Closed. Du, Tian, Ronanki, Rongali, … Peng (2025), *Context Length Alone Hurts* (arXiv:2510.05381, **EMNLP 2025 Findings**): 13.9–85% drop even with whitespace filler **and** with distractors **masked** out of the softmax. Reinforced by Wang, Min, and Zou (2026, arXiv:2601.15300) and Gu (2026, arXiv:2602.15028).

**Generic training-free fixes.** Closed, and several already **beat recitation**:

| Method | ID | Gain vs recitation |
|---|---|---|
| Recitation (the anchor) | 2510.05381 | +~4% GPT-4o RULER; **adds tokens** |
| Found in the Middle | 2406.16008 | **+15 pp RAG** (subtract positional attention bias) |
| PINE | 2407.01100 (Hao Peng coauthor) | +8–10 pp; order invariance |
| **STRING** | 2410.18745, **ICLR 2025** | **+>10 pts** RULER/InfiniteBench by **remapping RoPE indices** |
| LaMPE | 2508.02308 | Training-free length-aware remap; beats SelfExtend/DCA/YaRN |

**Mechanism (dilution vs position/RoPE).** Contested, and the Peng lab has now published the theoretical answer.

| Claim | Paper | Status |
|---|---|---|
| M1: softmax entropy ~log \(n\), attention fading | α-entmax, ICLR 2026 (2506.16640); “Long Context, Less Focus” (2602.15028); qTTT ICLR 2026 (2512.13898, “score dilution”) | Formalized; **training/architecture** fixes |
| M2: RoPE fails to distinguish positions *or* tokens as length grows | **Du, Harris, Tian, … Peng, arXiv:2605.15514**, submitted **NeurIPS 2026** | Same lab as the existence paper. Theory depends **only on length**. Raising RoPE base trades token-distinction for position-distinction. |
| U-shape is architectural at **initialization**, with or without RoPE | Chowdhury, *Lost in the Middle at Birth* (2603.10123) | Causal mask + residuals, not learned Softmax/RoPE |
| Position × filler × length for *reasoning* tasks | Zhang et al., *Positional Failures* / CRE (2605.23170) | Jointly varies three factors; **not** certified-perfect-retrieval + masked softmax |

### 6.2 The 2×2 is still open — one missing cell

Nobody has published this grid under **certified-perfect retrieval** (evidence extractable verbatim, retrieval-success checked per example):

```
                    | small RoPE index          | large RoPE index
--------------------+---------------------------+---------------------------
many tokens in attn | Lost-in-the-Middle / FitM | standard long NIAH / RULER
few tokens in attn  | short-context oracle      | **MISSING CELL**
(mask/whitespace)   |                           | masked distractors, evidence
                    |                           | still at large index
                    |                           | vs remapped to small index
```

Anchor masking ≈ “few tokens, **large** index” **if** they did not reindex — they never report the crossed remapped cell. STRING/LaMPE implement the M2 lever and beat recitation **without** diagnosing M1 vs M2 under perfect retrieval. Peng 2605.15514 is length-only theory, not that grid.

That is the remaining A* slice. It is **narrower than in June 2026**, not gone.

### 6.3 Recommended problem statement (P6)

> **After certified-perfect retrieval and after distractors are removed from the softmax, does remaining accuracy still track absolute RoPE index; and does remapping the evidence span to small indices recover more accuracy per extra token than recitation, Found-in-the-Middle, PINE, and STRING?**

**Target bar:**

1. 2×2 (× filler ∈ {natural, whitespace, masked}) on ≥3 open models (Llama-3.1-8B, Qwen2.5/3-7B, Gemma-2-9B) with bootstrap CIs.
2. In the masked regime: dilution metrics from 2506.16640 held **flat** while accuracy still moves with absolute index (or does not — a clean negative is publishable).
3. Mechanism-implied intervention (index compression of the evidence span) **beats STRING and FitM** on accuracy / extra-token Pareto in the no-distractor regime. Tying recitation is not enough.

**Venues:** ACL / EMNLP analysis track; ICLR if (1)+(2)+(3) all land. ICML 2027 is the realistic first deadline.

**Local code:** the July 2026 pivot already built the 2×2 generator. That is the right experiment. What is missing is (a) a real 7B+ run, (b) an explicit **remap** cell, (c) STRING/FitM/PINE as baselines, (d) dilution instrumentation in the masked regime.

**Scoop kill-trigger:** Hao Peng lab publishes the masked count × index grid, or STRING-style remap evaluated under 2510.05381’s masking protocol.

---

## 7. Synthesis — what is still worth it

### 7.1 Cross-topic pattern

All three original headlines were **correct diagnoses of 2024–early-2025 gaps**. The field then converged:

- P3: “read hidden states, stop thinking” → ROM + COLM 2025 probe + ICML 2026 Spotlight.
- P5: “batch union kills MoE sparsity” → four independent systems + ICLR 2026 SERE.
- P6: “length hurts even with perfect retrieval” → EMNLP 2025 Findings + Peng RoPE theory (NeurIPS 2026 submission) + STRING remap (ICLR 2025).

The remaining papers are **not new phenomena**. They are **discriminating experiments** and **failure-mode controllers**.

### 7.2 Ranking for a solo / small-lab A* attempt

| Rank | Bet | Why | First deadline that is realistic |
|---|---|---|---|
| 1 | **P6 mechanism 2×2 + remap vs STRING** | Highest feasibility (no training, 1×A100), clean figure, analysis-track friendly. Code already points at the right experiment. | ICML 2027 (~22 Jan 2027) or ACL 2027 |
| 2 | **P3 harmful-overthinking probe + real halt** | Highest “so what” if it works (accuracy, not just tokens). Needs counterfactual labels and a ROM bake-off. | ICML 2027 / ICLR 2028 |
| 3 | **P5 adaptive \(k_0\) × offload** | Only MLSys-shaped. High engineering, shrinking window. | MLSys 2027 |

**Do not submit the original headlines.** Reviewers in 2027 will cite ROM, SERE, and Du et al. 2025 in the first paragraph of the review.

### 7.3 Calendar (do not ignore this)

| Venue | Abstract | Paper | Feasible from this repo today? |
|---|---|---|---|
| **ICLR 2027** | **18 Sep 2026** | **25 Sep 2026** | **No.** Eight days, smoke-only results, ICLR 2027 also rate-limits new authors. |
| EMNLP 2026 | already passed (typical June) | — | No |
| COLM 2026 | conference 6–9 Oct 2026 | — | Too late for a full paper |
| **ICML 2027** | ~16 Jan 2027 | ~22 Jan 2027 | **Yes**, if Phase 2 of P6 starts this month |
| ACL 2027 | ~Jan 2027 (indicative) | — | Yes for P6 analysis |
| NeurIPS 2026 | typically May 2026 | Dec 2026 conference | Closed |
| NeurIPS 2027 | ~May 2027 | — | Comfortable backup |
| MLSys 2027 | typically ~Dec 2026 / Jan 2027 | — | Only for P5 |

---

## 8. Devil’s advocate

### Verdict: PASS (with major caveats on P6 scoop risk and P3 label cost)

**Critical:** none that invalidate the *assessment*. The critical issue for *execution* is calendar + scoop, not logic.

**Major**

1. **P6 may still lose to Peng lab.** They own existence (2510.05381) and RoPE theory (2605.15514, NeurIPS 2026 submission). A 2×2 that only says “RoPE is weird at long \(M\)” will read as a corollary of a paper already in review. The paper has to show the **interaction** (masked × index) and beat STRING, not rediscover M2.
2. **STRING already is the M2 fix.** If remap wins, reviewers will say “this is STRING under a nicer isolation.” The contribution then has to be the **causal diagnosis**, with STRING as a baseline, not “we invented remapping.”
3. **P3 harmful-overthinking labels are the real cost.** First-correct prefixes require generating long traces and scoring every prefix. ROM already uses FCS boundaries. The new claim is **flip prediction**, which needs paired correct→wrong continuations. If the probe cannot beat a trivial “entropy went up” rule, there is no paper.
4. **P5 “remaining slice” is future-work of a precursor.** Batch-adaptive \(k_0\) is listed as open by OEA. Publishing someone’s enumerated future work, without a new analysis, is a classic reject.

**Minor**

- Preprint-heavy corpus (post-2024 arXiv). Several 2026 IDs are not yet archival; treat metrics as indicative.
- CRE (2605.23170) jointly varies position × filler × length for reasoning. A hostile reviewer can claim the 2×2 is “CRE with masking.” The distinction (certified retrieval + RoPE index vs task-in-filler) must be explicit in the intro.
- Local `test_train_p5` / `test_p6` checkpoints are from **abandoned** trained-MLP scaffolds. Do not report them as results.

**Strongest counter-argument**

> “You are recommending the leftover cells of papers that already won the venues you want. Incremental causal grids and failure-mode probes are Findings/workshop material, not ICLR/NeurIPS orals.”

That is fair for a weak execution. It is wrong if P6 produces a clean falsification figure (dilution flat, accuracy tracks index, remap recovers, STRING-style remap **fails** when the mechanism is not M2) or if P3 is the first method to cut **harmful** flip-rate. Those are the only two bars that justify the bet.

**Stress test**

| Test | Result |
|---|---|
| Remove strongest source (ROM / SERE / 2510.05381) — does the *residual* RQ still stand? | Yes for P6 2×2 and P3 harmful-flip; no for original headlines |
| Flip the question — is “these topics are fully closed” credible? | Partially: original headlines yes; residual slices no |
| “So what?” | P6: tells builders whether to remap positions or change attention. P3: tells deployers whether early-stop is safe. P5: tells serving engineers whether adaptive \(k_0\) is worth breaking the fused kernel. |

---

## 9. Ethics / integrity

- **Integrity:** CLEARED. Secondary literature analysis; no human subjects; no dual-use operational detail.
- **Attribution:** all load-bearing IDs were opened on arXiv or venue pages this session. Author lists for a few 2026 papers are taken from abs pages; fill remaining author lists from PDF before any public citation dump.
- **Conflicts:** this assessment re-uses the user’s own June 2026 research docs as a starting corpus and then updates them. That is disclosed; it is not an independent systematic review registered on PROSPERO (nor does the field require one).

---

## 10. Next actions (if you accept the ranking)

**This week (decision + freeze RQs)**

- [ ] Accept: **P6 flagship, P3 pivot, P5 park.**
- [ ] Do **not** register an ICLR 2027 abstract on 18 Sep unless a paper already exists (it does not).
- [ ] Freeze RQ-P6 and RQ-P3 as written in §2.1.

**P6 (weeks 1–8) — the actual experiment**

- [ ] Run the existing 2×2 on Qwen2.5-1.5B (T4) then 7B/8B at 8K–32K (A100).
- [ ] Add the **remap** cell (evidence RoPE indices compressed to the short-context range) and STRING/FitM/PINE/recitation baselines.
- [ ] Log attention entropy/dispersion in masked vs natural filler (M1 falsification).
- [ ] Kill-switch: weekly arXiv search on Peng + “context length” + “RoPE” + “mask.”

**P3 (parallel, after P6 Phase 1 is in flight)**

- [ ] Relabel traces with first-correct vs later-wrong prefixes (the 2606.02835 protocol).
- [ ] Probe = flip predictor, not stop-from-length.
- [ ] Halt = EAB-style attention bias (2609.03633), not raw `</think>`.
- [ ] Bake off ROM/DEER/REFRAIN on flip-rate, not only tokens.

**P5**

- [ ] Leave the training-free hooks as infrastructure. Do not schedule GPU time until P6/P3 have a figure.

---

## References (load-bearing; APA-ish, arXiv IDs verified)

Caldarella, S., Talon, D., Aljundi, R., Ricci, E., & Mancini, M. (2026). *Thinking past the answer: Evaluating harmful overthinking in large reasoning models*. arXiv:2606.02835.

Chowdhury, B. D. (2026). *Lost in the middle at birth: An exact theory of transformer position bias*. arXiv:2603.10123.

Du, Y., Tian, M., Ronanki, S., Rongali, S., Bodapati, S., Galstyan, A., Wells, A., Schwartz, R., Huerta, E. A., & Peng, H. (2025). *Context length alone hurts LLM performance despite perfect retrieval*. arXiv:2510.05381. EMNLP 2025 Findings.

Du, Y., Harris, P., Tian, M., Huerta, E. A., Ronanki, S., Rongali, S., Galstyan, A., & Peng, H. (2026). *RoPE distinguishes neither positions nor tokens in long contexts, provably*. arXiv:2605.15514.

Hsieh, C.-Y., Chuang, Y.-S., Li, C.-L., Wang, Z., Le, L. T., Kumar, A., Lee, C.-Y., et al. (2024). *Found in the middle: Calibrating positional attention bias improves long context utilization*. arXiv:2406.16008.

An, C., Zhang, J., Zhong, M., Li, L., Gong, S., Luo, Y., Xu, J., & Kong, L. (2024). *Why does the effective context length of LLMs fall short?* arXiv:2410.18745. ICLR 2025. (STRING)

Koh, S., Choi, S., Kwon, M., Baek, S., & Kim, J. (2026). *</think> doesn't stop reasoning: Analysis of spurious CoT termination*. arXiv:2609.03633.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the middle: How language models use long contexts. *Transactions of the ACL*. arXiv:2307.03172.

Oncescu, C.-A., Wu, Q., Chung, W. T., Wu, R., Gopal, B., Wang, J., Dao, T., & Athiwaratkun, B. (2025). *Opportunistic expert activation: Batch-aware expert routing for faster decode without retraining*. arXiv:2511.02237.

Wang, X., Liu, X., Pei, M., & Xiao, C. (2026). *ROM: Real-time overthinking mitigation via streaming detection and intervention*. arXiv:2603.22016.

Wang, Z., Zhang, H., Li, X., Huang, K.-H., Han, C., Ji, S., Kakade, S. M., Peng, H., & Ji, H. (2024). *Eliminating position bias of language models: A mechanistic approach* (PINE). arXiv:2407.01100.

Wu, J., Cheng, J., Lv, F., Ou, D., & Yuan, L. (2026). *SERE: Similarity-based expert re-routing for efficient batch decoding in MoE models*. arXiv:2602.07616. ICLR 2026.

Zhang, A., Chen, Y., Pan, J., Zhao, C., Panda, A., Li, J., & He, H. (2025). *Reasoning models know when they’re right: Probing hidden states for self-verification*. arXiv:2504.05419. COLM 2025.

Zhang, S., Gao, G., Gan, Z., Yuan, C., Lin, Z., Peng, H., Li, B., & Hu, W. (2025). *LaMPE: Length-aware multi-grained positional encoding for adaptive long-context scaling without training*. arXiv:2508.02308.

Gupta, V., Ju, J. H., Sinha, K., Gavrilovska, A., & Iyer, A. P. (2024/2026). *Lynx: Enabling efficient MoE inference through dynamic batch-aware expert selection*. arXiv:2411.08982 (v3 May 2026).

Vankov, D., Ivkin, N., Ulrich, K., Song, X., Khetan, A., & Karypis, G. (2026). *XShare: Collaborative in-batch expert sharing for faster MoE inference*. arXiv:2602.07265.

Liu, B., Tian, K., Wang, W., Zhang, Z., Qiao, L., & Li, D. (2026). *Alloc-MoE: Budget-aware expert activation allocation for efficient mixture-of-experts inference*. arXiv:2604.08133. ACL 2026.

Xu, Z., Zhao, Z., Hu, X., Yu, J., Wen, H., Li, J., Jiang, Z., & Yang, D. (2026). *ACE: Adaptive calibration-free expert skipping for MoE-based LLMs*. arXiv:2609.05228.

Li, J., Qin, K., Wang, R., Ma, Y., Chen, Q., Li, M., & Liang, S. (2026). *Stop when further reasoning won’t help* (ASAG). arXiv:2606.15070. ICML 2026 Spotlight.

Han, J., Huang, Y., Liao, Y., et al. (2025/2026). *Your models have thought enough: Training large reasoning models to stop overthinking* (JET). arXiv:2509.23392. ICLR 2026.

Sun, R., Cheng, W., Li, D., Chen, H., & Wang, W. (2025). *Stop when enough: Adaptive early-stopping for chain-of-thought reasoning* (REFRAIN). arXiv:2510.10103. ACL 2026 long.

Bansal, R., Zhang, A., Tiwari, R., et al. (2025). *Let’s (not) just put things in context: Test-time training for long-context LLMs*. arXiv:2512.13898. ICLR 2026.

Vasylenko, P., Pitorro, H., Martins, A. F. T., & Treviso, M. (2025). *Long-context generalization with sparse attention*. arXiv:2506.16640. ICLR 2026.

Zhang, C., Cui, H., Huang, X., & Sang, J. (2026). *Positional failures in long-context LLMs: A blind spot in reasoning benchmarks*. arXiv:2605.23170.

Akgül, Ö. F., Kalaycı, Y. H., Kannan, R., Neiswanger, W., & Prasanna, V. (2025). *LYNX: Learning dynamic exits for confidence-controlled reasoning*. arXiv:2512.05325.

Zabounidis, R., Golatkar, A., Kleinman, M., Achille, A., Xia, W., & Soatto, S. (2025). *Re-FORC: Adaptive reward prediction for efficient chain-of-thought reasoning*. arXiv:2511.02130. ICML 2026.

Yang, C., Si, Q., et al. (2025). *Dynamic early exit in reasoning models* (DEER). arXiv:2504.15895.

Srivastava, et al. (2025). *Do LLMs overthink basic math reasoning?* (LLMThinkBench). arXiv:2507.04023. ACL 2026 Findings.

Wang, W., Min, J., & Zou, W. (2026). *Intelligence degradation in long-context LLMs: Critical threshold via natural length distribution*. arXiv:2601.15300.

Gu, S. (2026). *Long context, less focus: A scaling gap revealed through privacy and personalization*. arXiv:2602.15028.

Hsieh, C.-P., Sun, S., et al. (2024). *RULER: What’s the real context size of your long-context LMs?* arXiv:2404.06654. COLM 2024.

Kuratov, Y., Bulatov, A., et al. (2024). *BABILong: Testing the limits of LLMs with long-context reasoning-in-a-haystack*. arXiv:2406.10149. NeurIPS 2024.

*Fuller annotated maps:* `common/research/p03-research.md`, `p05-research.md`, `p06-research.md` (June 2026). This document supersedes their **verdicts** as of 2026-09-10; it does not replace their paper-by-paper notes.

---

*Last updated: 2026-09-10. Search executed 2026-09-10. ICLR 2027 dates from iclr.cc/Conferences/2027/Dates.*
