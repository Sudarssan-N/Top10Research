# P6 Problem Statement — Complete Research Backing

**Date:** 2026-09-10  
**Mode:** Deep research (Phase 1 scoping + Phase 2 verification of load-bearing papers)  
**Project:** `p06-long-context-degradation/`  
**Target venues:** ICML 2027 (~22 Jan 2027) / ACL 2027. Not ICLR 2027 (18/25 Sep 2026).

**AI disclosure:** Literature search and synthesis were AI-assisted (Grok 4.6, 2026-09-10). Load-bearing claims about Du et al. (arXiv:2510.05381), An et al. STRING (arXiv:2410.18745, ICLR 2025), and Du & Peng RoPE theory (arXiv:2605.15514) were checked against the HTML full text of those papers, not abstracts alone.

---

## 1. Why the previous statement was incomplete

The June 2026 / 10 Sep 2026 drafts framed the gap as:

> 2×2 `{few vs many tokens in attention} × {evidence at small vs large absolute position}` under perfect retrieval.

That is directionally right and still unpublished as a full factorial. It is **not precise enough**, because it conflates three quantities Du et al. actually separated:

| Quantity | What it is | Du et al. control |
|---|---|---|
| **Softmax support** \(n\) | How many tokens enter the softmax | Essay vs whitespace vs **mask** |
| **Relative distance** \(m\) | RoPE product \(S(m)=q^\top R_m k\) between question (query) and evidence (key) | Evidence at start → large \(m\); evidence moved next to question → small \(m\) |
| **Absolute index of the query** | Where the question sits in the sequence | Grows with length in every Du layout |

RoPE is a **relative** encoding: \(q_i^\top k_j = q^\top R_{i-j} k\). Absolute index of the evidence does not appear in the score except through \(m = i_{\text{query}} - j_{\text{key}}\) **and** through whatever the query rotation \(R_i\) does to \(q\) before the product. Treating “absolute evidence offset” as the mechanism axis (the current `core.py` 2×2) therefore mixes \(m\) with query index and never implements Du’s mask.

**What Du et al. actually ran** (verified in §4.1–4.2 of arXiv:2510.05381):

| Du condition | Layout | Softmax \(n\) | Rel. dist. \(m\) | Query abs. index | Result |
|---|---|---|---|---|---|
| Essay / default | `[E][essay][Q]` | many | large | large | Accuracy collapses; retrieval EM stays high |
| Whitespace, evidence start | `[E][WS][Q]` | many (WS) | large | large | Still drops (Fig. 4a) |
| Whitespace, evidence **end** | `[WS][E][Q]` | many (WS) | **small** (held fixed as length grows) | large | **Still drops** (Fig. 4b, up to 17–20% at 30K) |
| **Mask** | `[E][MASK][Q]` | **few** (only E+Q) | **large** | large | **Still drops** (Table 3, ≥7.9% at 30K; HumanEval −50%) |
| Recitation | retrieve then new short prompt | few | small | small | Recovers (their fix) |

They never ran the mask with evidence **next to** the question:

```
[MASK][Evidence][Question]
```

That cell is **few tokens in softmax, small relative distance, large absolute query index**. It is the one comparison that can tell M1 (dilution) from M2-relative (RoPE distance) from M2-absolute / representation drift (query sitting at a large index).

They also never crossed that cell with STRING-style **index remapping**.

---

## 2. Research Question Brief

### Topic area

Mechanism of length-induced degradation in decoder-only LLMs under certified-perfect retrieval, after Du, Tian, Peng et al. (EMNLP 2025 Findings).

### Primary research question

> After retrieval is certified perfect and distractors are removed from the softmax, does remaining accuracy still track the RoPE **relative distance** between evidence and question, the **absolute index of the query**, or neither — and does the intervention implied by that answer beat recitation, Found-in-the-Middle, PINE, and STRING at matched decode tokens on reasoning tasks?

### FINER assessment

| Criterion | Score | Justification |
|---|---|---|
| Feasible | 5/5 | Open 7–9B models, attention masking, and RoPE-index overwrite are all inference-time; 1×A100; Du already showed the protocol is runnable. |
| Interesting | 5/5 | Three published theories now disagree on *which* quantity Du measured. A single factorial can kill two of them. |
| Novel | 4/5 | Existence is closed; STRING is closed for *retrieval* NIAH; Peng 2605.15514 is closed for *theory*. The mask-end cell and the reasoning-task remap ablation are not. Novelty is conditional on running those cells, not on “length hurts.” |
| Ethical | 5/5 | No human subjects; public models and synthetic/public benchmarks. |
| Relevant | 5/5 | Tells builders whether to remap relative indices (STRING), compress the query’s absolute index, change attention, or just recite. Directly answers Peng’s NeurIPS 2026 submission with an experiment it does not contain. |
| **Average** | **4.8/5** | |

### Scope

**In scope**

- Open decoder-only RoPE models (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it; optional Mistral-v0.3-7B to match Du).
- Certified-perfect-retrieval **reasoning** tasks cloned from Du (GSM8K, MMLU subset, VarSum) plus a retrieval probe (RULER NIAH / KV registry) as a **control**, not the headline.
- Filler ∈ {natural essay, whitespace, attention-mask}.
- Layouts that independently set softmax \(n\), relative \(m\), query absolute index.
- Training-free interventions: recitation, FitM, PINE, STRING, and the intervention implied by the winning cell.
- Lengths 0 / 4K / 8K / 16K / 32K (40GB A100 cap); 64K only if 80GB.

**Out of scope**

- Training a new positional encoding (that is Peng’s stated future work).
- Claiming “we discovered length hurts” or “we discovered Lost-in-the-Middle.”
- Agent/RAG pipelines except as discussion.
- Closed models (cannot mask or remap RoPE).

**Assumptions**

- Transformers apply RoPE as relative \(R_{i-j}\) (true for Llama/Qwen/Gemma).
- Attention masks zero the distractor keys **without collapsing positions** (standard HF `attention_mask` / custom key mask). Positions of surviving tokens stay at their original indices unless we remap.
- “Perfect retrieval” means exact-match recitation of the evidence span on a **separate** probe, following Du §3.1, not “the model used the evidence.”

### Sub-questions

1. **SQ1 (existence, replication).** Does Du’s mask drop (`[E][MASK][Q]`) and Du’s evidence-at-end whitespace drop (`[WS][E][Q]`) replicate on the three open models at 8–32K?
2. **SQ2 (missing cell).** In `[MASK][E][Q]`, is the mask drop gone? If yes, dilution (M1) caused Fig. 4b. If no, something other than softmax \(n\) and relative \(m\) is causal.
3. **SQ3 (remap).** Does STRING-style shifting of relative indices recover `[E][MASK][Q]` (large \(m\)) without recovering `[MASK][E][Q]` (small \(m\), large query index) — or the reverse?
4. **SQ4 (fix).** Does the mechanism-implied intervention beat recitation / FitM / PINE / STRING on **GSM8K-style reasoning** (not only NIAH) at matched extra decode tokens?

### Sub-question bindings

1. inherits: models=open RoPE 7–9B; timeframe=2026-09–2027-01; domain=perfect-retrieval long context; deviations: none
2. inherits: same; deviations: none
3. inherits: same; deviations: none
4. inherits: same; deviations: none

### Candidates considered

| # | Candidate | FINER avg | Why not selected |
|---|---|---|---|
| 1 | Primary RQ above | 4.8 | Selected |
| 2 | “Does length hurt under perfect retrieval?” | 2.4 | Answered by Du (EMNLP 2025 Findings) |
| 3 | “Does STRING beat recitation on RULER?” | 2.6 | STRING already +>10 on RULER (ICLR 2025) |
| 4 | “Is RoPE theoretically broken at long \(M\)?” | 3.0 | Peng 2605.15514, NeurIPS 2026 submission |

---

## 3. Competing mechanisms (must be named, then killed)

Four stories are live in 2026. They are **not** equivalent.

| ID | Claim | Predicts Du `[E][MASK][Q]` drop? | Predicts Du `[WS][E][Q]` drop? | Predicts **missing** `[MASK][E][Q]` drop? | Canonical papers |
|---|---|---|---|---|---|
| **M1 Dilution** | Softmax entropy ~log \(n\); extra keys flatten mass on evidence | **No** (those keys are masked) | **Yes** (whitespace still in softmax) | **No** | α-entmax ICLR 2026 (2506.16640); Gu 2602.15028; qTTT ICLR 2026 (2512.13898, “score dilution”) |
| **M2-rel RoPE distance** | \(S(m)\) becomes unpredictable as relative distance grows; locality bias dies | **Yes** (large \(m\)) | **No** (small \(m\) held fixed) | **No** | Peng et al. 2605.15514; Base of RoPE 2405.14591 |
| **M2-abs / query index** | Query sitting at a large absolute index (or hidden-state drift with sequence position) hurts even when \(m\) and \(n\) are small | **Yes** (query at end) | **Yes** (query at end) | **Yes** | Du Fig. 4b *if* it survives masking; Silent Tokens/padding 2510.01238 (padding shifts activations) |
| **M3 Architectural U-shape** | Causal mask + residuals create primacy/recency at init, independent of RoPE | Weak (start and end should be *good*) | Weak (end should be *good*) | Weak | Lost in the Middle (TACL 2024); Chowdhury 2603.10123; Attention Basin 2508.05128 |

**Critical observation from Du, which the old P6 statement under-used:**

Du’s drop happens when evidence is at the **start** *and* when it is at the **end**. That is **not** Lost-in-the-Middle (U-shape: start/end good, middle bad). M3 is therefore a poor explanation of *length-alone* degradation. P6 should treat Lost-in-the-Middle as a **different** phenomenon and not use middle-position sweeps as the headline (the current `evidence_offsets` grid is closer to Lost-in-the-Middle than to Du).

**Critical observation about STRING:**

STRING (An et al., ICLR 2025) overwrites **large relative positions** with well-trained small ones so the model can gather *distant* keys. It is the natural fix for **M2-rel**. It is **not** obviously the fix for Du Fig. 4b, where \(m\) is already small. If `[MASK][E][Q]` still fails, STRING should **not** recover it, and recitation should. That is a publishable disagreement with “just apply STRING.”

---

## 4. Discriminating factorial (the actual contribution)

Five layouts. Retrieval certified on every example (separate recitation probe, exact match, Du protocol). Task = GSM8K / VarSum (reasoning), with NIAH as control.

```
Layout A  short oracle     [E][Q]
Layout B  Du mask          [E][MASK][Q]          few n, large m, large |q|
Layout C  Du end+WS        [WS][E][Q]            many n, small m, large |q|
Layout D  MISSING cell     [MASK][E][Q]          few n, small m, large |q|
Layout E  remap of B       [E][MASK][Q] + STRING few n, small m (fake), mixed |q|
```

Optional Layout F: remap of D (compress query absolute index without changing \(m\)), only if D fails.

### Predicted pattern (preregister this table)

| If the truth is… | A | B | C | D | E (STRING on B) |
|---|---|---|---|---|---|
| M1 dilution only | high | high | low | high | ≈B (no help) |
| M2-rel only | high | low | high | high | recovers B |
| M2-abs / query index | high | low | low | **low** | does **not** recover D; may help B |
| M1+M2-rel (mixture) | high | low | low | high | recovers B, not C |
| M3 U-shape | high | med | high | high | irrelevant |

Du already reported A high, B low, C low. **D and E are the paper.**

A clean paper is: replicate B and C (SQ1), run D (SQ2), run E (SQ3), then one implied fix vs recitation/FitM/PINE/STRING on GSM8K-at-length (SQ4).

---

## 5. Load-bearing literature (verified against full text)

### 5.1 Existence and protocol (must cite; must not claim as ours)

**Du, Tian, Ronanki, Rongali, Bodapati, Galstyan, Wells, Schwartz, Huerta, Peng (2025).** *Context Length Alone Hurts LLM Performance Despite Perfect Retrieval.* arXiv:2510.05381. **EMNLP 2025 Findings.**

- Perfect retrieval = separate exact-match recitation of evidence, not NIAH accuracy.
- Tasks: GSM8K, MMLU, HumanEval, VarSum. Models: Llama-3.1-8B, Mistral-v0.3-7B, GPT-4o, Claude-3.5/3.7, Gemini-2.0.
- Masking: “the model attends *only* to the evidence and the question, identical to the short-context setting **except for the increased distance between them**.”
- Evidence-at-end: “so that the distance does not change with input size” — drop remains.
- Fix: recite then solve on the short prompt; +~4% GPT-4o RULER QA; +~31% Mistral GSM8K with essay filler.
- Limitation they state: two open models in the mask table; recitation needs retrieval to succeed.
- **What they leave open (their §6):** the two-part retrieval/reasoning decomposition is incomplete; “underlying mechanisms” are unexplored. They cite An et al. 2024 (STRING) as training-time position-frequency bias, without running STRING under masking.

### 5.2 Mechanism theory (must beat, not repeat)

**Du, Harris, Tian, Huerta, Ronanki, Rongali, Galstyan, Peng (2026).** *RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably.* arXiv:2605.15514. **NeurIPS 2026 submission.** Same lab.

- Theory depends **only on length**, not content. RoPE product ≈ normal; position inversion and aliasing → 0.5 as \(M\) grows.
- Raising RoPE base trades token-distinction for position-distinction.
- Empirical: \(k\)-th item in a 4-value list → chance by ~4K tokens.
- Recommended workaround: do **not** extend context; chunk / agentic short contexts.
- **Does not contain** the mask-end cell, STRING ablation, or GSM8K-under-mask.

This paper **raises** P6’s bar: a 2×2 that only says “large \(m\) is bad” will read as a corollary. P6 must show **which of \(n\), \(m\), \(|q|\)** Du measured, with Layout D.

### 5.3 Relative-index remap (must be a baseline, not the claim)

**An, Zhang, Zhong, Li, Gong, Luo, Xu, Kong (2024/2025).** *Why Does the Effective Context Length of LLMs Fall Short?* arXiv:2410.18745. **ICLR 2025.** STRING / StRing.

- Cause: left-skewed **relative position frequency** in pretraining (\(f(i)=L-i\) times data-length skew).
- Fix: drop infrequent large relative indices; shift well-trained small indices into the distant triangle; keep a local window \(W\). FlashAttention implementation, **no extra decode tokens**.
- +>10 RULER / InfiniteBench on Llama-3.1-70B and Qwen2-72B; Llama-3.1-70B+STRING > GPT-4-128K.
- Evaluated on **NIAH / RULER / InfiniteBench**, i.e. gathering distant needles — exactly M2-rel.
- **Not** evaluated under Du masking, not on GSM8K-at-length with perfect retrieval, not on `[MASK][E][Q]`.

Sibling remaps: LaMPE (2508.02308), Self-Extend, DCA, ReRoPE, MrRoPE (2601.22181). STRING is the strongest published training-free remap; it is the baseline E.

### 5.4 Dilution / score-dilution (M1; the hypothesis Layout D kills)

- Vasylenko et al., α-entmax, arXiv:2506.16640, **ICLR 2026** — softmax entropy ~log \(n\).
- Bansal et al., *Let’s (not) just put things in context*, arXiv:2512.13898, **ICLR 2026** — “score dilution”; thinking tokens cannot recover buried targets; **qTTT** (train-time at test) as alternative. Training, not P6’s training-free lane.
- Gu, *Long Context, Less Focus*, arXiv:2602.15028 — asserts dilution without Du’s mask control.

If Layout D recovers, M1 explains Fig. 4b and Peng’s relative-distance story is **not** necessary for that figure. If D fails, M1 is insufficient (Du’s mask already said this for layout B; D says it for small \(m\)).

### 5.5 Position bias that is *not* our phenomenon

- Liu et al., Lost in the Middle, TACL 2024, arXiv:2307.03172 — U-shape at **fixed** length.
- Hsieh et al., Found in the Middle, arXiv:2406.16008 — subtract positional attention bias; +15 pp RAG.
- Wang, Zhang, … Peng, Ji, PINE, arXiv:2407.01100 — causal mask + relative PE; bidirectional inter-document attention.
- Chowdhury, Lost in the Middle at Birth, arXiv:2603.10123 — U-shape at **step 0**, with or without RoPE.
- Zhang et al., Positional Failures / CRE, arXiv:2605.23170 — jointly varies task position × filler × length for **reasoning-in-filler**; not certified retrieval + mask.
- Yi et al., Attention Basin, arXiv:2508.05128 — U-shape of attention over document lists.

Cite as related; do **not** make a middle-position sweep the main figure.

### 5.6 Padding / silent tokens (supports M2-abs)

Himelstein et al., *Silent Tokens, Loud Effects: Padding in LLMs*, arXiv:2510.01238, NeurIPS 2025 workshop. Padding that should be masked still shifts activations and quality. Relevant to whether a masked prefix is truly a no-op on the residual stream (LayerNorm, sinks). **P6 must log residual norms and sink mass in Layout D**, not assume the mask makes the prefix invisible to everything except RoPE.

---

## 6. Methodology Blueprint

### Research paradigm

**Selected:** positivist / confirmatory causal experiment.  
**Justification:** the RQ is “which of three measurable factors causes the residual drop?” — a factorial with preregistered cell predictions.

### Method

**Type:** quantitative.  
**Specific method:** controlled factorial experiment on frozen LLMs (inference-time interventions only), plus a planned contrast against published training-free baselines.

### Data strategy

**Primary:** synthetic long prompts built from short GSM8K / VarSum / MMLU items by inserting filler between or before a contiguous evidence span, following Du Fig. 2. Gold evidence always a single consecutive chunk.

**Retrieval certification:** for every example, a separate greedy recitation of the evidence span; keep only items with exact match (or report both “all items” and “retrieval-certified subset,” as Du does).

**Sampling:** \(n \geq 200\) certified items per cell per model at each length (Du used full test sets; 200 is the A100-feasible floor with CIs). Seeded so cells share the same items and differ only in layout/filler.

**Lengths:** \(\{0, 4096, 8192, 16384, 32768\}\) tokens of filler.

### Analytical framework

1. Per-cell accuracy with bootstrap 95% CIs (already in `phase1.py`).
2. Planned contrasts: \(A-B\) (Du mask), \(A-C\) (Du end), \(A-D\) (**new**), \(E-B\) (STRING on mask).
3. Mechanism instrumentation on a 20-item probe set: attention entropy / evidence mass (M1), RoPE product \(S(m)\) vs \(m\) (M2-rel), residual \(\ell_2\) at the last evidence token and at the query (M2-abs), first-token sink mass (M3).
4. Mitigation Pareto: accuracy vs extra generated tokens (recitation costs tokens; STRING/FitM/PINE/mask do not).

**Preregister** the prediction table in §4 before looking at Layout D.

### Validity

| Criterion | Strategy |
|---|---|
| Internal | Orthogonal layouts; retrieval certified per item; same items across cells |
| Construct | Separate \(n\), \(m\), \(|q|\); do not call “evidence offset” the mechanism |
| External | ≥3 model families; reasoning + retrieval control |
| Statistical | Bootstrap CIs; pre-specify \(n\); report negatives |
| Reproducibility | Open models, seeded generator, released grid JSON |

### Limitations (by design)

- Masking is not available on closed APIs — cannot claim GPT-4o Layout D.
- GSM8K-at-32K is still synthetic (Du’s own limitation).
- STRING hyperparameters (\(S=L/3\), \(W=128\)) copied from An et al.; ablation is secondary.
- If Layout D fails because LayerNorm/sinks leak the masked prefix (Himelstein), that is a result, not a confound to hide.

### Ethics

No human subjects. `review pathway: institutional determination required` is N/A (no human-subjects activity). Dual-use: none.

### Reporting standard

Follow empirical-ML convention (ICML/ACL): preregistered contrasts, full cell table, negative results.

### Preregistration

**Recommended: Yes.** OSF or a public GitHub `protocol.md` freeze before Layout D runs. Status: planned. Artifact declaration: `not_provided`.

---

## 7. What the current scaffold gets wrong (and what to keep)

| Piece | Status | Action |
|---|---|---|
| `core.py` KV-registry 2×2 `{total tokens} × {evidence offset}` | Implements Lost-in-the-Middle-style position sweep on a **retrieval** toy | Keep as a **control** NIAH; do **not** make it the headline. Headline tasks = Du GSM8K/VarSum. |
| `configs/default.yaml` relative positions `[0.0, 0.25, 0.5, 0.75, 1.0]` vs `config.py` absolute `[64, 768]` | **Inconsistent** | Unify on the five layouts A–E, not relative depth. |
| Mitigation = `none \| recite` only | Missing STRING, FitM, PINE, mask, remap | Add. |
| No attention mask of distractors | Missing Du Table 3 | Add `filler_mode ∈ {essay, whitespace, mask}`. |
| No `[MASK][E][Q]` | The missing cell | **Build this first.** |
| Paper abstract still says “2×2 few/many × small/large absolute position” | Too coarse | Replace with the \(n \times m \times |q|\) factorial. |
| Smoke test plants a position-only failure | Fine for CI plumbing | Keep. |

---

## 8. Target bar (what counts as an A* paper)

**Minimum (ACL/EMNLP analysis / ICML poster):**

1. Replicate Du B and C on ≥2 open models (SQ1).
2. Layout D with CIs (SQ2).
3. Honest mechanism sentence: which of M1 / M2-rel / M2-abs survives.

**Full (ICML / ACL oral range / ICLR 2028):**

4. Layout E: STRING on B, and STRING-or-index-compress on D (SQ3).
5. Head-to-head on GSM8K-at-length: none / recitation / FitM / PINE / STRING / implied fix, **accuracy vs extra tokens** (SQ4).
6. Instrumentation figure: entropy flat in D while accuracy still drops (or does not).

Tying recitation is **not** enough. Beating STRING on NIAH is **not** enough (STRING already owns that). Beating STRING **or showing STRING cannot fix D** on Du-style reasoning is the contribution.

---

## 9. Devil’s advocate

**Verdict: REVISE the old 2×2 language; PASS the Layout-D RQ.**

**Major**

1. **Peng 2605.15514 will be in NeurIPS 2026 review now.** Reviewers will have it. If P6 only shows “large \(m\) hurts,” they will say corollary. Layout D is the only defense.
2. **STRING authors can claim E is their method on a new task.** Then the paper’s novelty is D’s diagnosis, not a new remap. Write the intro that way from day one.
3. **Layout D may be a dud.** If D recovers fully, the paper becomes “Fig. 4b was dilution, mask it.” That is still publishable (ACL analysis) but smaller. Have the NIAH control and STRING bake-off so there is a second figure.
4. **Current code cannot answer the RQ.** Running the existing grid and putting numbers in `main.tex` would be answering a different, already-published question.

**Strongest counter-argument**

> “Du already told you masking does not save you, and STRING already remaps. You are filling in a table.”

Answer: Du’s mask holds \(m\) large; Du’s end-condition holds \(n\) large. Nobody crossed **small \(n\) and small \(m\) at large \(|q|\)**. STRING remaps \(m\), not \(|q|\), and was not tested on Du’s reasoning-under-perfect-retrieval protocol. That crossed cell is the only thing that can decide whether recitation is a hack or the uniquely correct fix.

---

## 10. Frozen problem statement (use this everywhere)

**Title (working):** *Which Length Is It? Dilution, Relative Distance, and Query Index under Perfect Retrieval*

**One sentence:**  
Length still hurts after perfect retrieval because prior controls never independently set softmax support, RoPE relative distance, and the query’s absolute index; we run the missing mask-at-end cell and the STRING remap of Du’s mask, and we take as the contribution whichever intervention that cell implies, evaluated against recitation, Found-in-the-Middle, PINE, and STRING on reasoning tasks.

**Do not say:** “we show length hurts”; “we propose reordering”; “we propose recitation”; “we propose a 2×2 of token count × evidence position” (too coarse).

---

## 11. Immediate implementation order (research, then code)

1. Freeze this protocol (this file).
2. Rewrite generator: Du layouts A–D, GSM8K/VarSum evidence/question split, retrieval-certification probe.
3. Implement mask (keys of filler zeroed, positions **not** collapsed).
4. Replicate B and C on Qwen2.5-1.5B then 7B.
5. Run D — this is the result that decides the paper.
6. Implement STRING (An et al. Algorithm 1) as Layout E.
7. FitM / PINE / recitation bake-off on the same items.
8. Only then write results into `paper/main.tex`.

Kill-switch: weekly arXiv alert on Peng + “mask” + “perfect retrieval.” If someone publishes Layout D, pivot to SQ4 only or stop.
