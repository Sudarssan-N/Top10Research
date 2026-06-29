# P4 Research Foundation — Test-Time Compute vs. Confidence Calibration + Corrective Probe

**Problem:** Does increasing the reasoning/test-time-compute budget help or hurt confidence calibration (ECE/Brier/selective risk), and can a lightweight post-hoc probe recover calibration without retraining the base model?

**Compute:** 1×A100-40GB / Colab Pro+ (~80–150 GPU-hrs)
**Venues:** ICLR / COLM / EMNLP (analysis track)
**Project:** `p04-calibration-probe/`
**Parent index:** [`../../research.md`](../../research.md) §P4
**Priority:** Rank-3 hedge / low-risk analysis paper

---

### TL;DR Verdict

**On track, but PIVOT THE FRAMING — do not sell the descriptive half as the contribution.** The phenomenon "more reasoning / longer CoT can worsen calibration and reasoning models stay overconfident" is, as of mid-2026, **largely already established** (anchor arXiv:2508.15050 plus a wave of 2025–2026 RL-calibration papers — RLCR 2507.16806, DCPO 2603.09117, C²GSPG 2509.23129). The lightweight-hidden-state-probe-for-correctness idea is **also already occupied** — arXiv:2504.05419 ("Reasoning Models Know When They're Right") trains exactly such a probe, reports it produces "highly calibrated scores," and uses it for early-exit. So P4's two headline ingredients each have a strong incumbent. **The defensible white space is the *synthesis nobody has cleanly done*: a controlled budget-as-independent-variable calibration audit across 3–4 open reasoning models on math/science reasoning (MATH-500 / AIME / GPQA), paired with a *budget-conditioned, budget-transferable* post-hoc recalibration map evaluated on selective risk (risk–coverage / AURC) at low-FPR operating points — under an explicit no-retrain constraint.** Novelty: **medium** (the delta is the controlled axis + selective-risk framing + frozen-model constraint, not the discovery). Feasibility: **very high** (pure inference + a <1M-param probe). Biggest risk: **being scooped on the analysis** and/or a **null/ noisy budget→ECE trend on math** (web evidence already shows the relationship is non-monotonic and model-dependent, so the clean "more compute → worse calibration" story may not replicate outside the anchor's climate-QA domain). Verdict: **keep as the hedge, re-aim the contribution at the recalibration map + selective risk, and ship fast.**

---

### Problem Statement & Framing

> Does increasing reasoning budget help or hurt calibration, and can a lightweight probe fix it — without retraining the base model?

**Two coupled questions:**

1. **Descriptive (audit):** Treat test-time compute as a *controlled independent variable* — vary (a) the thinking-token budget (`max_new_tokens` / explicit budget forcing / thinking on-off), (b) the number of samples (self-consistency width), and measure calibration (ECE, adaptive-ECE, Brier, NLL) and **selective risk** (risk–coverage curve, AURC, accuracy@coverage) as the *dependent* variables, across 3–4 open reasoning models and 3–4 reasoning benchmarks.
2. **Corrective (probe):** Can a small frozen-model probe (on last-token / mean-pooled hidden states and/or raw self-confidence signals) **recalibrate** the model's confidence — and crucially does the *optimal recalibration map shift with budget*, so the probe must be **budget-conditioned** to stay calibrated as compute changes? Evaluate at **low-FPR / high-precision selective operating points** (the deployment-relevant regime), not just average ECE.

**Why the no-retrain constraint is the load-bearing framing choice.** The strongest competitor (RLCR, 2507.16806) is a *training-time* RL method that explicitly reports beating "classifiers trained to assign post-hoc confidence scores." P4 cannot win a head-to-head with RLCR on raw calibration if retraining is allowed. P4's reason to exist is the **served-model reality**: you have a frozen API/checkpoint reasoning model, you cannot RL-finetune it, and you need to bolt calibration on at inference for a few hundred dollars. That constraint is real, common, and under-served — but it must be stated as the contract, or a reviewer will ask "why not just do RLCR?"

**Target bar (re-scoped to be honest):**
- Audit: a clean, reproducible budget×model×benchmark grid of ECE/Brier/AURC — a *result*, not a SOTA claim.
- Probe: beat (i) raw softmax/sequence confidence, (ii) global temperature scaling, (iii) verbalized confidence (Tian-style "just ask"), at **matched compute**, on ECE **and** selective risk; show the **budget-conditioned** probe transfers across budgets better than a single-budget probe.
- Do **not** claim to beat RLCR/DCPO (different, retrain-allowed setting); cite them as the upper bound that motivates the cheaper alternative.

---

### Implementation Walkthrough (Current Scaffold → Full System)

#### What exists today (`p04-calibration-probe/`)

| Component | File | Status | Assessment |
|---|---|---|---|
| Hidden-state calibration probe | `src/core.py` — `CalibrationProbe` | ✅ MLP→sigmoid, BCE-trainable | Right *family* (= a 2504.05419-lite correctness probe), but it is a **probe**, not a **recalibration map** |
| `recalibrate(raw_conf)` | `src/core.py` | ⚠️ **placeholder** — returns `sigmoid(raw_conf)`, a no-op | Must become a real Platt/temperature/isotonic head over raw confidence |
| BCE training loop + ckpt | `src/train.py` | ✅ BCEWithLogits (a proper scoring rule — good for calibration), checkpoints each epoch | OK; add Brier/NLL eval + early stop on val ECE |
| ECE helper | `src/evaluate.py` — `compute_ece()` | ✅ fixed-bin ECE | Need adaptive-bin ECE (Nixon), Brier, NLL, AURC, reliability diagrams |
| Eval harness | `src/evaluate.py` — `evaluate_model()` | ❌ **stub** (returns dummy dict) | No generation, no grading, no budget sweep |
| Synthetic data | `src/utils.py` — `make_synthetic_batch()` | ✅ smoke only | Replace with real frozen-LM features + correctness labels |
| Config | `src/config.py` | ⚠️ defaults to **Qwen2.5-1.5B-Instruct (non-reasoning)** | Switch default to a **Qwen3 thinking** model; add `budget` knobs as first-class config |

```13:33:p04-calibration-probe/src/core.py
class CalibrationProbe(nn.Module):
    def __init__(self, hidden_size: int = 2048, probe_dim: int = 128, num_layers: int = 1):
        ...
    def recalibrate(self, raw_conf: torch.Tensor) -> torch.Tensor:
        """If only raw model confidence (no hidden), a simple wrapper (extend as needed)."""
        return torch.sigmoid(raw_conf)  # placeholder  <-- NO-OP; this is the method that should
                                        #     hold temperature/Platt/isotonic + budget conditioning
```

**Scaffold verdict:** matches the right family but is missing the four things that make P4 publishable: (1) a **reasoning model + budget-sweep generation pipeline**, (2) **post-hoc recalibration baselines** (temperature/Platt/isotonic + verbalized confidence), (3) **proper scoring + selective-risk metrics** (Brier/NLL/AURC/reliability), and (4) **budget conditioning** (the probe must see budget as a feature, or be shown to transfer across budgets).

#### Phase 1 — Budget-controlled generation + calibration audit (Weeks 1–3)
1. **Models (frozen):** 3–4 open reasoning models in the Qwen3 1.7B–8B "thinking" range (e.g., Qwen3-1.7B, Qwen3-4B, Qwen3-8B thinking variants) + one cross-family check (e.g., a DeepSeek-R1-Distill-Qwen-7B) for generalization.
2. **Benchmarks:** MATH-500, GSM8K, AIME 2024/25, GPQA-diamond (math/science reasoning — the domain the anchor did *not* test; this is part of the delta).
3. **Budget axis (the independent variable):** sweep thinking budget via explicit token caps / budget-forcing ("Wait" injection, s1-style) / thinking-on-off; and sample-width N ∈ {1,4,8,16}. Log tokens/query.
4. **Confidence signals to record per query:** sequence log-prob / mean-token-prob, answer-token prob, self-consistency agreement, verbalized confidence ("just ask"), P(True) (Kadavath), and **last-token + mean-pooled hidden states** (cache to `data/features/`).
5. **Metrics:** ECE (fixed + adaptive bins), Brier, NLL, AURC / risk–coverage, accuracy@coverage, plus per-difficulty-quartile breakdown.
6. **Output:** `results/audit/{model}/{benchmark}_budget_sweep.json` + ECE-vs-budget and AURC-vs-budget figures. **This is the analysis paper's Figure 1–2.**

#### Phase 2 — Post-hoc recalibration baselines (Week 3–4)
Implement and benchmark, all **frozen-model, no-retrain**:
- Global **temperature scaling** (Guo 2017) on a held-out calibration split.
- **Platt scaling** and **isotonic regression** on raw confidence.
- **Verbalized confidence** elicitation (Tian "Just Ask", Xiong elicitation framework).
- **Thermometer**-style auxiliary calibrator (2403.08819) as a strong learned post-hoc baseline.
These are the bars the probe must beat at matched compute.

#### Phase 3 — Budget-conditioned recalibration probe (Weeks 4–6)
1. Turn `recalibrate()` into a real head: input = [hidden state ⊕ raw-confidence features ⊕ **budget embedding**]; output = calibrated prob. Keep it tiny (<1M params).
2. **Loss:** BCE / focal + an explicit calibration term (Brier as a differentiable proper scoring rule); optionally a low-FPR-weighted loss to optimize the selective operating point directly.
3. **Key ablations:** (a) hidden-state-only vs. +raw-conf vs. +budget; (b) **single-budget probe vs. budget-conditioned probe evaluated across budgets** (the novelty test: does conditioning buy cross-budget robustness?); (c) train-on-model-A → test-on-model-B transfer.

#### Phase 4 — Selective risk + conformal layer (Weeks 6–8)
1. Risk–coverage curves and AURC for every method × budget.
2. **Split-conformal / selective-prediction wrapper** on the recalibrated score → distribution-free coverage guarantee at a target risk; report coverage at fixed risk and accuracy@coverage. This is the main differentiator from RLCR (which gives no coverage guarantee) and from plain probes.
3. Low-FPR operating-point table (the deployment regime named in the problem statement).

#### Phase 5 — Analysis & paper artifacts (Weeks 8–10)
- Fig 1: ECE/Brier/AURC vs. budget per model (the audit; replicate-or-refute the anchor on math).
- Fig 2: reliability diagrams pre/post recalibration at low vs. high budget.
- Fig 3: budget-conditioned vs. single-budget probe across the budget axis (novelty figure).
- Table: probe vs. temp/Platt/isotonic/verbalized at matched compute; cross-model transfer.
- Honest negatives: budgets/models/domains where the probe does **not** help, and where the anchor's "more reasoning hurts calibration" claim **fails to replicate** on math (a publishable nuance in its own right).

---

### Research Landscape 2024–2026 (what already exists / what is solved)

**(a) "More reasoning ↛ better calibration" is now a crowded, largely-established finding.** The anchor (2508.15050) shows it on climate/health expert-confidence QA. The motivation sections of the RL-calibration wave (RLCR 2507.16806; DCPO 2603.09117; C²GSPG 2509.23129; process-margin supervision 2604.23333) all assert that accuracy-only RLVR and longer reasoning degrade calibration / produce overconfidence. Independent web evidence shows the relationship is **non-monotonic and model/scale/length-dependent** (e.g., "When More is Less: Understanding CoT Length" 2502.07266; reports of stable ECE below ~10k tokens then degradation; worst calibration sometimes at *light*, not heavy, reasoning). Takeaway: the bare descriptive claim is **mostly solved and contested**, so P4 must move past "does it happen" to "characterize it cleanly on math + fix it cheaply."

**(b) Post-hoc / inference-time calibration of LLMs is mature.** Temperature scaling (Guo 2017), Platt/isotonic, Thermometer (2403.08819, auxiliary universal calibrator), verbalized confidence (Lin 2022; Tian 2023; Xiong 2024), P(True)/P(IK) (Kadavath 2022), semantic uncertainty (Kuhn 2023). 2026 adds post-hoc methods aimed specifically at reasoning models: CaliDist (perturbation/distraction stability, 2606.05799), single-generation unsupervised calibration (2604.19444), perturbed-representation-stability probing (2505.21772). The toolbox is full; P4's probe must be positioned as *budget-aware* and *selective-risk-optimized*, not "yet another recalibrator."

**(c) Hidden-state probes for correctness on reasoning models already exist.** 2504.05419 is the direct incumbent (probe → calibrated correctness scores → early exit, −24% tokens). InternalInspector (EMNLP 2024 Findings) and "Thinking Out Loud: Do Reasoning Models Know When They're Right?" (2504.06564) are adjacent. P4's `CalibrationProbe` is essentially a lite version of this line.

**(d) Training-time calibration is the strong (but different-setting) competitor.** RLCR (2507.16806, Damani et al. — same author as P1's "Learning How Hard to Think") jointly trains accuracy + Brier-reward confidence and **beats post-hoc classifiers**; DCPO (2603.09117, ICML 2026) decouples a gradient conflict between accuracy and calibration. These bound what's achievable *with* retraining and must be cited as the ceiling P4 deliberately forgoes for a no-retrain, frozen-model recipe.

**(e) Selective prediction / abstention is adjacent and only partially connected to the budget axis.** SelectLLM (coverage–risk for LLMs), AbstentionBench (notably: *reasoning fine-tuning can worsen abstention*), budget-aware selective verification (2606.19808). Few works tie **selective risk to the test-time-compute budget** — this is part of P4's opening.

---

### Deep Paper Reviews (15) — with web-verification flags

> "Verified" = I confirmed title + arXiv ID + authors via web this session (abstract fetched for the three marked ✓✓). "Canonical" = foundational paper relied on from prior P1/P2 web checks, not re-fetched this session.

**1. Don't Think Twice! Over-Reasoning Impairs Confidence Calibration — Lacombe, Wu, Dilworth.** arXiv:2508.15050, ICML 2025 Workshop on Reliable & Responsible FMs (Aug 2025). **Verified: yes ✓✓.**
Increasing reasoning budget *consistently impairs* calibration; reasoning LLMs hit only 48.7% on expert-confidence assessment while search-augmented generation hits 89.3% — i.e., information access, not reasoning depth, is the bottleneck. **Crucial caveat the master doc omits: the domain is ClimateX climate/health expert-confidence QA, NOT math reasoning, and they propose NO recalibration method.** Relevance: the anchor/motivation. **Borrow:** the budget→ECE framing and "negative returns beyond modest budgets." **Beat:** test on math/science reasoning (does it even replicate?), and actually *fix* it with a probe — both gaps they leave open.

**2. Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification — Zhang, Chen, Pan, Zhao, Panda, Li, He.** arXiv:2504.05419 (Apr 2025). **Verified: yes ✓✓.**
Trains a probe on reasoning-model hidden states to predict answer correctness; reports "highly calibrated" scores and uses the probe as a verifier for early exit (−24% tokens, no perf loss). **This is the closest method neighbor and the main scoop risk for the probe half.** Relevance: near-identical core mechanism to `CalibrationProbe`. **Borrow:** the hidden-state extraction + probe recipe, the "predict future correctness" angle. **Beat:** they do not study calibration *as a function of budget*, do not condition the probe on budget, do not report selective-risk/AURC across budgets, and do not frame as post-hoc recalibration of confidence. P4's delta lives in exactly those gaps.

**3. Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR) — Damani et al.** arXiv:2507.16806 (2025); code: github.com/damanimehul/RLCR. **Verified: yes.**
RL with a correctness + Brier calibration reward; jointly improves accuracy and calibration in- and out-of-domain and **explicitly outperforms post-hoc confidence classifiers.** Relevance: the strongest competitor and the reason P4 must commit to the no-retrain framing. **Borrow:** Brier as a proper scoring objective; the verbalized-confidence-after-reasoning setup. **Beat (carefully):** not on raw calibration (you'll lose) — beat on *cost* ($hundreds, no RL, frozen model) and on *selective-risk guarantees via conformal*, which RLCR does not provide.

**4. Decoupling Reasoning and Confidence: Resurrecting Calibration in RLVR (DCPO) — Ma, Wen, Cao, Lu, Lin, Yang, He, Han, Sun.** arXiv:2603.09117, ICML 2026 (Mar→May 2026). **Verified: yes ✓✓.**
Identifies a gradient conflict between accuracy maximization and calibration-error minimization in RLVR; proposes DCPO to decouple them, improving calibration without sacrificing accuracy. Training-time. Relevance: confirms the accuracy↔calibration tension is fundamental, not incidental. **Borrow:** the framing that the two objectives conflict (motivates a *separate*, frozen recalibration head rather than touching the base model). **Beat:** N/A (different setting) — cite as upper bound.

**5. On Calibration of Modern Neural Networks — Guo, Pleiss, Sun, Weinberger.** arXiv:1706.04599, ICML 2017. **Verified: canonical** (also web-checked in P2).
Temperature scaling: single-parameter post-hoc calibration; introduced reliability diagrams + ECE to the modern era. Relevance: the baseline every calibration paper must beat. **Borrow:** temperature scaling as Baseline #1; reliability-diagram visualization. **Beat:** global temperature is budget-agnostic and operating-point-agnostic — P4's budget-conditioned, low-FPR-targeted probe is the contrast.

**6. Measuring Calibration in Deep Learning — Nixon, Dusenberry, Zhang, Jerfel, Tran.** arXiv:1904.01685 (2019). **Verified: canonical** (web-checked in P2).
Shows fixed-bin ECE is biased; proposes adaptive-binning / better calibration metrics. Relevance: the metric correctness backbone. **Borrow:** adaptive-bin ECE + thresholded variants for the audit (the scaffold's `compute_ece` uses naive fixed bins — upgrade it). **Beat:** N/A (tooling).

**7. Calibration of Pre-trained Transformers — Desai & Durrett.** arXiv:2003.07892, EMNLP 2020. **Verified: yes.**
BERT/RoBERTa are calibrated in-domain but degrade out-of-domain; temperature scaling helps, especially OOD. Relevance: pre-LLM baseline + the OOD-degradation lesson (AIME vs. GSM8K is an OOD axis for P4). **Borrow:** in-domain vs. OOD calibration split design. **Beat:** N/A (historical baseline).

**8. Language Models (Mostly) Know What They Know — Kadavath et al.** arXiv:2207.05221 (2022). **Verified: yes.**
P(True) self-evaluation and P(IK) are calibrated and scale with model size. Relevance: P(True) is a must-include confidence signal/baseline. **Borrow:** P(True) prompting as a per-query confidence feature for the probe and as a baseline. **Beat:** P(True) is uncalibrated under the budget shift and at low-FPR — show the probe corrects it.

**9. Teaching Models to Express Their Uncertainty in Words — Lin, Hilton, Evans.** arXiv:2205.14334 (2022). **Verified: yes.**
Models can emit calibrated verbalized confidence. Relevance: origin of verbalized-confidence baselines. **Borrow:** verbalized confidence as a cheap baseline. **Beat:** verbalized confidence drifts with reasoning length (the anchor's whole point) — quantify and recalibrate it.

**10. Just Ask for Calibration — Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, Manning.** arXiv:2305.14975, EMNLP 2023. **Verified: yes.**
For RLHF'd models, *verbalized* confidence is often better-calibrated than the conditional logprob (≈50% relative ECE reduction). Relevance: the verbalized-vs-logit baseline contrast is central to P4. **Borrow:** the elicitation prompts. **Beat:** show this breaks down under budget scaling on reasoning models and that a budget-conditioned probe restores it.

**11. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation — Xiong, Hu, Lu, Li, Fu, He, Hooi.** arXiv:2306.13063, ICLR 2024. **Verified: yes.**
Systematic black-box framework (prompt × sample × aggregate) for confidence; benchmarks calibration + failure prediction across 5 datasets/5 LLMs. **NOTE: master doc §P4 #2 attributes "Can LLMs Express Their Uncertainty?" to "Jiang et al." — that is wrong; it is Xiong et al.** Relevance: the methodological template for the elicitation baselines. **Borrow:** the consistency/aggregation confidence estimators. **Beat:** add the budget axis and hidden-state signal they (black-box) cannot use.

**12. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG — Kuhn, Gal, Farquhar.** arXiv:2302.09664, ICLR 2023. **Verified: yes.**
Semantic entropy (cluster generations by meaning, then entropy) — uncertainty for free-form outputs. Relevance: the right uncertainty signal for non-extractable / open answers; bridges to self-consistency width on the budget axis. **Borrow:** semantic-entropy as a confidence feature, especially for GPQA-style free responses. **Beat:** it's compute-heavy (needs samples); contrast against the single-pass hidden-state probe.

**13. Thermometer: Towards Universal Calibration for LLMs.** arXiv:2403.08819, ICML 2024. **Verified: yes (arXiv ID + title confirmed).**
Learns an auxiliary model that predicts a per-task temperature, generalizing calibration across tasks without per-task labels. Relevance: the strongest *learned post-hoc* baseline to beat. **Borrow:** auxiliary-calibrator architecture pattern. **Beat:** Thermometer is budget-agnostic and not selective-risk-targeted — P4 adds the budget conditioning and AURC/low-FPR objective.

**14. Thought Calibration: Efficient and Confident Test-Time Scaling.** arXiv:2505.18404 (2025). **Verified: yes (arXiv ID + title confirmed via search).**
Uses calibrated confidence over the thinking trace to decide when to stop, halving thinking tokens at matched accuracy (up to −60%). Relevance: directly couples calibration to the *budget* — the closest "calibration × test-time-compute" cousin, but aimed at efficiency, not at fixing miscalibration. **Borrow:** the trace-level confidence-to-stop machinery and the budget instrumentation. **Beat:** P4's goal is recovering *calibration quality* (ECE/AURC) under budget, not just saving tokens — complementary, citeable as related not competing.

**15. CarBoN: Calibrated Best-of-N Sampling Improves Test-Time Reasoning.** arXiv:2510.15674 (2025). **Verified: yes (arXiv ID + title confirmed via search).**
Calibration of the selection signal improves best-of-N test-time reasoning. Relevance: ties calibration to the *sample-width* budget axis and to selection (links P4 to P1/P2's verifier line). **Borrow:** calibrated-selection framing for the N-sample budget sweep. **Beat:** focused on accuracy via better selection, not on confidence calibration / selective risk of the final answer — P4 occupies the latter.

**Also tracked (landscape, not full reviews — all web-surfaced this session):** C²GSPG (2509.23129, training-time), Process-Supervision of Confidence Margin (2604.23333), Unsupervised Calibration from a Single Generation (2604.19444), CaliDist (2606.05799), Perturbed-Representation-Stability probing (2505.21772), InternalInspector (EMNLP 2024 Findings), "Thinking Out Loud" (2504.06564), "When More is Less: CoT Length" (2502.07266), "When More Thinking Hurts" (2604.10739), Think Again or Think Longer (2606.19808), SelectLLM, AbstentionBench, Mielke linguistic calibration (2012.14983, TACL 2022).

---

### Gaps & Novelty — what does NOT yet exist

**Already exists (do not claim as novel):**
- The descriptive claim "more reasoning / overconfident reasoning models hurt calibration" (2508.15050 and the RL-calibration wave).
- Hidden-state probes that predict reasoning-model correctness and yield calibrated scores (2504.05419).
- Post-hoc recalibration toolbox: temperature/Platt/isotonic/Thermometer/verbalized/P(True)/semantic entropy.
- Training-time calibration that beats post-hoc classifiers (RLCR, DCPO).
- Calibration-to-stop for efficiency (Thought Calibration); calibrated selection (CarBoN).

**Genuine white space (the contribution must live here):**
1. **A controlled budget-as-IV calibration audit on *math/science* reasoning across 3–4 open reasoning models.** The anchor is climate-QA only; others touch budget tangentially. A clean budget × model × benchmark grid of ECE/Brier/**AURC** on MATH/AIME/GPQA — including a *replicate-or-refute* of the anchor outside its domain — does not exist in one place.
2. **A budget-CONDITIONED post-hoc recalibration map.** No existing probe takes the compute budget as an input feature, or demonstrates that the optimal recalibration *shifts with budget* and that conditioning buys cross-budget robustness. This is the sharpest novel mechanism.
3. **Selective-risk / low-FPR evaluation of recalibration under budget variation**, with a **split-conformal coverage guarantee** layered on the recalibrated score. Calibration papers report ECE/Brier; almost none report AURC/risk–coverage *as a function of test-time compute*, and the conformal layer gives a guarantee RLCR lacks.
4. **Cross-budget and cross-model probe transfer** (train at one budget/model, deploy at another) as a cheap deployability result — untested in the incumbents.

**One-line positioning:** *"Recalibrating a frozen reasoning model whose calibration degrades with compute — budget-conditioned, selective-risk-guaranteed, no retraining."*

---

### Critical Assessment

**Well-posedness:** Good, once re-scoped. The original phrasing ("does compute help/hurt calibration") is empirically *answered and contested*; the well-posed, defensible question is the *budget-conditioned no-retrain recalibration with selective risk*. Define the budget axis precisely (thinking tokens vs. sample width are different axes — report both).

**Novelty:** Medium. Each ingredient has an incumbent; the synthesis (controlled axis + budget conditioning + selective risk + frozen constraint) is the novel cell. This is a respectable *analysis-with-a-method* paper, not a breakthrough — which matches its rank-3 "hedge" role.

**Feasibility:** Very high — the project's strongest dimension. Pure inference + a <1M-param probe; Qwen3 1.7B–8B thinking models run on a single A100/Colab Pro+. No RL, no base-model training. The scaffold already has the probe + BCE loop; the missing pieces (generation pipeline, recalibration baselines, metrics, budget conditioning) are engineering, not research risk.

**Target-bar realism:** Realistic *if* you do not over-claim. Beating temperature/Platt/verbalized at matched compute and showing budget-conditioning helps is achievable. Beating RLCR is not (and shouldn't be attempted). The conformal/selective-risk angle is the most reliable source of a clean, guaranteed result.

**Scoop risk:** **HIGH on the analysis half, MODERATE on the probe half.** The space added multiple papers in 2025–2026 H1; expect more by submission. Mitigation: lead with the *method + selective-risk guarantee* (harder to scoop than an observation) and ship the audit fast as supporting evidence.

**Null-result risk:** **Moderate-high.** Web evidence already shows the budget→ECE relationship is non-monotonic and model/length-dependent (stable below ~10k tokens, worst at *light* reasoning in some models). The headline "more compute → worse calibration" may **not** replicate cleanly on math, weakening the motivation if you stake the paper on it. Defense: frame the audit as *characterizing* the (possibly non-monotonic) relationship, and anchor the contribution on the probe + conformal layer, which deliver value regardless of the sign of the trend. A "fails to replicate on math; here's the nuance + a fix that works either way" paper is still publishable.

**Scaffold-vs-method fit:** Partial. `CalibrationProbe` is the right family but is currently a correctness probe, not a recalibration map; `recalibrate()` is a no-op; metrics are ECE-only; config defaults to a non-reasoning model; there is no budget conditioning or generation pipeline. All fixable in Phases 1–4.

---

### Recommended Implementation Phases (tied to the scaffold)

| Phase | Weeks | Goal | Scaffold work |
|---|---|---|---|
| **1 — Audit** | 1–3 | Budget × model × benchmark calibration grid | Build real `evaluate.py` generation+grading; add budget sweep + confidence/hidden logging; upgrade `compute_ece` → adaptive ECE + Brier + NLL + AURC; switch config default to Qwen3-thinking |
| **2 — Post-hoc baselines** | 3–4 | Temperature/Platt/isotonic/verbalized/Thermometer | New `recalibration.py`; replace the `recalibrate()` no-op with real heads |
| **3 — Budget-conditioned probe** | 4–6 | Train `CalibrationProbe` with [hidden ⊕ raw-conf ⊕ budget]; Brier+BCE loss | Extend `core.py` forward to take budget; ablations single- vs. budget-conditioned; cross-model transfer in `train.py` |
| **4 — Selective risk + conformal** | 6–8 | Risk–coverage, AURC, split-conformal coverage guarantee | New `selective.py`; low-FPR operating-point tables |
| **5 — Paper** | 8–10 | Figures, transfer tables, honest negatives | `plots.py` (reliability diagrams, ECE/AURC-vs-budget, conditioning ablation) |

**Reuse:** generation + grading + feature extraction can be shared with P1/P2/P3 (`features.py`, `eval_harness.py`); P4's marginal cost over a P1/P2 run is hidden-state caching + probe training (near-zero), exactly as the master doc's parallel-track plan intends.

---

### Compute Budget Estimate

- **Generation (dominant cost):** 3–4 models × 4 benchmarks × ~5 budget settings × N≤16 samples. Most is small models (1.7B–8B) at modest token counts. ~50–110 A100-hrs (use vLLM for eval generation to cut this substantially).
- **Feature caching + probe training:** <1M-param probe on cached hiddens — a few GPU-hrs total across all ablations.
- **Total:** ~**80–150 A100-hrs**, **$150–350** — consistent with the master/compass estimate. Comfortably within single-A100 / Colab Pro+. **No base-model training**, so no large bursts.

---

### Citation Caveats / Unverified or Wrong IDs

**Corrections to master `research.md` §P4 20-paper map:**
- **#2 "Jiang et al. — Can LLMs Express Their Uncertainty? (2023)" — WRONG AUTHOR.** The paper of that exact title is **Xiong et al., arXiv:2306.13063, ICLR 2024**. (There is a separate Jiang et al. 2021 "How Can We Know When Language Models Know?", TACL — different paper. Decide which you mean; don't conflate.)
- **#7 "Mielke et al. — Reducing Calibration Error (2022)" — TITLE WRONG/PARAPHRASED.** Actual: **"Reducing Conversational Agents' Overconfidence Through Linguistic Calibration," Mielke, Szlam, Dinan, Boureau, TACL 2022, arXiv:2012.14983.**
- **#10 "Xiong et al. — Approximate Nearest Neighbor Calibration (2024)" — UNVERIFIABLE / LIKELY BOGUS.** No such Xiong paper found; the title looks garbled. Xiong et al.'s real contribution is the confidence-elicitation paper miscredited to "Jiang" at #2. **Drop or replace #10.**
- **#14 "Conformal Prediction for LLMs (2024)", #15 "Selective Prediction surveys (2023)", #16 "Process Reward Models — calibration at step level (ICLR 2024)", #13 "DeepSeek-R1 / reasoning model reports"** — these are *category placeholders*, not real citations; resolve each to a specific arXiv ID before camera-ready (e.g., conformal-for-LLM and selective-prediction each have concrete 2024–2025 papers; pick one).

**Anchor caveat:** arXiv:2508.15050 is **verified** but is an **ICML 2025 *Workshop*** paper on **ClimateX climate/health QA**, not a peer-reviewed math-reasoning study and it proposes **no recalibration method** — cite precisely and do not overstate its scope.

**2026-dated IDs — verified existence this session (post my Jan-2026 cutoff, web-confirmed):** 2603.09117 (DCPO, ICML 2026) ✓✓; surfaced in search with matching titles/IDs: 2604.23333, 2604.19444, 2606.05799, 2606.19808, 2604.10739, 2601.03042 (BaseCal). Treat any not directly abstract-fetched as "title+ID match via search" and re-confirm before citing in a submission.

**Verified-as-real foundational IDs:** 1706.04599 (Guo), 1904.01685 (Nixon), 2003.07892 (Desai & Durrett), 2207.05221 (Kadavath), 2205.14334 (Lin), 2305.14975 (Tian), 2306.13063 (Xiong), 2302.09664 (Kuhn), 2403.08819 (Thermometer), 2504.05419, 2507.16806 (RLCR), 2508.15050, 2505.18404, 2510.15674, 2012.14983.

---

### Next-Actions Checklist

- [ ] **Re-aim the README/run.md framing** from "discover that reasoning hurts calibration" → "budget-conditioned, selective-risk post-hoc recalibration of frozen reasoning models." (Anti-scoop.)
- [ ] Switch config default from Qwen2.5-1.5B-Instruct → a **Qwen3-thinking** model; add `budget` knobs as first-class config.
- [ ] Build the **real generation + grading + budget-sweep** pipeline in `evaluate.py` (reuse P1/P2 harness).
- [ ] Upgrade metrics: adaptive-bin ECE (Nixon), **Brier, NLL, AURC / risk–coverage**, reliability diagrams.
- [ ] Replace the `recalibrate()` **no-op** with real **temperature/Platt/isotonic** heads; add **verbalized confidence** + **P(True)** + **Thermometer** baselines.
- [ ] Extend `CalibrationProbe.forward` to ingest **[hidden ⊕ raw-conf ⊕ budget]**; run the **single- vs. budget-conditioned** ablation (the novelty test).
- [ ] Add **split-conformal** selective-prediction wrapper + low-FPR operating-point tables.
- [ ] Run **cross-budget and cross-model transfer** experiments.
- [ ] **Replicate-or-refute** arXiv:2508.15050 on MATH/AIME/GPQA; report the (likely non-monotonic) trend honestly.
- [ ] Resolve master-doc citation errors (#2 author, #7 title, #10 bogus, #13–16 placeholders) before any submission.

---

*Last updated: 2026-06-28. Source: `research.md` §P4 + web research (2024–2026). Anchor verified: arXiv:2508.15050. Closest method neighbor: arXiv:2504.05419. Strongest competitor: arXiv:2507.16806 (RLCR).*
