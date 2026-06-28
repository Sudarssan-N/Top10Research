# P2 Research Foundation — Calibrated, False-Positive-Bounded Verifiers

**Problem:** Can verifiers be trained/calibrated to minimize false positives at selection-relevant operating points, raising the best-of-N ceiling?

**Compute:** 1–4×A100 (~200 GPU-hrs)  
**Venues:** NeurIPS / ICLR  
**Project:** `p02-calibrated-fp-verifiers/`  
**Parent index:** [`../../research.md`](../../research.md) §P2

---


### Problem Statement

> Can verifiers be trained/calibrated to minimize false positives at selection-relevant operating points, raising the best-of-N ceiling?

**Core claim:** False positives impose a hard asymptotic ceiling on best-of-N (Stroebl; Agrawal). Verifiers are trained for accuracy/AUC, not **low-FPR precision at selection operating points**. P2 trains a lightweight probe with precision-first loss + post-hoc calibration, measures **ceiling shift** vs. standard PRMs, and exports `precision@τ` / FPR labels to P1's trust head.

**Target bar:** Demonstrate measurable ceiling shift at fixed FPR budget (e.g., FPR≤5%) on MATH-500/GSM8K; beat standard outcome/PRM verifier at matched FPR, not just matched AUC.

**Venues:** NeurIPS / ICLR  
**Compute:** 1–4×A100, ~200 GPU-hrs

---

### Implementation Walkthrough (Current Scaffold → Full System)

#### What exists today (`p02-calibrated-fp-verifiers/`)

| Component | File | Status |
|---|---|---|
| FPR-bounded verifier probe | `src/core.py` — `FPRBoundedVerifier` | ✅ score head, `find_threshold_for_target_fpr()`, FPR/precision metrics |
| Asymmetric training loop | `src/train.py` | ✅ FP-weighted BCE; configurable loss types (Phase 3) |
| Calibration utilities | `src/calibration.py` | ✅ ECE (Nixon-style), temperature scaling (Guo 2017) |
| Verifier scorers | `src/verifier.py` | ✅ mock + outcome-style baselines |
| Ceiling analysis | `src/ceiling.py` | ✅ accuracy-vs-N, theoretical ceiling, SNR diagnostic |
| Phase 1 ceiling baselines | `src/phase1.py` | ✅ multi-N generation + verifier selection sweep |
| Shared eval harness | `src/benchmarks.py`, `grading.py`, `generation.py` | ✅ ported from P1 |
| Entry point | `scripts/run_experiment.py`, `scripts/run_phase1_ceiling.py` | ✅ smoke + phase1 modes |
| Config | `configs/default.yaml`, `configs/phase1.yaml` | ✅ |

```14:60:p02-calibrated-fp-verifiers/src/core.py
class FPRBoundedVerifier(nn.Module):
    ...
    def find_threshold_for_target_fpr(self, scores, labels, target_fpr=0.05) -> float:
        """Binary search for threshold achieving approx target FPR (key for bounded selection)."""
```

#### Phase 1 — Ceiling & verifier baselines (Weeks 1–4)

**Goal:** Quantify the false-positive ceiling before training P2.

1. **Reuse P1 generations** (or regenerate): N ∈ {1, 4, 8, 16, 32} per query on MATH-500, GSM8K.
2. **Verifier baselines in `verifier.py`:**
   - Outcome verifier (Cobbe recipe)
   - Discriminative PRM proxy (length/heuristic mock → real PRM800K fine-tune)
   - GenRM (Zhang) — optional Week 4
   - Naive best-of-N (argmax score)
3. **Ceiling analysis (Stroebl / Agrawal):**
   - Per-query FP labels: verifier accepts incorrect candidate
   - Plot accuracy vs. N, **theoretical ceiling** vs. empirical
   - Report precision@τ, FPR@τ, optimal N per difficulty quartile
4. **Scorer SNR diagnostic (Dalal):** estimate useful k̂ / N_max before search hurts.
5. **Output:** `results/phase1/p2_ceiling_summary.json` + ceiling-shift figure.

**Research rationale:** Cannot claim P2 raises the ceiling without Stroebl/Agrawal curves on the same generations.

#### Phase 2 — Label & calibration data pipeline (Weeks 4–5)

**Goal:** Build training/eval sets for `FPRBoundedVerifier`.

1. **Extend P1 `features.py` pattern:** hidden states at end of generation (or mean-pool over solution tokens).
2. **Label schema per (query, candidate):**
   - `is_correct` (sympy / `grading.py`)
   - `verifier_fp` = verifier accepted ∧ ¬correct
   - `verifier_score` from each baseline PRM/GenRM
3. **Calibration holdout:** 20% queries for temperature scaling + threshold search only.
4. **Cache:** `data/p2_labels/{model}/{benchmark}.pt`

**Research rationale:** P2 trains on the same frozen-LM features as P1; labels feed both FPR probe and P1 `target_trust`.

#### Phase 3 — FPR-bounded verifier training (Weeks 5–7)

**Goal:** Train `FPRBoundedVerifier` + ablations.

1. **Losses (implement ≥3):**
   - Asymmetric BCE with `alpha` (precision emphasis — current scaffold)
   - Focal loss on FP events
   - OC-loss pairwise variant (Agrawal) on (pos, neg) solution pairs
2. **Post-hoc calibration:** temperature scaling (Guo 2017) on validation.
3. **Baselines to beat:** standard PRM, outcome verifier, GenRM at **matched FPR** (not just matched AUC).
4. **Metrics:** ROC, PR curve, **precision@target_FPR**, ECE (Nixon), overoptimization knee (Gao).

**Key ablation:** accuracy-optimized verifier vs. FPR-bounded verifier at τ where FPR=5%.

#### Phase 4 — Threshold & aggregation policy (Weeks 7–8)

**Goal:** Turn scores into selection policies.

| Policy | Mechanism | Prior art |
|---|---|---|
| **Fixed τ** | `find_threshold_for_target_fpr(target=0.05)` | P2 core |
| **Pessimistic agg.** | min step score / min candidate score | Agrawal beam search |
| **Conservative BoN** | argmax only among candidates with score ≥ τ | Stroebl FP mitigation |
| **SNR-gated N** | cap N at k̂ from Dalal diagnostic | P1 trust coupling |

1. Sweep τ ∈ {1%, 5%, 10% FPR}.
2. Compare mean vs. min vs. product aggregation for process scores.
3. Export per-query `(precision@τ, fpr@τ)` → P1 trust head labels.

#### Phase 5 — Ceiling-shift evaluation (Weeks 8–10)

**Goal:** Prove P2 raises best-of-N ceiling and feeds P1.

1. **Primary metric:** Δaccuracy vs. N at **fixed FPR budget** (e.g., FPR≤5%).
2. **Secondary:** accuracy vs. tokens/samples (Pareto, shared `plots.py`).
3. **Integrate with Snell-style search** (simplified compute-optimal BoN).
4. **Cross-benchmark:** MATH-500, GSM8K, AIME subset, GPQA-diamond.
5. **P1 handoff:** write `data/p2_trust_labels/{benchmark}.pt` with per-query trust = precision@τ.
6. **Statistical testing:** bootstrap CIs on ceiling shift.

#### Phase 6 — Analysis & paper artifacts

- Figure 1: FP ceiling limits best-of-N (Stroebl reproduction)
- Figure 2: ceiling shift — standard PRM vs. FPR-bounded vs. OC-loss
- Figure 3: reliability diagram + precision@τ operating curve
- Table: useful N_max / k̂ before Dalal-style degradation
- Failure cases: domains where FP bounding hurts recall

---

### Research Foundation for P2

**Thesis chain:**

1. **Best-of-N depends on verifiers** (Cobbe; Lightman; Snell) — test-time gains come from ranking many candidates with a reward/verification signal.
2. **Imperfect verifiers impose a hard ceiling** (Stroebl; Agrawal) — false positives cannot be eliminated by resampling; optimal N is often surprisingly small (<10).
3. **Verifiers optimize the wrong objective** (Gao; GenRM) — accuracy/AUC does not equal **low-FPR precision at selection operating points**.
4. **Scorer quality limits search width** (Dalal; Snell) — noisy verifiers cause wider search to *hurt*; signal-to-noise governs useful N.
5. **Gap → P2:** Train/calibrate verifiers with **FPR-bounded, precision-first objectives** + threshold tuning; measure ceiling shift; export trust labels to P1.

**Closest prior work (not equivalent):**

- Agrawal: precision-first PRM (full PRM, not lightweight probe + calibration-at-τ)
- Stroebl: ceiling theory only, no training recipe
- GenRM/ThinkPRM: strong verifiers, not FPR-calibrated at τ
- Snell: uses PRM, does not retrain for low-FPR selection

---

### 20 Closest Papers for P2

| # | Paper | ID / Venue | Relevance to P2 | Priority |
|---|---|---|---|---|
| 1 | **The Limits of Inference Scaling Through Resampling** — Stroebl, Kapoor & Narayanan | arXiv:2411.17501 | Theoretical FP ceiling; optimal N often <10 | **P0** |
| 2 | **Cut the Overcredit: Precision First Process Rewards** — Agrawal et al. | OpenReview:7mVZy4mI1J (ICLR 2026 sub.) | OC loss; FP ceiling in best-of-N | **P0** |
| 3 | **Let's Verify Step by Step** — Lightman et al. | ICLR 2024, arXiv:2305.20050 | PRM800K backbone; primary verifier baseline | **P0** |
| 4 | **Generative Verifiers: Reward Modeling as Next-Token Prediction** — Zhang et al. | ICLR 2025, arXiv:2408.15240 | GenRM baseline; positivity bias context | **P1** |
| 5 | **Training Verifiers to Solve Math Word Problems** — Cobbe et al. | 2021, arXiv:2110.14168 | Outcome verifier + best-of-N baseline | **P0** |
| 6 | **Process- and Outcome-Based Feedback** — Uesato et al. | 2022, arXiv:2211.14275 | Outcome vs. process label ablation | **P1** |
| 7 | **Scaling Laws for Reward Model Overoptimization** — Gao et al. | 2023, arXiv:2210.10760 | Proxy-vs-gold knee as N grows | **P1** |
| 8 | **ThinkPRM: Process Reward Models That Think** — Khalifa et al. | OpenReview:V727xqBYIW | Data-efficient generative PRM baseline | **P1** |
| 9 | **More Test-Time Compute Can Hurt** — Dalal et al. | arXiv:2603.15377 | Scorer SNR / k̂ diagnostic | **P0** |
| 10 | **Scaling LLM Test-Time Compute Optimally** — Snell et al. | ICLR 2025, arXiv:2408.03314 | PRM search consumer; compute-optimal BoN | **P0** |
| 11 | **LLMs Are Better Reasoners with Self-Verification** — Weng et al. | EMNLP 2023, arXiv:2212.09561 | Self-verifier bias precursor | **P2** |
| 12 | **LLMs Cannot Self-Correct Reasoning Yet** — Huang et al. | ICLR 2024, arXiv:2310.01798 | External verifier justification | **P2** |
| 13 | **Discovering LM Behaviors with Model-Written Evals** — Perez et al. | 2022, arXiv:2212.09251 | Verifier failure-mode discovery methodology | **P2** |
| 14 | **RMB: Comprehensively Benchmarking Reward Models** — Zhou et al. *(proxy for "Empirical Study of RMs")* | ICLR 2025, arXiv:2410.09893 | Cross-domain BoN RM eval | **P1** |
| 15 | **Self-Rewarding Language Models** — Yuan et al. | ICML 2024, arXiv:2401.10020 | Closed-loop self-judge counterpoint | **P2** |
| 16 | **Direct LM Alignment from Online AI Feedback** — Guo et al. | 2024, arXiv:2402.04792 | Online label refresh / FPR drift | **P2** |
| 17 | **VeryTrace** — Zhong et al. *(proxy for "Natural Reasoning Verifier"; Kale co-author)* | arXiv:2606.24124 | Structured/oracle verifier upper bound | **P2** |
| 18 | **RAP: Retrieval-Augmented Planning** — Kagaya et al. *(proxy for Chen et al. 2023)* | arXiv:2402.03610 | Search-then-verify architecture pattern | **P2** |
| 19 | **On Calibration of Modern Neural Networks** — Guo et al. | ICML 2017, arXiv:1706.04599 | Temperature scaling post-hoc calibration | **P0** |
| 20 | **Measuring Calibration in Deep Learning** — Nixon et al. | 2019, arXiv:1904.01685 | ECE evaluation toolkit | **P0** |

**Implementation priority for P2:** Papers 1–3, 5, 9–10, 19–20 (ceiling + baselines + calibration), then 2 (OC loss), then 4, 6–8, 14 (strong verifier baselines).

**Papers sharing infrastructure with P1:** Snell (#10), Stroebl (#1), Agrawal (#2), Lightman (#3), Cobbe (#5), Gao (#7), Dalal (#9), Guo 2017 (#19).

**Citation caveats (P2-specific):**
- Papers #14, #17, #18: original titles in the strategy doc were unresolved; proxies noted above.
- Agrawal / ThinkPRM are under review — cite Stroebl for peer-reviewed ceiling claims.
- Dalal 2026 (arXiv:2603.15377) is a preprint — verify venue before camera-ready.

---

---

*Last updated: 2026-06-26. Source: `research.md` §P2.*
