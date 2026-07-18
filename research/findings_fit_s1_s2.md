# Findings and Results: Model-Dependent Signal Analysis (Fit Split)

**Date:** 2026-07-16  
**Stage:** Signal validation S1–S2 (`signals/query_model/WORKFLOW.md` §9)  
**Data:** `signals/query_model/processed/fit.jsonl` joined with `datasets/processed/corpus_v1/queries_fit.jsonl`  
**Model:** weak = `meta-llama/Llama-3.1-8B-Instruct` (HF backend)  
**Not yet available:** strong-model probes, paraphrase \(U\), query complexity \(C(q)\) join → S3–S6 deferred

This note reports what the fit analysis shows about whether **unsupervised model-dependent signals**—answer entropy \(H\), max option probability \(p_{\max}\), margin, and related transforms—align with weak-model failure, and what that implies for routing before any policy is trained.

---

## 1. Experimental setup

| Item | Value |
|------|--------|
| Split | **fit** only (signal validation; no threshold tuning) |
| Queries | 2,278 total |
| Sources | ARC-Challenge 1,000 · HotpotQA 1,000 · MMLU 278 |
| Probe (MC) | Single-pass option-letter distribution → \(H\), \(p_{\max}\), margin, … |
| Probe (Hotpot) | Free-form cluster entropy over \(n=5\) samples (997/1000); 3 rows still MC-shaped |
| Labels | Gold from corpus; `weak_wrong` = prediction ≠ gold (EM for Hotpot, letter match for MC) |
| Metric (S2) | AUROC of signal → `weak_wrong` (direction: higher uncertainty ⇒ more likely wrong) |
| Success gate (WORKFLOW) | ≥1 family beats chance AUROC on **pooled MC fit**, or clear per-source story |

**What this analysis is not:** It is not a routing curve, not PGR/CPT, and not a claim that Hotpot cluster entropy is production-ready. It answers Chuang-style **RQ1 alignment**: *does the weak model’s own uncertainty rank its mistakes?*

---

## 2. S1 — Weak-model accuracy (baseline capability)

### 2.1 Overall table

| Slice | \(n\) | Accuracy | Wrong |
|-------|------:|---------:|------:|
| **ARC-Challenge** | 1,000 | **0.837** | 163 |
| **MMLU** | 278 | **0.680** | 89 |
| **MC pooled** (ARC+MMLU) | 1,278 | **0.803** | 252 |
| **HotpotQA** | 1,000 | **0.041** | 959 |
| **Pooled (all)** | 2,278 | 0.468 | 1,211 |

### 2.2 Analysis

**Multiple-choice tasks look healthy.** On ARC-Challenge the 8B model is strong (~84%); on MMLU it is moderate (~68%). Pooled MC accuracy ~80% means there is a meaningful minority of failures (~20%) for uncertainty to detect—ideal for an alignment study (neither ceiling nor floor).

**HotpotQA accuracy is not a fair capability read.** Exact-match against short gold phrases yields only 4.1% “correct.” Many predictions are near-answers, refusals, or differently phrased spans; EM understates semantic correctness. For routing validation this still matters: the **label** `weak_wrong` is almost always 1, so Hotpot dominates pooled error counts and **must not be mixed with MC when reporting a single AUROC**.

**Pooled accuracy (46.8%) is misleading** as a system headline. It is an average of a high-accuracy MC regime and a near-zero EM Hotpot regime. All paper tables should be **per-source** (and MC-pooled separately).

**Implication for routing:** On MC, the weak model already solves most queries; a good router should escalate only the uncertain minority. On Hotpot (as currently scored), almost everything would escalate if gated on correctness—so the interesting question shifts from “detect rare failures” to “can any free-form signal distinguish the few EM hits from the rest?” (S2 says: barely / inversely.)

---

## 3. S2 — Uncertainty–correctness alignment (primary finding)

### 3.1 AUROC by slice and signal family

Canonical direction: score increasing ⇒ more likely `weak_wrong` (entropy/surprisal positive; \(p_{\max}\)/margin/top2 flipped in the analyzer).

| Slice | \(H\) | \(p_{\max}\) | margin | top2_mass | Verdict |
|-------|------:|-------------:|-------:|----------:|---------|
| **ARC-Challenge** | **0.852** | **0.855** | 0.855 | 0.825 | Strong alignment |
| **MMLU** | **0.823** | **0.823** | 0.818 | 0.805 | Strong alignment |
| **MC pooled** | **0.854** | **0.857** | 0.856 | 0.830 | **Passes success gate** |
| HotpotQA | 0.428 | 0.409 | 0.362 | 0.529 | **Below / ~chance; inverted for most families** |
| Pooled (all) | 0.656 | 0.666 | 0.665 | 0.443 | Mixed; **do not cite as main result** |

Chance = 0.5. Values \(\gtrsim 0.8\) on MC are strong for an unsupervised, single-probe signal.

### 3.2 Distributional evidence (why MC AUROC is high)

**ARC-Challenge — entropy by correctness**

| Condition | \(n\) | Mean \(H\) | Median \(H\) | Mean \(p_{\max}\) |
|-----------|------:|-----------:|-------------:|------------------:|
| Correct | 837 | 0.158 | 0.011 | 0.945 |
| Wrong | 163 | 0.628 | 0.630 | 0.738 |

**MMLU — entropy by correctness**

| Condition | \(n\) | Mean \(H\) | Median \(H\) | Mean \(p_{\max}\) |
|-----------|------:|-----------:|-------------:|------------------:|
| Correct | 189 | 0.307 | 0.067 | 0.883 |
| Wrong | 89 | 0.810 | 0.840 | 0.656 |

Wrong answers sit at clearly higher entropy and lower peak probability. Quantiles reinforce separation on ARC: correct answers have \(H\) q75 ≈ 0.155, while wrong answers have \(H\) q25 ≈ 0.308—distributions barely overlap in the middle.

**Error rate by \(H\) tertile (ARC)**

| Tertile | \(H\) range | \(n\) | Error rate |
|---------|-------------|------:|-----------:|
| Low | [0.000, 0.004) | 333 | **1.2%** |
| Mid | [0.004, 0.175) | 333 | 8.1% |
| High | [0.175, 1.377) | 334 | **39.5%** |

This is the operational story for a rule-based router \(R_H\): escalate when weak-model \(H\) is high. On ARC, the top tertile concentrates ~40% errors vs ~1% in the bottom tertile—without any preference labels or trained probe.

### 3.3 Signal-family redundancy on MC

On MC-pooled rows, correlations with \(H\):

| Pair | Correlation |
|------|------------:|
| \(H\) vs \(p_{\max}\) | −0.973 |
| \(H\) vs margin | −0.953 |
| \(H\) vs surprisal | +0.950 |
| \(H\) vs top2_mass | −0.858 |

**Finding:** Entropy, confidence, and margin are near-monotone transforms of the same option-letter distribution. Reporting all four as independent “wins” would overstate novelty. For the paper, pick **one representative per family** (WORKFLOW: \(H\), \(p_{\max}\), margin, top2_mass) and state that other members track the representative.

`top2_mass` is slightly weaker (AUROC ~0.83 vs ~0.85) but still useful as a “mass on top-two options” confidence summary.

### 3.4 HotpotQA — negative / null finding (important to report)

| Observation | Detail |
|-------------|--------|
| Accuracy (EM) | 4.1% (41/1000) |
| Unique \(H\) values | Only 7 discrete levels (cluster-count induced) |
| Cluster counts | Mostly 1–2 clusters over 5 samples (321 ones, 339 twos) |
| Mean \(H\) \| correct | **0.692** |
| Mean \(H\) \| wrong | **0.601** |
| AUROC(\(H\) → wrong) | **0.428** (worse than chance) |

**Analysis:** Correct Hotpot rows are *slightly more* uncertain under cluster entropy than wrong ones—the opposite of the MC pattern. Causes likely compound:

1. **Label noise / harsh EM** — many “wrong” answers may be semantically right; signal cannot align with a noisy binary.
2. **Low-resolution \(H\)** — with \(n=5\) samples and EM clustering, entropy takes few values; ranking power collapses.
3. **Degenerate confidence** — large mass at \(p_{\max}\in\{1.0, 0.8, 0.6, \ldots\}\) from small discrete supports.
4. **Task mismatch** — multi-hop open QA needs better free-form UQ (semantic clustering, paraphrase \(U\), more samples) than letter entropy.

**Paper stance:** Report Hotpot as a **limitation of the current free-form probe**, not as evidence against model-dependent routing. The MC success gate is met; Hotpot motivates S6/S7 and paraphrase \(U\), not abandonment of \(H\).

### 3.5 Why pooled AUROC (~0.66) must not be the headline

Pooling mixes:
- MC rows where \(H\) strongly ranks errors (AUROC ~0.85), and
- Hotpot rows where \(H\) is flat/inverted (AUROC ~0.43) and wrong rate ~96%.

The mixture AUROC (~0.66) is an artifact. **Cite MC-pooled and per-source; footnote Hotpot separately.**

---

## 4. Interpretation for unsupervised routing

### 4.1 What the results support

1. **Pre-inference, model-dependent signals are informative on MC.** A single weak-model probe yields \(H\) / \(p_{\max}\) that predict `weak_wrong` at AUROC 0.82–0.86 on ARC and MMLU—without Arena preferences, quality-gap labels, or trained hidden-state probes.
2. **Rule-based escalation is empirically justified on MC.** Error rate rises sharply with \(H\) tertile on ARC (1% → 8% → 40%). Threshold \(\tau_H\) can be fit on calib later; the signal itself is unsupervised.
3. **Success gate before router: PASSED for MC.** WORKFLOW requires ≥1 family beating chance on pooled MC fit; \(p_{\max}\) AUROC **0.857** and \(H\) **0.854** clear that bar.
4. **Positioning vs supervised routers (related work):** These signals are computed from the **live query–model interaction**, not used only offline to manufacture training winners (cf. Zhang et al. SE-as-labels). Inference input is a **signal vector**, not query-only embedding.

### 4.2 What the results do *not* yet show

| Claim | Status |
|-------|--------|
| Signals predict `needs_strong` (weak wrong ∧ strong right) | **Blocked** — no strong-model rows in current `fit.jsonl` (S3) |
| \(H\) adds beyond query complexity \(C(q)\) | **Blocked** — query stream not joined (S4) |
| Weak–strong gap \(H_w - H_s\) helps | **Blocked** — need paired strong probes (S5) |
| Paraphrase \(U\) complements \(H\) | **Blocked** — no `paraphrase` field yet (S6) |
| Routing recovers quality at low strong-call rate (PGR/CPT) | Deferred to router stage after thresholds on calib |
| Hotpot free-form \(H\) is routing-ready | **Rejected for now** — AUROC ≤ chance under EM |

### 4.3 Recommended signal inventory for v1 MC router

| Role | Signal | Rationale |
|------|--------|-----------|
| Primary | \(H(Y\mid q, M_{\mathrm{weak}})\) or \(p_{\max}\) | Best AUROC; nearly interchangeable |
| Secondary | margin | Same family story; slightly redundant |
| Hold | top2_mass | Mildly weaker; keep for ablation |
| Defer | Hotpot cluster \(H\) | Fix probe / metric before policy |
| Next | \(U(q\mid M)\), \(C(q)\) | Complementarity tests S4/S6 |

---

## 5. Threats to validity

1. **Fit-only reporting.** Numbers are for signal *discovery*; eval must be held out for the paper’s final tables. Do not tune \(\tau\) on fit.
2. **Weak-only pool.** Without strong correctness, we cannot separate “hard for everyone” from “needs strong.” S3 is required before claiming routing benefit.
3. **Hotpot EM.** Understates quality; may invert uncertainty alignment. Prefer judge/normalized match for analysis, or restrict Hotpot claims to S7 (level correlation) until scoring improves.
4. **Family redundancy.** Multiple near-identical AUROCs are one scientific finding, not four.
5. **Manifest `sources_filter: hotpotqa`** is stale relative to the full three-source file; document actual contents (ARC+Hotpot+MMLU) in experiment text.
6. **No paraphrase / no \(C(q)\).** The unsupervised *inventory* is only partially measured; conclusions are about **answer uncertainty on MC**, not the full \(\mathbf{z}(q,M)\).

---

## 6. Concise results summary (for paper draft)

> On the fit split with Llama-3.1-8B-Instruct as the weak model, unsupervised answer uncertainty from a single multiple-choice probe aligns strongly with weak-model errors: AUROC 0.85 on ARC-Challenge (\(n{=}1000\), accuracy 83.7\%) and 0.82 on MMLU (\(n{=}278\), accuracy 68.0\%). Confidence summaries \(p_{\max}\) and margin track entropy (correlations \(|r| > 0.95\)) and yield essentially the same ranking. Error rates on ARC rise from 1.2\% in the lowest entropy tertile to 39.5\% in the highest, supporting a pre-inference escalation rule based on \(H(q \mid M_{\mathrm{weak}})\). On HotpotQA, five-sample cluster entropy under exact-match scoring does not beat chance (AUROC 0.43) and is not used for routing claims until free-form uncertainty and evaluation metrics are improved. Strong-model probes, paraphrase uncertainty \(U\), and query complexity \(C(q)\) remain for S3–S6.

---

## 7. Next experiments (ordered)

1. **S3** — Run strong model on same fit IDs; define `needs_strong`; report AUROC of \(H_w\), \(p_{\max,w}\).
2. **S4** — Join `signals/query` complexity; ΔAUROC of \([C(q), H]\) vs \(C(q)\) alone.
3. **S6** — Enable paraphrase \(U\); correlation with \(H\); joint AUROC.
4. **S7** — Hotpot: better clustering / soft match; correlate \(H\) with `meta.level`.
5. **Calib thresholds** — Only after S3 passes; then eval PGR/CPT curves.
6. **S8** — Optional weak-model family ablation (Qwen vs Llama).

---

## Appendix: Reproduce

```bash
python signals/query_model/analyze.py \
  --role fit --model-role weak --skip-verify --s2-all-signals \
  --json-out /tmp/fit_weak.json
```

Primary artifact: `signals/query_model/processed/fit.jsonl`  
Corpus: `datasets/processed/corpus_v1/queries_fit.jsonl`
