# Which Signals Inform Unsupervised LLM Routing?

**Status:** Fit signal study **+** calib \(\tau^\star\) / eval routing (rule-based)  
**Pool:** `local_strong` — weak = Llama-3.1-8B-Instruct; strong = Qwen2.5-32B-Instruct  
**Data:** fit \(n=2278\) · calib \(n=499\) · eval \(n=2597\) (ARC · HotpotQA · MMLU)  
**Artifacts:** `{fit,calib,eval}_{weak,strong}.jsonl`, `signals/query/processed/{role}.jsonl`, `router/artifacts/rule_based/`  
**Companion figures:** `research/figures/fig_signal_correlation.png`, `research/figures/fig_routing_curve.png`

---

## Abstract

We study which unsupervised features carry information for binary weak–strong LLM routing. Signals fall into two families: **model-independent** query complexity \(C(q)=g(\phi(q))\), and **model-dependent** answer uncertainty \(\psi(q,M_{\mathrm{weak}})\) obtained from a weak-model probe. On the fit corpus, weak-probe entropy \(H\) and confidence \(p_{\max}\) (equivalently margin) align strongly with both weak-model error and the routing label \(\texttt{needs\_strong}\) (weak wrong \(\wedge\) strong right), with AUROC in the range \(0.72\)–\(0.86\) depending on task. In contrast, individual complexity components \(C_{\mathrm{length}}\), \(C_{\mathrm{density}}\), \(C_{\mathrm{atypical}}\), \(C_{\mathrm{linguistic}}\), and the composite \(C_{\mathrm{query}}\) remain near chance (\(\mathrm{AUROC}\approx 0.50\)–\(0.58\)). Pearson correlations between the two families are near zero (\(|r|\lesssim 0.14\)), while within the weak confidence family signals are nearly collinear (\(|r|\approx 0.96\)–\(0.99\)). We conclude that, under the present estimators and corpus, **routing decisions should be driven by weak-model \(H\) / \(p_{\max}\)**; model-independent \(C_*\) are retained as a cheap but currently weak complementary inventory, not as primary decision features. A subsequent calib→eval rule-based study (§12) confirms this: with \(\tau^\star\) chosen by CPT80 on calib, \(S_H\) recovers ~85% of the weak→strong gap on eval at ~45% strong-call rate, matching richer sets at lower cost.

---

## 1. Introduction

Supervised routers learn a mapping from queries to models using preference battles, quality-gap labels, or correctness matrices (e.g., RouteLLM, Hybrid LLM, IRT-Router). Our formulation instead defines an a-priori **signal vector**
\[
\mathbf{z}(q,M)=\bigl[\phi(q),\;\psi(q,M)\bigr]
\]
and routes by thresholding rules or lightly weighted combinations of those signals, without Arena-scale preference supervision.

A central empirical question follows: **which coordinates of \(\mathbf{z}\) actually carry routing information?** We answer this by measuring alignment of each signal with (i) weak-model failure and (ii) the escalate-or-not oracle label \(\texttt{needs\_strong}\), and by examining within- and cross-family correlations.

---

## 2. Signal inventory

### 2.1 Model-independent signals \(\phi(q)\)

Computed from query text alone (no candidate LLM), using row-local structural and linguistic features plus embedding geometry fit only on the fit split:

| Symbol | Definition (summary) |
|--------|----------------------|
| \(C_{\mathrm{length}}\) | Z-scored prompt / question token length |
| \(C_{\mathrm{density}}\) | Lexical density: higher MATTR and lower zlib compression ratio increase the score |
| \(C_{\mathrm{atypical}}\) | Semantic atypicality: \(0.5\,z(\mathrm{centroid\_distance})+0.5\,z(\mathrm{lof\_score})\) in MiniLM embedding space |
| \(C_{\mathrm{linguistic}}\) | Mean of z-scored Bloom-style reasoning depth, multi-hop cues, domain breadth, requirement markers, sentence count |
| \(C_{\mathrm{query}}\) | Equal-weight composite of the four \(C_*\) (weights frozen from config; intended for calib retuning) |

**Intended decision rule:** \(R_C(q)=M_{\mathrm{strong}}\) if \(C_{\mathrm{query}}\ge\tau_C\), else \(M_{\mathrm{weak}}\).

### 2.2 Model-dependent signals \(\psi(q,M_{\mathrm{weak}})\)

Computed by probing the **weak** model on query \(q\):

| Symbol | Estimator |
|--------|-----------|
| \(H(q\mid M_{\mathrm{weak}})\) | MC: Shannon entropy of the renormalized option-letter distribution (one forward score). HotpotQA: discrete cluster entropy over \(n\) free-form samples |
| \(p_{\max}\) | Maximum option (or cluster) probability from the same probe |
| margin | Gap between top-1 and top-2 probabilities |
| top2_mass | Mass on the top two options / clusters |
| \(U(q\mid M_{\mathrm{weak}})\) | Paraphrase disagreement (deferred in current runs: `--no-paraphrase`) |

**Intended decision rule:** escalate if \(H(q\mid M_{\mathrm{weak}})\ge\tau_H\) (or equivalently low \(p_{\max}\) / margin).

### 2.3 Labels (evaluation only)

These are **not** routing inputs:

- \(\texttt{weak\_wrong}\): weak prediction incorrect under the corpus metric (letter match for MC; exact match for HotpotQA).
- \(\texttt{needs\_strong}\): weak incorrect **and** strong correct — the queries for which escalation recovers quality.
- \(\texttt{strong\_wrong}\): strong incorrect (used only for strong self-alignment).

---

## 3. Experimental protocol

| Item | Setting |
|------|---------|
| Split | **fit** (signal validation; thresholds not frozen for eval here) |
| Weak | Llama-3.1-8B-Instruct |
| Strong | Qwen2.5-32B-Instruct |
| Alignment metric | Binary AUROC; scores oriented so that higher uncertainty / lower confidence \(\Rightarrow\) higher predicted positive rate |
| Correlation | Pearson \(r\) (Spearman reported where noted) |
| Join | Inner join on `query_id` across complexity, weak, and strong streams (\(n=2278\)) |

We report per-source slices and an MC-pooled slice (ARC+MMLU). Pooled statistics that mix HotpotQA with MC are interpreted cautiously because free-form EM and option-letter probes are different estimators.

---

## 4. Capability gap and escalate opportunity

### 4.1 Accuracy

| Slice | Weak | Strong | Gap |
|-------|-----:|-------:|----:|
| Pooled | 0.691 | 0.831 | +0.140 |
| MC pooled | 0.803 | 0.933 | +0.131 |
| ARC-Challenge | 0.837 | 0.960 | +0.123 |
| MMLU | 0.680 | 0.838 | +0.158 |
| HotpotQA | 0.547 | 0.699 | +0.152 |

A non-trivial weak–strong gap exists on every slice, so routing has a well-defined quality budget to recover.

### 4.2 Contingency of outcomes

| Slice | both correct | \(\texttt{needs\_strong}\) | both wrong | strong-only wrong | % of weak errors fixed by strong |
|-------|-------------:|---------------------------:|-----------:|------------------:|---------------------------------:|
| Pooled | 1507 | 385 | 320 | 66 | 54.6% |
| ARC | 826 | 134 | 29 | 11 | 82.2% |
| MMLU | 181 | 52 | 37 | 8 | 58.4% |
| HotpotQA | 500 | 199 | 254 | 47 | 43.9% |

On ARC, most weak failures are recoverable by the strong model. On HotpotQA, a large **both-wrong** mass limits the maximum benefit of escalation: strong cannot fix what neither model solves under EM.

---

## 5. Model-dependent signals as routing information

### 5.1 Self-alignment: weak \(\psi\) \(\rightarrow\) \(\texttt{weak\_wrong}\)

| Slice | \(H\) | \(p_{\max}\) | margin | top2_mass |
|-------|------:|-------------:|-------:|----------:|
| ARC | 0.852 | 0.855 | 0.855 | 0.825 |
| MMLU | 0.823 | 0.823 | 0.818 | 0.805 |
| HotpotQA | 0.766 | 0.766 | 0.766 | 0.624 |
| MC pooled | 0.854 | 0.857 | 0.856 | 0.830 |
| Pooled | 0.740 | 0.749 | 0.750 | 0.568 |

**Interpretation.** On multiple-choice tasks, a single option-letter probe yields strong uncertainty–error alignment (AUROC \(\approx 0.85\)). Hotpot cluster entropy is weaker but clearly above chance (\(\approx 0.77\)). Within the MC confidence family, \(H\), \(p_{\max}\), and margin are interchangeable for ranking; top2_mass is slightly inferior.

### 5.2 Routing alignment: weak \(\psi\) \(\rightarrow\) \(\texttt{needs\_strong}\)

| Slice | \(H\) | \(p_{\max}\) | margin | top2_mass |
|-------|------:|-------------:|-------:|----------:|
| ARC | 0.855 | 0.859 | 0.859 | 0.828 |
| MMLU | 0.760 | 0.770 | 0.773 | 0.738 |
| HotpotQA | 0.719 | 0.719 | 0.719 | 0.574 |
| MC pooled | 0.836 | 0.841 | 0.842 | 0.810 |
| Pooled | 0.756 | 0.759 | 0.761 | 0.610 |

**Interpretation.** The same weak-probe features that predict weak failure also predict **useful** escalation. Alignment to \(\texttt{needs\_strong}\) is essentially as strong as alignment to \(\texttt{weak\_wrong}\) on ARC, and only modestly lower on MMLU and HotpotQA. This is the central positive result: **weak \(H\) / \(p_{\max}\) provide actionable routing information**, not merely a post-hoc description of errors.

### 5.3 Strong self-signals (reference, not routing inputs)

Strong-model \(H\) / \(p_{\max}\) align well with \(\texttt{strong\_wrong}\) on MC (AUROC \(\approx 0.88\)–\(0.91\)) but poorly on HotpotQA (\(\approx 0.60\)). These scores are **not** used for the live escalate decision: the system probes the weak model, then optionally calls the strong model. Reporting them clarifies that MC option-letter UQ remains informative for the strong model, while free-form cluster entropy remains limited.

---

## 6. Model-independent signals as routing information

### 6.1 Alignment of \(C_*\) \(\rightarrow\) \(\texttt{needs\_strong}\)

Using the better of the two orientations (higher-is-harder vs. flipped), best AUROCs remain modest:

| Signal | Pooled | ARC | MMLU | HotpotQA | MC pooled |
|--------|-------:|----:|-----:|---------:|----------:|
| \(C_{\mathrm{length}}\) | 0.54 | 0.54 | 0.54 | 0.56 | 0.55 |
| \(C_{\mathrm{density}}\) | 0.55 | 0.54 | 0.58 | 0.53 | 0.55 |
| \(C_{\mathrm{atypical}}\) | 0.52 | 0.52 | 0.54 | 0.53 | 0.54 |
| \(C_{\mathrm{linguistic}}\) | 0.50 | 0.51 | 0.54 | 0.52 | 0.52 |
| \(C_{\mathrm{query}}\) | 0.53 | 0.51 | 0.51 | 0.54 | 0.52 |
| *weak \(H\) (reference)* | **0.76** | **0.86** | **0.77** | **0.72** | **0.84** |

### 6.2 Component-wise remarks

- **\(C_{\mathrm{length}}\).** Weak positive association with hardness on MC; on HotpotQA the sign **inverts** (longer full-context prompts associate with fewer \(\texttt{needs\_strong}\) cases). Pooling therefore mixes conflicting regimes.
- **\(C_{\mathrm{density}}\).** Frequently better when **flipped** (higher density \(\neq\) harder under our construction). Best flipped AUROC on MMLU (\(\approx 0.58\)) remains far below weak \(H\).
- **\(C_{\mathrm{atypical}}\).** Most stable positive orientation among \(C_*\) (embedding atypicality / LOF). Still near chance for routing.
- **\(C_{\mathrm{linguistic}}\).** Near chance; raw Bloom / multi-hop / domain cues individually \(\approx 0.50\)–\(0.55\). Highly correlated with length (\(r\approx 0.73\)), so partly a length proxy.
- **\(C_{\mathrm{query}}\).** Equal-weight blend inherits length and Hotpot sign conflicts; it does not dominate any component.

A simple joint score \(\tfrac12 z(C_{\mathrm{query}})+\tfrac12 z(H)\) **degrades** AUROC relative to \(H\) alone (drops of \(0.05\)–\(0.10\) across slices). Orthogonality alone does not imply useful complementarity when one family is uninformative.

---

## 7. Correlation structure

Figure: `research/figures/fig_signal_correlation.png`.

### 7.1 Cross-family (pooled Pearson \(r\))

|  | \(H\) | \(p_{\max}\) | margin | top2_mass |
|--|------:|-------------:|-------:|----------:|
| \(C_{\mathrm{length}}\) | 0.02 | 0.06 | 0.07 | 0.02 |
| \(C_{\mathrm{density}}\) | −0.10 | 0.13 | 0.14 | 0.10 |
| \(C_{\mathrm{atypical}}\) | 0.08 | −0.08 | −0.08 | −0.07 |
| \(C_{\mathrm{linguistic}}\) | 0.03 | 0.02 | 0.03 | −0.01 |
| \(C_{\mathrm{query}}\) | 0.01 | 0.05 | 0.05 | 0.02 |

**Conclusion.** Model-independent and model-dependent families are **nearly orthogonal**. They are not substitutes for one another.

### 7.2 Within-family

- **Weak MD:** \(H\) and \(p_{\max}\) are almost redundant (\(r\approx -0.97\)); \(p_{\max}\) and margin \(r\approx 0.99\). For papers and systems, report one primary MD signal (\(H\) or \(p_{\max}\)) and treat the others as ablations.
- **\(C_*\):** \(C_{\mathrm{query}}\) is dominated by length and linguistic load (\(r\approx 0.75\)–\(0.85\)); \(C_{\mathrm{atypical}}\) is the most distinct component.

---

## 8. What should drive the routing decision?

We distinguish **defined** signals (the inventory in §2) from **empirically informative** signals (those with material AUROC on \(\texttt{needs\_strong}\)).

### 8.1 Primary routing information (supported)

On the present fit evidence, the decision-relevant coordinates are:

\[
\psi^\star(q)=\bigl\{H(q\mid M_{\mathrm{weak}}),\; p_{\max}(q\mid M_{\mathrm{weak}}),\; \mathrm{margin}(q\mid M_{\mathrm{weak}})\bigr\},
\]

with the practical recommendation to use **either \(H\) or \(p_{\max}\)** from a single weak probe (same MC forward pass; Hotpot uses the free-form cluster estimator). The escalate rule
\[
R(q)=
\begin{cases}
M_{\mathrm{strong}} & \text{if } H(q\mid M_{\mathrm{weak}})\ge \tau_H,\\
M_{\mathrm{weak}} & \text{otherwise}
\end{cases}
\]
is justified by AUROC \(0.72\)–\(0.86\) against \(\texttt{needs\_strong}\). Threshold \(\tau_H\) should be set on **calib** and frozen for **eval**; fit AUROC alone does not choose \(\tau\).

### 8.2 Secondary / deferred

| Signal | Status |
|--------|--------|
| \(U(q\mid M_{\mathrm{weak}})\) | Defined; not measured in current `--no-paraphrase` runs |
| \(C_{\mathrm{atypical}}\) | Best-behaved \(C_*\); still too weak for primary gating |
| \(C_{\mathrm{length}}\) on MC only | Mild; must not be pooled naively with Hotpot |
| \(C_{\mathrm{query}}\) equal-weight | Not validated as primary; needs per-source sign / weight retuning |

### 8.3 Non-inputs

Gold outcomes (\(\texttt{weak\_wrong}\), \(\texttt{needs\_strong}\)), preference winners, and strong-model self-entropy at decision time are evaluation constructs or alternative architectures—not the unsupervised decision rule studied here.

---

## 9. Implications for evaluation and the paper narrative

1. **Signal validation before policy.** Fit AUROC of weak \(H\)/\(p_{\max}\) vs \(\texttt{needs\_strong}\) (earlier sections); then routing curves / CPT / PGR on calib–eval (§12).
2. **Do not claim \(C(q)\) as a strong escalate feature yet.** Position \(C_*\) as a model-independent baseline inventory with limited predictive power on this corpus; the correlation figure supports “orthogonal but weak.” Adding \(C_*\) into \(S_{\mathrm{all}}\) increases \(\alpha\) without a meaningful accuracy gain on eval.
3. **Task stratification.** Always separate MC from Hotpot when discussing both accuracy and signal quality.
4. **Probe cost honesty.** MD routing pays one weak probe (MC) or \(n\) samples (Hotpot); this is architectural cost, distinct from query-only supervised routers.

---

## 10. Limitations

- Hotpot quality uses **exact match**, which understates semantic correctness and inflates both-wrong.
- Strong model is Qwen2.5-32B (local pool), not Llama-3.1-70B (paper primary pair).
- Paraphrase uncertainty \(U\) was not enabled (`--no-paraphrase`).
- \(\tau^\star\) selection uses pooled CPT80; per-source thresholds are not tuned separately (Hotpot still escalates more under the pooled rule).
- Complexity weights are default equal weights; retuning may improve \(C_*\) slightly but is unlikely to close the gap to \(H\) given current AUROCs.

---

## 11. Conclusion

Among the unsupervised signals defined for weak–strong routing, **information for the routing decision resides primarily in the weak model’s answer uncertainty**—entropy \(H\) and the confidence summaries \(p_{\max}\) and margin from the same probe. Model-independent complexity features \(C_*\) are nearly uncorrelated with these probes and, under current estimators, do not reliably rank queries that need escalation. On held-out **eval**, a calib-frozen threshold on \(H\) alone recovers about **85%** of the weak→strong gap while calling strong on about **45%** of queries—matching richer feature sets at lower cost. The empirically supported decision pathway is therefore: probe \(M_{\mathrm{weak}}\) \(\rightarrow\) read \(H\) or \(p_{\max}\) \(\rightarrow\) threshold on **calib** \(\rightarrow\) optionally call \(M_{\mathrm{strong}}\) on **eval**, with \(C(q)\) retained as an optional pre-inference baseline rather than the primary routing signal.

---

## 12. Routing policy: calib \(\tau^\star\) and eval (rule-based)

This section closes the loop after the fit-only signal study: choose \(\tau\) on **calib**, freeze it, report on **eval**. No preference labels; equal-weight oriented / z-scored feature sets; selection rule **`cpt80`** (minimal \(\alpha\) with PGR \(\ge 0.80\) on calib).

**Commands / artifacts**

```bash
./run.sh route rank --role fit
./run.sh route sweep --role calib   # → tau_star.json, zscore_fit.json, curves_calib.json
./run.sh route eval                 # → eval_eval.json
./run.sh route curves --role eval   # → fig_routing_curve.png (plot-only eval sweep)
```

Fit rank order (pooled AUROC → `needs_strong`):  
`margin`, `p_max`, `H`, `top2_mass`, `C_atypical`, …  
So \(S_{\mathrm{top2}}=\{\mathrm{margin},p_{\max}\}\), \(S_{\mathrm{top4}}\) adds \(H\) and `top2_mass`, \(S_{\mathrm{top5}}\) adds `C_atypical`.

### 12.1 Baselines

| Split | \(n\) | Always weak | Always strong | Gap |
|-------|------:|------------:|--------------:|----:|
| calib | 499 | 0.717 | 0.844 | +0.126 |
| eval | 2597 | 0.586 | 0.753 | +0.166 |

Eval is harder than calib (lower absolute accuracies, larger gap); \(\alpha\) at frozen \(\tau^\star\) therefore tends to rise on eval even when PGR stays high.

### 12.2 Calib: chosen \(\tau^\star\) (CPT80)

| Set | Features (summary) | \(\alpha\) | Acc | PGR | CPT80 \(\alpha\) |
|-----|--------------------|----------:|----:|----:|-----------------:|
| \(S_H\) | \(H\) | 0.341 | 0.820 | 0.810 | 0.341 |
| \(S_p\) | \(p_{\max}\) | 0.341 | 0.820 | 0.810 | 0.341 |
| \(S_{\mathrm{md}}\) | \(H,p_{\max},\mathrm{margin}\) | 0.341 | 0.820 | 0.810 | 0.341 |
| \(S_{\mathrm{top2}}\) | margin, \(p_{\max}\) | 0.341 | 0.820 | 0.810 | 0.341 |
| \(S_{\mathrm{top4}}\) | + \(H\), top2_mass | 0.341 | 0.820 | 0.810 | 0.341 |
| \(S_{\mathrm{top5}}\) | + \(C_{\mathrm{atypical}}\) | 0.381 | 0.820 | 0.810 | 0.381 |
| \(S_{\mathrm{all}}\) | all \(C_*\) + \(\psi\) | 0.519 | 0.820 | 0.810 | 0.519 |

On calib, single-signal \(S_H\) / \(S_p\) already hit the CPT80 target at \(\alpha\approx 0.34\). Adding all complexity features inflates the call rate without improving calib PGR under this selector.

### 12.3 Eval: frozen \(\tau^\star\) (no retune)

| Set | \(\alpha\) | Acc | PGR |
|-----|----------:|----:|----:|
| \(S_H\) | **0.452** | **0.728** | **0.850** |
| \(S_p\) | 0.449 | 0.727 | 0.845 |
| \(S_{\mathrm{md}}\) | 0.449 | 0.727 | 0.847 |
| \(S_{\mathrm{top2}}\) | 0.448 | 0.727 | 0.845 |
| \(S_{\mathrm{top4}}\) | 0.451 | 0.727 | 0.847 |
| \(S_{\mathrm{top5}}\) | 0.512 | 0.727 | 0.847 |
| \(S_{\mathrm{all}}\) | 0.625 | 0.730 | 0.863 |

**Reading:** router accuracy ~**73%** sits between always-weak (**58.6%**) and always-strong (**75.3%**). Simple sets recover **~85%** of the gap at **~45%** strong calls. \(S_{\mathrm{all}}\) gains only ~1–2 pp of PGR while calling strong on **62.5%** of queries — worse cost efficiency.

### 12.4 Eval slices for \(S_H\)

| Slice | \(n\) | \(\alpha\) | Acc | PGR |
|-------|------:|----------:|----:|----:|
| pooled | 2597 | 0.452 | 0.728 | 0.850 |
| MC pooled | 1597 | 0.413 | 0.877 | 0.865 |
| ARC-Challenge | 1000 | 0.326 | 0.927 | 0.859 |
| MMLU | 597 | 0.559 | 0.794 | 0.875 |
| HotpotQA | 1000 | 0.513 | 0.489 | 0.827 |

Hotpot drives higher \(\alpha\) and lower absolute accuracy (EM); MC slices look stronger. PGR remains high on every slice, so the rule generalizes in gap-recovery terms even where EM is harsh.

### 12.5 Routing curves

Figure `research/figures/fig_routing_curve.png` shows **plot-only** eval sweeps of accuracy and PGR vs \(\alpha\). Stars mark the **calib \(\tau^\star\)** operating point on each curve (not re-chosen on eval). \(S_H\) / \(S_{\mathrm{top2}}\) / \(S_{\mathrm{top4}}\) dominate the upper-left frontier; \(S_{\mathrm{all}}\) lies below/right (more calls for similar quality).

### 12.6 Takeaway for the paper

1. Fit ranking predicted the right primary family (\(\psi\): \(H\) / \(p_{\max}\) / margin).  
2. Calib CPT80 yields a usable frozen threshold.  
3. Eval confirms **~85% PGR at ~45% \(\alpha\)** for \(S_H\) — competitive with top-\(k\) sets, cheaper than \(S_{\mathrm{all}}\).  
4. Prefer reporting \(S_H\) (or \(S_p\)) as the main unsupervised rule; treat \(S_{\mathrm{all}}\) as a negative control for “dump all features.”
