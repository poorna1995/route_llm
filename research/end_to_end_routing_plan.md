# End-to-End Plan: Unsupervised Rule-Based LLM Routing

One-page mental model, then the full pipeline. Read §1 first if you feel lost.

**Implementation detail (files, deps, CLI):** [`implementation_plan_next.md`](implementation_plan_next.md)

---

## 1. Mental model (read this first)

You have **three data splits**. Each has a different job:

```text
FIT  (2278)   →  learn what signals mean (science only)
CALIB (~499)  →  pick thresholds / which feature set to use
EVAL (~2597)  →  report final numbers once (never tune here)
```

For **every** query you will eventually need:

1. **Signals** — so the router can decide (weak probe + query complexity).
2. **Answers** — weak and strong predictions, so you can score quality after routing.

So you **probe eval**, but you **do not tune on eval**.  
Probing = compute features/answers. Tuning = choosing \(\tau\) or top‑k.

```text
                    ┌─────────────┐
  query q  ───────► │  ROUTER R   │ ──► weak or strong
                    │  uses z(q)  │
                    └─────────────┘
                           ▲
                           │
              z = [ φ(q)  |  ψ(q, M_weak) ]
                   query     weak probe
                   only      (H, p_max, …)
```

**Gold / `needs_strong` are never inputs to R.** They only score whether R was good.

---

## 2. What we already have vs what we still need

| Piece | fit | calib | eval |
|-------|:---:|:-----:|:----:|
| Corpus queries | ✓ | ✓ | ✓ |
| Complexity \(\phi(q)\) / \(C_*\) | ✓ | ✓ | ✓ |
| Weak probe \(\psi\) | ✓ | ✗ | ✗ |
| Strong probe (for scoring) | ✓ | ✗ | ✗ |
| Feature ranking | can do now | — | — |
| Threshold \(\tau\) | — | need probes | — |
| Final PGR / CPT | — | — | need probes |

**Next compute job:** probe calib then eval (weak, then strong), same upsert files as fit.

**Cost / latency:** each new probe row includes a `cost` block (`latency_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `n_calls`). Progress logs print per-query ms/tokens. Optional USD via `pricing:` in `signals/query_model/config.yaml`. Fit rows probed before this change have no `cost` field until re-probed.

---

## 3. End-to-end pipeline (phases)

### Phase A — Signal inventory (DONE on fit)

Build two vectors per query:

- \(\phi(q)\): `C_length, C_density, C_atypical, C_linguistic, C_query`
- \(\psi(q,M_{\mathrm{weak}})\): `H, p_max, margin, top2_mass`
- \(\mathbf{z} = [\phi \| \psi]\) (9 features) — `signals/feature_vector.py`

On **fit only**, measure AUROC of each feature vs `needs_strong`.  
This tells you which signals carry routing information.  
It does **not** choose the production threshold.

**Fit finding (already known):** weak \(H\) / \(p_{\max}\) / margin are strong; \(C_*\) are weak; MD features are highly correlated.

---

### Phase B — Probe calib & eval (NEXT, blocking)

```bash
# Calib (~499)
./run.sh query-model --backend hf --experiment local_strong \
  --roles calib --model-roles weak --no-paraphrase --progress-every 50

./run.sh query-model --backend hf --experiment local_strong \
  --roles calib --model-roles strong --no-paraphrase --progress-every 50

# Eval (~2597) — for testing the frozen router, not for tuning
./run.sh query-model --backend hf --experiment local_strong \
  --roles eval --model-roles weak --no-paraphrase --progress-every 50

./run.sh query-model --backend hf --experiment local_strong \
  --roles eval --model-roles strong --no-paraphrase --progress-every 50
```

Outputs (upsert, safe):  
`calib_weak.jsonl`, `calib_strong.jsonl`, `eval_weak.jsonl`, `eval_strong.jsonl`.

Why eval probes?  
So on eval you can (1) compute \(z(q)\), (2) apply frozen \(R\), (3) score accuracy / PGR / CPT.  
You still **never** search \(\tau\) or top‑k on eval.

---

### Phase C — Rank features (fit only)

1. Join fit: complexity + weak + strong → label `needs_strong`.
2. AUROC for each of the 9 features (correct orientation).
3. Save ranking (e.g. `artifacts/rank_fit.json`).

Example nested sets to evaluate later (fixed before looking at eval):

| Set | Features | Purpose |
|-----|----------|---------|
| \(S_H\) | \(\{H\}\) | Headline simple rule |
| \(S_p\) | \(\{p_{\max}\}\) | Twin single-signal |
| \(S_{\mathrm{top2}}\) | top‑2 by fit AUROC | Your ablation |
| \(S_{\mathrm{top4}}\) | top‑4 | Your ablation |
| \(S_{\mathrm{top5}}\) | top‑5 | Your ablation |
| \(S_{\mathrm{md}}\) | \(\{H,p_{\max},\mathrm{margin}\}\) | Family control (collinearity) |
| \(S_{\mathrm{all}}\) | all 9 | Upper bound of equal-weight \(z\) |

---

### Phase D — Define the rule (still unsupervised / rule-based)

**Single-feature rule (headline):**
\[
R(q)=\begin{cases}
M_{\mathrm{strong}} & \text{if } H(q\mid M_{\mathrm{weak}})\ge\tau \\
M_{\mathrm{weak}} & \text{otherwise}
\end{cases}
\]

**Multi-feature rule (ablations):** for a set \(\mathcal{F}\),
1. Z-score features with **fit** mean/std only.
2. Orient so higher ⇒ escalate (e.g. use \(-p_{\max}\)).
3. Score \(s(q)=\frac{1}{|\mathcal{F}|}\sum_{j\in\mathcal{F}}\tilde z_j(q)\).
4. Escalate if \(s(q)\ge\tau\).

No Arena / preference training.

---

### Phase E — Choose \(\tau\) (calib only)

For each feature set \(S\):

1. Sweep \(\tau\) over calib scores.
2. For each \(\tau\), compute:
   - strong-call rate \(\alpha\)
   - routed accuracy
   - PGR
3. Pick \(\tau^\star_S\) with a **declared** rule, e.g.:
   - CPT(80%): smallest \(\alpha\) with PGR ≥ 0.80, **or**
   - best PGR with \(\alpha \le 0.25\)
4. Freeze \(\tau^\star_S\). Do not change it after seeing eval.

Also keep baselines on calib/eval: always-weak, always-strong, random@\(\alpha\).

---

### Phase F — Test (eval only, once)

For each frozen \((S, \tau^\star_S)\):

1. Compute \(z(q)\) on eval (needs eval weak probe + existing \(C(q)\)).
2. Apply \(R\).
3. If route = weak → use weak answer; if strong → use strong answer.
4. Report: accuracy, PGR, \(\alpha\), CPT(50/80) from the eval sweep curve (curve can be plotted; **chosen** \(\tau\) stays the calib one).

**Figures:** fit rank bars · routing curves · table comparing \(S_H\) vs top‑2/4/5 vs \(S_{\mathrm{md}}\).

---

## 4. One diagram of the full story

```text
[Corpus fit/calib/eval]
        │
        ├─► complexity build  →  φ(q) for all splits          ✓ done
        │
        └─► query-model probe →  ψ(q, M_weak), answers
               │
               ├─ fit     ✓ done   → rank features, AUROC
               ├─ calib   ✗ next   → choose τ*, compare sets
               └─ eval    ✗ next   → apply τ*, report PGR/CPT
                        │
                        ▼
              Rule R based on z(q)
                        │
                        ▼
              weak or strong answer
                        │
                        ▼
              Metrics vs gold (eval)
```

---

## 5. What “success” looks like

- **Science:** clear ranking — MD uncertainty ≫ \(C_*\).
- **System:** \(S_H\) (or \(S_p\)) recovers most of the weak–strong gap at low \(\alpha\) on **eval**.
- **Ablation:** top‑5 ≈ top‑2 ≈ \(\{H\}\) (expected, because of collinearity); adding \(C_*\) does not help CPT.
- **Honesty:** probe cost stated (1 weak MC probe or \(n\) Hotpot samples).

---

## 6. Order of work (checklist)

1. [ ] Probe **calib** weak → strong  
2. [ ] Probe **eval** weak → strong  
3. [ ] Join + build \(\mathbf{z}\) for fit/calib/eval  
4. [ ] Fit: rank 9 features → save ranking + figure  
5. [ ] Calib: sweep \(\tau\) for \(S_H, S_p, S_{\mathrm{top2/4/5}}, S_{\mathrm{md}}, S_{\mathrm{all}}\)  
6. [ ] Freeze \(\tau^\star\) per set  
7. [ ] Eval: one-shot table + routing curves  
8. [ ] Write results into paper / `routing_signal_analysis.md`

---

## 7. Short answers to common confusions

**Q: Why probe eval if eval is for testing?**  
A: Testing needs inputs (signals) and outputs (answers) on eval queries. Probing creates them. Tuning is forbidden on eval; probing is required.

**Q: Do we lock only on \(H\)?**  
A: Headline rule = \(H\) or \(p_{\max}\). Ablations = top‑2/4/5 and family sets. Both.

**Q: Can we pick top‑5 on fit then test on eval?**  
A: Yes — rank on fit, set \(\tau\) on calib, report on eval.

**Q: Is gold used at route time?**  
A: No. Only for measuring AUROC / PGR after the fact.
