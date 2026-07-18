# Router code review (recheck)

Date: 2026-07-17. Scope: `router/join.py`, `router/metrics.py`, `router/rule_based/*`.

---

## Verdict

**Concept and linkage are correct** for rule-based routing over \(z=[\phi\|\psi]\). Several bugs were found and fixed; remaining items are intentional limitations / next polish.

---

## Dependency graph (OK)

```text
corpus_v1/queries_{role}.jsonl
signals/query/processed/{role}.jsonl          → φ
signals/query_model/processed/{role}_weak.jsonl   → ψ, pred
signals/query_model/processed/{role}_strong.jsonl → labels / scoring
        ↓
router/join.py  (uses analyze.is_correct, feature_vector.z_vectors)
        ↓
rule_based/rank.py  → artifacts/rank_fit.json
rule_based/sweep.py → tau_star.json, zscore_fit.json   (needs calib+strong)
rule_based/evaluate.py → eval_*.json                   (frozen τ)
```

CLI: `./run.sh route {rank|sweep|eval|join-check}` → `router/rule_based/cli.py`.

`learn_weight/` is stub only — correct for current focus.

---

## Bugs found and fixed

| Issue | Severity | Fix |
|-------|----------|-----|
| `best_auroc = max(auc, auc_flip)` → `nan` if one side nan | Medium | Finite-safe best |
| `pick_tau` CPT50 dead assignment (overwrote `cands`) | Medium | Rewrote `_best_cpt` |
| Sweep claimed ±inf endpoints but did not add them | Medium | Always-weak / always-strong endpoints |
| `json.dumps` with NaN (invalid JSON) | Medium | `_json_safe` → null |
| Missing strong ⇒ `both_wrong=True` incorrectly | Low | Labels only when `has_strong` |
| Eval silent if `tau_star` tuned on fit | Medium | WARNING if `role != calib` |

---

## Logic checks (pass after fixes)

| Check | Status |
|-------|--------|
| Join key `query_id` | OK |
| Router inputs = φ + weak ψ only | OK |
| `needs_strong` = weak✗ ∧ strong✓ | OK |
| Gold not in \(R(q)\) | OK |
| Z-score μ/σ from **fit** only | OK |
| τ chosen on calib role (when used correctly) | OK |
| Eval applies frozen τ | OK |
| Directions: \(H\)↑ escalate, \(p_{\max}\)↓ escalate | OK |
| Smoke: `route rank` + `route sweep --role fit` | OK (dry-run) |

Fit dry-run CPT80 ≈ 0.34 for \(S_H\) (sensible order of magnitude). **Re-run sweep on calib** before trusting `tau_star.json` (current file may be from fit dry-run).

---

## Concept notes (not bugs)

1. **Top‑k ≈ MD family** — rank order is `margin, p_max, H, top2, C_*…`; top‑5 includes one weak \(C\). Collinearity ⇒ \(S_{\mathrm{md}}\approx S_H\).
2. **Rank uses default orientation AUROC**, not flipped — correct so directions stay consistent with `score.py`.
3. **Equal-weight after z-score** is rule-based, not learned \(\lambda\) — belongs in `rule_based/`; `learn_weight/` later.
4. **No random / always-\*** baselines in eval table yet** — add when writing paper tables.
5. **Empty top-level `feature_vector/`** folder is unused; real code is `signals/feature_vector.py`.

---

## Remaining polish (optional)

- [ ] Add always-weak / always-strong / random rows to `evaluate.py` report  
- [ ] Refuse `route eval` if `tau_star.role != "calib"` (hard fail vs warn)  
- [ ] Per-source τ ablation  
- [ ] Avoid double `load_joined("fit")` in sweep (cache once)  
- [ ] Delete / ignore dry-run fit `tau_star` before calib sweep  

---

## What to run next

```bash
# after calib weak+strong finish:
./run.sh route join-check --role calib
./run.sh route sweep --role calib    # overwrites fit dry-run tau_star
# after eval probes:
./run.sh route eval
```
