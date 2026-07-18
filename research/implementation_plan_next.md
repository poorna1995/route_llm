# Detailed Implementation Plan (Next Steps)

How to execute the end-to-end routing plan **on top of existing code**, with concepts, dependencies, and file linkages.

Related docs:
- Concept / pipeline: [`end_to_end_routing_plan.md`](end_to_end_routing_plan.md)
- Signal science (fit): [`routing_signal_analysis.md`](routing_signal_analysis.md)
- Problem math: [`problem_formulation.md`](problem_formulation.md)
- Signal stage workflow: [`../signals/query_model/WORKFLOW.md`](../signals/query_model/WORKFLOW.md)

---

## 1. Concept map (what depends on what)

```text
datasets/config.yaml
        │
        ▼
datasets/build.py  ──►  corpus_v1/queries_{fit,calib,eval}.jsonl
        │                         │
        │                         ├──────────────────────────────┐
        ▼                         ▼                              ▼
signals/query/build.py      signals/query_model/build.py    (gold in corpus)
        │                         │
        ▼                         ▼
query/processed/            query_model/processed/
  {fit,calib,eval}.jsonl      {role}_{weak|strong}.jsonl
  φ(q) / C_*                    ψ(q,M) + pred + cost
        │                         │
        └────────────┬────────────┘
                     ▼
           signals/feature_vector.py
                     │  z = [φ | ψ]
                     ▼
           ★ router/  (existing folder — use this, not signals/routing/)
              rule_based/   ← current focus (thresholds, top-k)
              learn_weight/ ← deferred λ
                     │
                     ▼
           router/artifacts/rule_based/ + research/figures
```

**Firewall (do not break):**
| Split | May use for |
|-------|-------------|
| fit | rank features, z-score μ/σ, AUROC discovery |
| calib | choose \(\tau^\star\), optional equal-weight sets |
| eval | report once with frozen \(\tau^\star\) |

Gold / `needs_strong` → **metrics only**, never an input to \(R(q)\).

---

## 2. Existing building blocks (reuse, don’t rewrite)

| Module | Path | Provides | Next stage uses it for |
|--------|------|----------|------------------------|
| Corpus | `datasets/processed/corpus_v1/queries_*.jsonl` | `query_id`, `prompt`, `gold`, `metric`, `source`, `role` | Join key + correctness |
| Complexity | `signals/query/processed/{fit,calib,eval}.jsonl` | `complexity.C_*` (= \(\phi\)) | Query-only features |
| Geometry artifacts | `signals/query/artifacts/{zscore,geometry,complexity_weights}.json` | Fit-only transforms (already applied in rows) | Do **not** refit on calib/eval |
| Probes | `signals/query_model/processed/{role}_{weak\|strong}.jsonl` | `answer_scores`, `pred`, `cost` | \(\psi\), answers, latency |
| Backend | `signals/query_model/backend.py` | HF/mock + latency/token trace | Probing only |
| Build | `signals/query_model/build.py` | Upsert probes per role×model | Phase B compute |
| Store | `signals/query_model/store.py` | Atomic write, backups, paths | Safe probes |
| Analyze | `signals/query_model/analyze.py` | `join_rows`, `is_correct`, `auroc`, slices | **Import** into router |
| Feature vector | `signals/feature_vector.py` | `PHI_KEYS`, `PSI_KEYS`, `z_vectors` | Dense \(z\) |
| CLI | `run.sh` | `complexity`, `query-model`, `query-model-analyze`, `restore` | Add `route` later |
| Pool | `datasets/config.yaml` → `local_strong` | weak=8B, strong=Qwen-32B | Keep experiment flag |

---

## 3. Data linkage contract (join key)

Everything joins on **`query_id`**:

```text
corpus[query_id]     → gold, metric, source, prompt
complexity[query_id] → φ (C_*)
weak[query_id]       → ψ, pred_weak, cost_weak
strong[query_id]     → pred_strong, cost_strong   (scoring / oracle only)
```

Correctness (reuse `analyze.is_correct`):
```text
weak_ok   = is_correct(pred_weak, gold, metric)
strong_ok = is_correct(pred_strong, gold, metric)
needs_strong = (not weak_ok) and strong_ok   # label only
```

Router decision uses **only** \(\phi\) + weak \(\psi\) (and \(\tau\) from calib).

---

## 4. Phase-by-phase implementation

### Phase B — Probe calib & eval (compute; no new module)

**Dependency:** GPU free; `fit_weak` / `fit_strong` already done; do **not** run mock against fit.

**Order (one model at a time — memory):**
```bash
# 1) calib weak (~499)
./run.sh query-model --backend hf --experiment local_strong \
  --roles calib --model-roles weak --no-paraphrase --progress-every 50

# 2) calib strong
./run.sh query-model --backend hf --experiment local_strong \
  --roles calib --model-roles strong --no-paraphrase --progress-every 50

# 3) eval weak (~2597)
./run.sh query-model --backend hf --experiment local_strong \
  --roles eval --model-roles weak --no-paraphrase --progress-every 50

# 4) eval strong
./run.sh query-model --backend hf --experiment local_strong \
  --roles eval --model-roles strong --no-paraphrase --progress-every 50
```

**Outputs (new files):**
```text
signals/query_model/processed/calib_weak.jsonl
signals/query_model/processed/calib_strong.jsonl
signals/query_model/processed/eval_weak.jsonl
signals/query_model/processed/eval_strong.jsonl
```

**Verify after each:**
```bash
./run.sh query-model-analyze --role calib --model-role weak
./run.sh query-model-analyze --role calib --model-role strong
# same for eval
```

**Links:** `build.py` → `store.signal_path` → `{role}_{model_role}.jsonl`; backups automatic.

**Cost fields:** new rows get `cost.{latency_ms,prompt_tokens,...}`. Old fit rows may lack `cost` (OK).

---

### Phase C — Join + rank (small new code)

**Package (uses existing `router/` folders):**
```text
router/
  join.py              # shared: 4-stream join + z + labels
  metrics.py           # shared: PGR / CPT
  rule_based/          # ★ implement now
    config.yaml
    rank.py / score.py / sweep.py / evaluate.py / cli.py
  learn_weight/        # deferred λ (README stub only)
  artifacts/rule_based/
```

**`join.py` logic (reuse analyze):**
```python
# For role in (fit, calib, eval):
#   corpus  = corpus_v1/queries_{role}.jsonl
#   phi     = query/processed/{role}.jsonl
#   weak    = query_model/processed/{role}_weak.jsonl
#   strong  = query_model/processed/{role}_strong.jsonl
#   for each query_id in intersection:
#       z = z_vectors(query_row=phi, model_row=weak)
#       labels from is_correct(weak/strong)
#       attach cost_weak / cost_strong if present
```

**`rank.py` (fit only):**
- For each key in `Z_KEYS`, AUROC vs `needs_strong` (use `analyze.auroc`, directions: \(H\)=+1, \(p_{\max}\)/margin/top2=−1, all \(C_*\)=+1; also try flip for \(C_*\) and keep best **only for reporting**, not for silent flipping on eval without documenting).
- Write `signals/routing/artifacts/rank_fit.json`.
- Define sets: `S_H`, `S_p`, `S_top2`, `S_top4`, `S_top5`, `S_md`, `S_all`.

**CLI (after implement):**
```bash
./run.sh route rank --role fit
```

**Can start ranking on fit now** (before calib probes) for the figure; τ needs calib.

---

### Phase D — Scoring rule (reuse feature_vector)

**`score.py`:**
1. Fit μ, σ of each feature on **fit** weak+φ join (save `artifacts/zscore_routing.json`).
2. Orient: \(\tilde H = z(H)\), \(\tilde p_{\max} = -z(p_{\max})\), etc.
3. For feature set \(\mathcal{F}\):  
   \(s(q)=\frac{1}{|\mathcal{F}|}\sum_{j\in\mathcal{F}}\tilde z_j(q)\)
4. \(R(q)=\mathrm{strong}\) iff \(s(q)\ge\tau\).

Single-feature \(S_H\): \(s(q)=H\) (raw or z-scored — pick one and stick to it; raw \(H\) is fine for single-signal).

---

### Phase E — Calib sweep → freeze τ

**`sweep.py` inputs:** joined calib rows + feature set + score function.  
**Outputs per τ:** strong-call rate \(\alpha\), accuracy, PGR.

```text
PGR = (r_router - r_weak) / (r_strong - r_weak)
CPT(x%) = min α such that PGR ≥ x/100
```

**Selection policy (declare in config):**
```yaml
# signals/routing/config.yaml
select: cpt80          # or max_pgr_at_alpha with alpha_max: 0.25
feature_sets: [S_H, S_p, S_top2, S_top4, S_top5, S_md, S_all]
```

Write `artifacts/tau_star.json`:
```json
{"S_H": {"tau": 0.42, "alpha_calib": 0.21, "pgr_calib": 0.80}, ...}
```

**CLI:**
```bash
./run.sh route sweep --role calib
```

---

### Phase F — Eval report (frozen)

**`evaluate.py`:**
- Load `tau_star.json` (no re-search).
- Score eval with same fit z-score stats + sets.
- Table: accuracy, PGR, α, mean latency/tokens of path (weak always + strong if escalated).
- Curves: optional re-sweep on eval **for plotting only**; mark calib \(\tau^\star\) point.

**CLI:**
```bash
./run.sh route eval
./run.sh route curves --role eval   # figure only
```

**Figures →** `research/figures/fig_routing_curve.png`, `fig_feature_rank.png`.

---

## 5. Wire into `run.sh`

Add case (same pattern as `query-model-analyze`):
```bash
  route)
    shift || true
    exec "$ROOT/.venv/bin/python" "$ROOT/signals/routing/cli.py" "$@"
    ;;
```

Subcommands: `rank | sweep | eval | curves | join-check`.

---

## 6. Dependency order (what blocks what)

```text
[B] calib/eval probes ──────────────────────────────┐
                                                     │
[A] fit probes ✓  →  [C] rank (fit) ✓ can start now │
                                                     ▼
                         [D] score + zscore fit
                                                     │
                         [E] sweep calib  ◄──────────┘ (needs B calib)
                                                     │
                         [F] eval report  ◄── needs B eval + E tau_star
```

**Minimum path to first routing number:**
1. Probe **calib** weak+strong  
2. Implement `join` + `sweep` for `S_H` only  
3. Probe **eval** weak+strong  
4. Eval with frozen \(\tau\)  
5. Then add top‑k ablations

---

## 7. Suggested file APIs (minimal)

```python
# join.py
def load_joined(role: str) -> list[dict]: ...

# rank.py
def rank_features(joined_fit) -> dict: ...

# score.py
def fit_zscore(joined_fit) -> dict: ...
def score_row(row, feature_set, zstats) -> float: ...

# sweep.py
def sweep_tau(joined, feature_set, zstats) -> list[dict]: ...
def pick_tau(curve, policy) -> dict: ...

# evaluate.py
def eval_frozen(joined_eval, tau_star, zstats) -> dict: ...
```

Reuse from `analyze.py`: `read_jsonl`, `is_correct`, `is_mc_row`, `auroc`, `join_rows` (extend or wrap for dual weak/strong).

Reuse from `feature_vector.py`: `PHI_KEYS`, `PSI_KEYS`, `Z_KEYS`, `z_vectors`, `as_floats`.

---

## 8. Config linkage

| Config | Role |
|--------|------|
| `datasets/config.yaml` | splits, `local_strong` pool |
| `signals/query/config.yaml` | φ weights (already baked into rows) |
| `signals/query_model/config.yaml` | hotpot n, paraphrase, **pricing** for `$` |
| `signals/routing/config.yaml` **(new)** | feature sets, select policy, paths |

Experiment flag must stay **`local_strong`** so weak/strong ids match fit.

---

## 9. Latency / cost in the routing stage

| Quantity | Source | Use |
|----------|--------|-----|
| Weak probe cost | `weak.cost` | Always paid for MD routing |
| Strong call cost | `strong.cost` | Paid only if \(R\)→strong (for system cost) |
| Architectural | 1× MC / \(n\)× Hotpot | Signal comparison table |
| System latency est. | `latency_weak + 1[escalate]*latency_strong` | Eval table column |

Fit without `cost`: leave blank or mark “n/a (pre-trace)”.

---

## 10. Checklist (execution order)

### Compute
- [ ] `calib` weak probe  
- [ ] `calib` strong probe  
- [ ] verify calib via `query-model-analyze`  
- [ ] `eval` weak probe  
- [ ] `eval` strong probe  
- [ ] verify eval  

### Code (`router/`)
- [x] `join.py` / `metrics.py` — shared  
- [x] `rule_based/` rank · score · sweep · evaluate · cli  
- [x] `./run.sh route`  
- [ ] Run `route sweep` after calib weak+strong  
- [ ] Run `route eval` after eval probes  
- [ ] `learn_weight/` (later)

### Paper artifacts
- [ ] rank bar figure  
- [ ] routing curves (\(S_H\) vs top‑k vs random)  
- [ ] update `routing_signal_analysis.md` with calib/eval  

---

## 11. What not to do

- Do not choose \(\tau\) on fit or eval.  
- Do not train a preference router in this stage.  
- Do not overwrite `fit_weak` with `--limit` mock runs.  
- Do not load weak and strong HF models at once if memory is tight — sequential `--model-roles`.  
- Do not refit complexity geometry on calib/eval (artifacts already frozen on fit).

---

## 12. First concrete command to run now

```bash
./run.sh query-model --backend hf --experiment local_strong \
  --roles calib --model-roles weak --no-paraphrase --progress-every 50
```

While that runs, implement `signals/routing/join.py` + `rank.py` against **fit** (already complete) so ranking/figures are ready when calib finishes.
