# Workflow: model-independent query complexity φ(q)

## 1. Research framing

### Core problem
Choose which LLM should answer query \(q\) to trade **quality vs cost**, using signals estimated **without** a large supervised map \(q \mapsto\) best model.

### Innovation (this stage)
Estimate **query complexity** \(C(q)\) from the **query text alone**, **before** any candidate-LLM generation (pre-inference / pre-execution).

Later stages add **model-dependent** signals (entropy, paraphrase uncertainty). This file covers **only** model-independent complexity.

### What “complexity” means here
Operational definition for routing:

> A query is more complex if a **weaker / cheaper** model is more likely to fail it, so the system should escalate to a **stronger / costlier** model.

We do **not** assume complexity = length. We approximate complexity with a vector \(\phi(q)\) and an optional scalar \(C(q) = g(\phi(q))\).

### Constraints
| Must | Must not |
|------|----------|
| Use query text (+ MC options if present, for length stats only) | Call candidate LLMs |
| Fit transforms on **fit** (never **eval**) | Train difficulty on Arena / preference labels |
| Disclose frozen tokenizer / embedder | Claim “model-free” if using ST embedder — say **candidate-independent** |

---

## 2. Complexity dimensions → features

| Dimension | Intuition | Feature block | Signals |
|-----------|-----------|---------------|---------|
| **D1 Surface load** | Longer / denser text → more to process | `structural` | token lengths, MATTR, compression_ratio |
| **D2 Semantic atypicality** | Far from fit distribution → harder / OOD-ish | `embedding_geometry` | centroid distance, low kNN similarity, high LOF; PCA coords |
| **D3 Cognitive / linguistic demand** | Reasoning / multi-hop / multi-domain wording | `linguistic_cues` | Bloom ordinal depth, multi-hop cues, domain breadth, requirement count |
| **D4 Task form** | MC vs free-form changes what “hard” looks like | `task_form` | `is_mc`, `n_choices`, `task_type` / `source` |

**Required v1:** D1 + D2 + D3 + D4.

EffGen-style precedent (adapted to QA): length, #requirements, domain breadth, reasoning depth (+ tools skipped for pure QA).

---

## 3. Outputs

For every `query_id`:

```text
φ(q) = {
  structural,          # D1
  linguistic_cues,     # D3
  task_form,           # D4
  embedding_geometry,  # D2
  complexity           # scalars derived from φ
}
```

### `complexity` scalars (always write these)

| Field | Definition |
|-------|------------|
| `C_length` | z-scored prompt (or question) length |
| `C_density` | combination of MATTR ↑ and compression_ratio ↓ (document formula in config) |
| `C_atypical` | e.g. `0.5 * centroid_distance + 0.5 * lof_score` (z-scored) |
| `C_linguistic` | e.g. normalized sum of reasoning + multi-hop + domain_breadth + n_requirements |
| `C_query` | **primary scalar**: weighted sum of \(C_*\) (weights in config; tune on **calib** only) |

Routing rule prototype (Stage 1):  
`if C_query >= τ_calib → strong else weak`  
(τ from calib; report on eval.)

---

## 4. Correct end-to-end workflow

```text
datasets/processed/corpus_v1/queries_{fit,calib,eval}.jsonl
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE A — Row-local complexity features (no shared fit) │
│  For each query:                                         │
│    structural      ← D1                                  │
│    linguistic_cues ← D3  (requirements, reasoning,       │
│                          multi-hop, domain breadth)      │
│    task_form       ← D4                                  │
└──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE B — Frozen embedding of canonical prompt          │
│  SentenceTransformer (id frozen in config)               │
│  cache: embeddings/{query_id}.npy                        │
└──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE C — Geometry = semantic atypicality (D2)          │
│  GeometryModel.fit( FIT embeddings ONLY )                │
│  → PCA≤3, centroid, kNN, LOF                             │
│  freeze artifacts/geometry.*                             │
│  transform(fit, calib, eval)                             │
└──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE D — Normalize + compose complexity                │
│  ZScoreModel.fit( FIT preferred, or CALIB )              │
│  never EVAL                                              │
│  compute C_length, C_density, C_atypical,                │
│          C_linguistic, C_query                           │
│  freeze artifacts/zscore.* + complexity_weights.yaml     │
└──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE E — Emit                                          │
│  signals/query/processed/{fit,calib,eval}.jsonl          │
│  signals/query/artifacts/manifest.json                   │
└──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE F — Validate complexity (then route)              │
│  F1 Hotpot meta.level vs C_*   (fit, no LLMs)            │
│  F2 After oracle: AUROC(C_query → needs_strong) (fit)    │
│  F3 τ on calib; PGR/CPT on eval per source               │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Feature contracts (minimal specs)

### 5.1 `structural` (D1)
- `prompt_token_len`, `question_token_len`
- `mean_option_token_len`, `std_option_token_len`, `question_option_ratio` (0 / null if no options)
- `mattr` (windowed type–token)
- `compression_ratio` (zlib len / raw len; **lower** ⇒ denser ⇒ higher complexity contribution)

### 5.2 `linguistic_cues` (D3) — **explicit complexity language**
| Key | Capture |
|-----|---------|
| `n_requirements` | counts of distinct ask markers (`;`, numbered steps, “and then”, multi-sentence questions) |
| `reasoning_depth_score` | max revised-Bloom level from process verbs/phrases on the stem (`0` or `2..6`; Remember/interrogatives omitted) |
| `multihop_score` | bridge cues (“both”, “as well as”, “which … also”, Hotpot-style) |
| `domain_breadth` | # distinct domain buckets with ≥1 keyword hit |
| `n_question_marks` / `n_sentences` | structural ask load |

### 5.3 `task_form` (D4)
- `is_mc` ∈ {0,1}
- `n_choices` (0 if free-form)
- `source`, `task_type` (from corpus; for stratified analysis)

### 5.4 `embedding_geometry` (D2)
- `pc1..pc3`, `centroid_distance`, `mean_knn_similarity`, `lof_score`  
Fit on **fit only**.

---

## 6. Leakage firewall

| Transform | Train on | Apply to |
|-----------|----------|----------|
| Geometry (PCA/kNN/LOF) | **fit** | fit, calib, eval |
| Z-score | **fit** (default) | all |
| Complexity weights / τ | **calib** | eval for reporting |
| Paper metrics | — | **eval only** |

Assert: `eval_ids ∩ geometry_train_ids = ∅`.

---

## 7. Experiments (complexity-centric)

| ID | Question | Data | Pass criterion (guidance) |
|----|----------|------|---------------------------|
| **E0** | Leakage / freeze artifacts | — | Asserts green |
| **E1a** | Do \(C_*\) track Hotpot `level`? | fit Hotpot | Spearman \(C_{\text{query}}\) vs level > length-only |
| **E1b** | Block ablation on Hotpot level | fit Hotpot | D3 and/or D2 add value over D1 |
| **E2** | Does \(C_{\text{query}}\) predict needs_strong? | fit + oracle | AUROC vs length baseline |
| **E3** | Routing utility | calib τ → eval | PGR/CPT per ARC/Hotpot/MMLU |
| **E4** | Geometry sensitivity | fit/eval | Stability across k / per-source geometry |

Always include **length-only** baseline so complexity ≠ “just long prompts.”

---

## 8. Artifact layout

```text
signals/query/
  WORKFLOW.md
  config.yaml
  artifacts/
    geometry.*
    zscore.json
    complexity_weights.yaml
    manifest.json
  processed/
    fit.jsonl
    calib.jsonl
    eval.jsonl
  embeddings/
```

### Record schema (processed jsonl)

```json
{
  "query_id": "...",
  "role": "fit",
  "source": "hotpotqa",
  "structural": {},
  "linguistic_cues": {},
  "task_form": {"is_mc": 0, "n_choices": 0},
  "embedding_geometry": {},
  "complexity": {
    "C_length": 0.0,
    "C_density": 0.0,
    "C_atypical": 0.0,
    "C_linguistic": 0.0,
    "C_query": 0.0
  }
}
```

---

## 9. Out of scope (next stages)

| Signal | Why not here |
|--------|----------------|
| Token / semantic entropy \(H(q\|M)\) | Needs candidate model |
| Paraphrase uncertainty | Needs generations |
| Preference-trained routers | Supervised \(q \to\) win |

Those compose with \(C(q)\) later:  
\(\text{score}(q,M) = \lambda_C C(q) + \lambda_H H(q|M) + \cdots\)

---

## 10. Paper statement

> We define model-independent query complexity via a feature vector \(\phi(q)\) that explicitly targets surface load, linguistic/reasoning demand, task form, and semantic atypicality relative to the fit distribution. A scalar \(C(q)\) is composed from these dimensions without candidate-LLM calls; geometric and normalization transforms are fit only on the fit split. We validate \(C(q)\) against known difficulty labels where available and against oracle escalate-or-not outcomes, then use calib-tuned thresholds for cost–quality routing on held-out eval sets.

---

## 11. Implementation order

1. Config: lexicons (D3), structural params, embedder, complexity weights init.  
2. Stage A extractors → raw jsonl.  
3. Stage B–C embeddings + geometry (fit-only).  
4. Stage D z-score + \(C_*\) / \(C_{\text{query}}\).  
5. E0 + E1 (Hotpot levels) **before** choosing final LLM pool.  
6. After oracle: E2–E3 for routing tables.
