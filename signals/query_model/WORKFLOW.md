# Workflow: model-dependent signals (query × model)

## 1. Research framing

### Core problem
Choose which LLM should answer query \(q\) to trade **quality vs cost**, using **unsupervised signals** — not a preference-trained map \(q \mapsto M\) (RouteLLM, Hybrid LLM, RouterBench).

### Where this stage sits

```text
Stage 0 (done)     Stage 1 (this file)        Stage 2 (later)
─────────────────────────────────────────────────────────────────
C(q) from query    H(Y|q,M), U(q|M) from        Rules / τ / λ /
text alone         query × model probe          routing curves
signals/query/     signals/query_model/       (separate doc)
```

**Professor order:** (1) extract signals from query and from query×model → (2) test which predict mistakes → (3) **then** route.

**Implementation runbook:** §15 (repo map, commands, packages, existing code).

### What “model-dependent” means
Information that exists only after we **probe** candidate model \(M\) on query \(q\):

| Signal | Symbol | Source |
|--------|--------|--------|
| Answer uncertainty | \(H(q \mid M) = H(Y \mid q, M)\) | Option-letter distribution entropy (MC); discrete cluster entropy (Hotpot) |
| Paraphrase instability | \(U(q \mid M)\) | Same probe on reworded **question text** |

Model-independent complexity \(C(q)\) is defined in [`signals/query/WORKFLOW.md`](../query/WORKFLOW.md). This stage **does not** recompute \(C(q)\); analysis **joins** the two jsonl streams later.

### Positioning vs external work

See [`research/confidence_routing_alignment.md`](../../research/confidence_routing_alignment.md) for full detail. Short map:

| Lane | Examples | Our relation |
|------|----------|--------------|
| **Learned router** | RouteLLM, Hybrid LLM, RouterBench | **Contrast** — they train \(q\to M\) from preferences/outcomes; we do not |
| **Generator UQ cascade** | Chuang et al. arXiv:2502.04428 | **Closest neighbor** — SLM uncertainty → offload; we use option-letter \(H\) / `p_max` (not trained probes) |
| **Production confidence** | Sawant (entropy zones, verifier) | **Borrow** alignment mindset + future calibration; skip verifier in v1 |

**Honest cost framing:** \((q,M)\) signals require calling \(M\) (at least an option-letter probe or short answer samples). This is **not** RouteLLM-style query-only routing. On MC, one probe yields `pred` + all `answer_scores` — no second full generation for correctness checks.

### Plain language & terminology

| Prefer | Avoid (unless citing literature) |
|--------|----------------------------------|
| **answer uncertainty** \(H(Y \mid q, M)\) | “answer-distribution uncertainty” (vague) |
| **option-letter distribution** / **choice entropy** (MC) | “closed-set letter probabilities”; bare “letter distribution” |
| **discrete cluster entropy** (Hotpot v1) | “grouped-answer entropy”; bare **SE** |
| **semantic-entropy family** (when citing Kuhn / Zhang) | claiming full **semantic entropy** in v1 (exact-match only) |
| question text, answer choices | “stem” without context |
| score option letters A/B/C/D | “teacher-forced” without explanation |
| cluster sampled answers by meaning / string | “SE” with no gloss |

**Symbol:** \(H(q \mid M)\) is shorthand for \(H(Y \mid q, M)\) where \(Y\) is the model’s answer (option letter or short free-form string).

**Output block name:** `answer_scores` — numeric fields derived from the probe (`H`, `p_max`, `margin`, …).

---

## 2. Adapted principles, methods, and approach

This section translates [`research/confidence_routing_alignment.md`](../../research/confidence_routing_alignment.md) into **our model-dependent stage only**. Routing policy (τ, λ, cascade) is **explicitly deferred** — see §14.

### 2.1 Principles we adopt (model-dependent scope)

| # | Principle | Source lane | How we use it **now** |
|---|-----------|-------------|-------------------------|
| **P1** | **Generator-side uncertainty matters** | Chuang, Sawant | \(H(Y\mid q,M)\), `p_max`, `margin` from the **weak (and strong) model’s** option-letter / answer-cluster probe — not a separate trained router |
| **P2** | **Alignment ≠ calibration** | Chuang Finding ❶ | A confident wrong answer is possible; we test **ROC-AUC** (confidence ranks correctness), not “does 0.7 mean 70% accurate” yet |
| **P3** | **One probe, many scores** | Chuang avg token prob + Sawant logprob family | Single MC option-letter pass → full `answer_scores` + `pred`; no duplicate inference per monotone metric |
| **P4** | **Input stability is a separate signal** | Our design (not in Chuang’s 8 UQ) | \(U(q\mid M)\) via **paraphrase of question text** — not same-prompt self-consistency |
| **P5** | **Honest interaction cost** | vs RouteLLM / Accredian | Probing \(M\) is required; we document it and still beat “always call strong” later — we do **not** claim query-only routing |
| **P6** | **Signals before policy** | Professor + Accredian caution | Extract → validate (S1–S8) → **then** router; no Arena labels, no hidden-state probes in this stage |
| **P7** | **Cost-gated extras** | Sawant zones (future) | Paraphrase \(U\) and Hotpot multi-sample run for **all fit rows** in v1 (analysis); in **router** stage, gate expensive paths to mid-confidence band only |

**Principle we do not adopt in this stage:** learned \(q\to M\) from preferences (RouteLLM), trained hidden-state probes (Chuang top performer), verbalized confidence (Chuang Finding ❷), verifier accept/reject (Sawant).

### 2.2 External methods → our adaptation

Status key: **✓** adapted now · **~** partial · **○** deferred · **✗** not adopted (contrast or rejected).

| External method | Literature | Status | **Our adaptation (Stage 1)** | **Router stage (later)** |
|-----------------|------------|--------|------------------------------|---------------------------|
| Average token prob (MC option letter) | Chuang §3.1 | **~** | `p_max`, `surprisal` from **full** renormalized option-letter softmax (not chosen token only) | Low `p_max` → escalate |
| Token / sequence perplexity | Chuang | **~** | `perplexity_H` (= \(e^H\)) on option-letter distribution | Same as high \(H\) |
| Choice / answer entropy (MC) | COLING 2025; Sawant analog | **~** | `H`, `margin`, `top2_mass` on A/B/C/D — **answer-space**, not token-stream | High \(H_w\) → escalate |
| Jaccard / sample consistency | Chuang #6 | **~** | Hotpot: **discrete cluster entropy** over \(n\) samples (exact-match clusters v1) | Optional mid-band check |
| Self-consistency (same prompt) | Sawant, Chuang #6 | **✗** (MC) | **Not v1** on MC; overlaps Hotpot path | Gated sampling if needed |
| Verbalization 1s/2s | Chuang #4–5 | **✗** | **Skip** — Finding ❷: poor SLM routing alignment | — |
| p(True) follow-up | Chuang #3 | **✗** | **Skip** — extra generation round | — |
| Trained / OOD probe | Chuang #7–8 | **✗** | **Skip** — supervised; contrast in paper | — |
| Paraphrase disagreement | **Ours** | **✓** | \(U(q\mid M)\) — frozen T5 rewrites of **question text** | High \(U_w\) → escalate |
| Query complexity \(C(q)\) | Sawant tier-0; **ours** | **~** | Join in S4 only (`signals/query/`) — unsupervised \(\phi(q)\), not Arena-trained | \(C(q)\ge\tau_C\) → strong |
| ROC-AUC alignment metric | Chuang RQ1 | **✓** | S2, S3 primary metric on fit | — |
| Greedy letter scoring (temp 0) | Chuang | **✓** | MC probe decoding | — |
| Calibration data / pooled CDF | Chuang §4 | **○** | **Defer** | Set \(\tau\) on calib pool without target-task labels |
| Routing curves (acc vs offload) | Chuang RQ1 | **○** | **Defer** | PGR / CPT on eval |
| Confidence zones (high / uncertain / low) | Sawant | **○** | **Defer** | Zone policy on calib |
| Verifier model | Sawant | **○** | **Defer** | Low-confidence band only |
| Isotonic / ECE calibration | Sawant | **○** | **Defer** | Post-hoc on calib |
| RouteLLM (SW, MF, BERT, causal LLM) | Accredian | **✗** | **Contrast** — preference-trained \(q\to M\) | Baseline comparison |
| Hybrid LLM (DeBERTa + quality-gap labels) | Accredian | **✗** | **Contrast** — supervised at train, query-only at serve | Baseline comparison |
| IRT-Router / MixLLM / ICL-Router | related_work | **✗** | Out of scope — N-way supervised routers | — |
| Zhang et al. semantic entropy (training labels) | related_work | **~** | Hotpot cluster entropy as **live signal** (v1 exact-match; not full SE) | — |

### 2.3 Our approach (query × model interaction)

**Core loop for this stage** — no routing decision:

```text
For each query q and each model M ∈ {weak, strong}:
  1. (Optional) Build k paraphrases of question text
  2. Probe M:
       MC     → option-letter logprobs → answer_scores + pred
       Hotpot → n samples → cluster entropy H → answer_scores + pred
  3. On each paraphrase surface → collect preds → U
  4. Write (q, M) record to processed/*.jsonl
  5. On fit: measure AUROC(signal → weak_wrong | needs_strong)
```

**What “probe” means in practice:**

| Task | One call to M gives | Correctness check |
|------|---------------------|-------------------|
| ARC / MMLU | Option-letter distribution + `pred` | `pred` vs gold (no second generation) |
| Hotpot | `n` short answers + cluster entropy \(H\) | Match / judge later in S7 |

**Weak vs strong in signal phase:** score **both** models on every query so we can study S3 (needs_strong) and S5 (gap). Cascade “weak only, then maybe strong” is **router logic**, not signal extraction.

### 2.4 Evaluation approach (Chuang RQ1 without routing curves)

Adapt Chuang’s **benchmarking** mindset, not their offload policy:

| Step | Action | Split |
|------|--------|-------|
| 1 | S0 smoke — schema, mock backend | any |
| 2 | S1 — `pred` accuracy per model | **fit** |
| 3 | S2 — AUROC per **signal family** → `weak_wrong` | **fit** |
| 4 | S3 — AUROC → `needs_strong` | **fit** |
| 5 | S4 — join \(C(q)\); ΔAUROC | **fit** |
| 6 | S5 — weak−strong gap features | **fit** |
| 7 | S6 — \(U\) vs \(H\) redundancy | **fit** |
| 8 | S7 — Hotpot cluster entropy \(H\) vs `meta.level` / match | **fit** |
| 9 | S8 — ablation weak model | **fit** |
| 10 | Confirm direction on **calib** | calib (no τ tuning yet) |
| 11 | Hold **eval** for paper tables | eval |

**Success gate before router:** ≥1 model-dependent family (typically `H` or `p_max` or `U`) beats chance AUROC on S2/S3 on pooled MC fit, **or** a clear per-source story is documented.

**Metrics we borrow now:** ROC-AUC (uncertainty–correctness alignment), Spearman, accuracy.  
**Metrics we defer:** routing ratio curves, PGR, CPT, ECE/isotonic calibration.

### 2.5 Firewall (model-dependent stage)

| Do now | Defer to router |
|--------|-----------------|
| Compute \(H, U\) on fit/calib/eval | Choose \(\tau_H, \tau_U, \tau_C\) |
| Pick winning **family** on fit analysis | Combine families with \(\lambda\) |
| Use gold to **label** weak_wrong, needs_strong | Implement cascade weak→strong |
| Score both weak and strong | “Probe weak only” production path |
| Document probe cost per query | PGR / CPT / strong-call rate |

Gold and oracle labels **never** train the paraphraser, backend, or a RouteLLM-style router.

### 2.6 Contrast sentence (for methods section)

> We adapt the **generator-uncertainty** line (Chuang et al.; Sawant) without preference-trained routers (RouteLLM) or supervised hidden-state probes: model-dependent **answer uncertainty** \(H(Y \mid q, M)\), `p_max`, margin, and paraphrase \(U\) are **defined**, **extracted** with a single option-letter probe (MC) or clustered answer samples (Hotpot), and **validated** via uncertainty–correctness alignment before any routing policy is fixed.

### 2.7 Related-work audit (Chuang 8 + strategies)

Full positioning lives in [`research/confidence_routing_alignment.md`](../../research/confidence_routing_alignment.md). This subsection is the **Stage 1 scorecard** — what we took from each lane vs what is original.

#### Chuang’s eight UQ methods (arXiv:2502.04428, Table 1)

| # | Chuang method | Status | What we do |
|---|---------------|--------|------------|
| 1 | Average token prob (MC option letter) | **~** | `p_max`, `surprisal`, `inv_p_max` from full renormalized option-letter softmax |
| 2 | Perplexity | **~** | `perplexity_H` = \(e^H\) on option-letter distribution (monotone with \(H\)) |
| 3 | p(True) | **✗** | Skipped — extra generation round |
| 4 | Verbalization-1s | **✗** | Skipped — Finding ❷: poor SLM routing alignment |
| 5 | Verbalization-2s | **✗** | Same as #4 |
| 6 | Jaccard degree (5 samples, same prompt) | **~** | Hotpot only: **discrete cluster entropy** over \(n\) answers (exact-match v1; not Jaccard matrix) |
| 7 | Trained probe (hidden states) | **✗** | Contrast — strongest in their benchmark but supervised |
| 8 | OOD probe | **✗** | Contrast — supervised |

**Chuang strategies (not separate UQ methods):**

| Strategy | Status | Our status |
|----------|--------|------------|
| Uncertainty–correctness alignment (ROC-AUC) | **✓** | S2, S3 primary metric |
| Greedy letter scoring (temp 0) | **✓** | MC probe decoding |
| Routing curves (accuracy vs offload rate) | **○** | Router §14 |
| Calibration data (30 bins, 10%, leave-one-out) | **○** | Router §14; analogous to pooled **calib** split |
| “Seek stronger if low confidence” cascade | **○** | Router policy, not signal extraction |

**Chuang gap we fill:** \(C(q)\) (query-only, Stage 0) and \(U(q\mid M)\) (paraphrase of **question**) are **not** among their eight methods.

#### Sawant methods and strategies

| Component | Status | Our status |
|-----------|--------|------------|
| Token entropy (mean over generated tokens) | **~** | Option-letter / answer-space \(H\) on MC — same *role*, different *estimator* |
| Logprob thresholds (first / critical tokens) | **~** | `p_max`, `margin`, `surprisal` on option-letter distribution |
| Self-consistency (N same-prompt samples) | **~** | Hotpot multi-sample only; not on MC in v1 |
| Verifier model | **○** | Deferred §14 |
| Calibration (isotonic, ECE) | **○** | Deferred §14 |
| Confidence zones (high / uncertain / low) | **○** | Deferred §14 |
| Complexity tier before confidence | **~** | \(C(q)\) in `signals/query/`; joined in S4, not used to route yet |

**Principle adapted:** output **(decision signal, confidence)** — we separate \(C(q)\) (query), \(H(Y\mid q,M)\) (answer uncertainty), and \(U\) (input stability), validate each before combining.

#### Accredian / RouteLLM / related_work

| Method / strategy | Status | Our status |
|-------------------|--------|------------|
| RouteLLM — SW ranking | **✗** | Contrast — preference-trained, query-only |
| RouteLLM — matrix factorization | **✗** | Contrast |
| RouteLLM — BERT classifier | **✗** | Contrast |
| RouteLLM — causal LLM classifier | **✗** | Contrast |
| Hybrid LLM — quality-gap labels + DeBERTa | **✗** | Contrast |
| Hybrid LLM — multi-sample for label building | **✗** | We sample at **inference** for \(H\)/\(U\), different purpose |
| Zhang et al. — semantic entropy for training labels | **~** | Hotpot cluster entropy related; we use it as a **live** signal (v1 exact-match) |
| IRT-Router / MixLLM / ICL-Router | **✗** | Out of scope |
| RouterBench eval harness (PGR, CPT) | **○** | Router stage baselines |

#### Original to our research (not adapted from Chuang’s 8)

| Component | Notes |
|-----------|-------|
| \(U(q\mid M)\) — paraphrase **question** disagreement | Not in Chuang’s 8; distinct from same-prompt consistency |
| \(C(q)\) — unsupervised query complexity | Stage 0; not Arena-trained tier classifier |
| Signal-family ablation (entropy vs confidence vs margin vs \(U\)) | S2 / S8 protocol |
| Weak−strong gap \(H_w - H_s\) | S5 analysis |
| Join \(C + H + U\) before any router | S4; literature usually uses one signal type |
| Signals-first, router-second firewall | Professor order; §2.5, §8 |

#### Summary scorecard

```text
ADAPTED NOW (Stage 1 — model-dependent signals)
───────────────────────────────────────────────
✓ Option-letter H, p_max, margin, top2_mass      ← Chuang #1–2 + Sawant logprob idea
✓ perplexity_H (monotone transform)              ← Chuang perplexity family
✓ Hotpot discrete cluster entropy over samples   ← Chuang #6 + semantic-entropy family (partial)
✓ ROC-AUC alignment metric                       ← Chuang RQ1
✓ Greedy MC option-letter probe                  ← Chuang decoding discipline
✓ Paraphrase U                                   ← OURS (not in Chuang 8)

DEFERRED (router stage — §14)
─────────────────────────────
○ Cascade weak → strong if low confidence         ← Chuang + Sawant
○ Calibration data / pooled τ on calib          ← Chuang §4
○ Confidence zones, isotonic calibration          ← Sawant
○ Verifier                                        ← Sawant
○ Routing curves, PGR, CPT                        ← Chuang + RouterBench culture
○ Cost-gated U / multi-sample on mid-band only    ← Sawant P7

NOT ADOPTED (contrast or rejected)
──────────────────────────────────
✗ Trained / OOD probe                             ← Chuang #7–8
✗ Verbalization 1s/2s                             ← Chuang #4–5
✗ p(True)                                         ← Chuang #3
✗ RouteLLM / Hybrid / IRT routers                 ← Accredian / related_work
✗ Same-prompt self-consistency on MC              ← Sawant / Chuang #6 on MC
✗ Open-ended token entropy on MC tasks            ← Sawant (wrong tool for ARC/MMLU)
```

**Paper caveat:** Chuang’s best methods (#7–8 trained probes) are supervised and score higher on AUROC than perplexity alone. We argue for **unsupervised** \(H\), `p_max`, and \(U\) as the right tradeoff for our framing — and report whether they are **good enough** on S2/S3 before building the router.

---

## 3. Signals we extract

For each pair \((q, M)\):

### 3.1 Answer uncertainty \(H(Y \mid q, M)\)

\(Y\) is the model’s answer (option letter on MC; short free-form string on Hotpot). We use one symbol for both task types but **different estimators** (cf. Chuang et al. token-probability UQ on MC vs sample-consistency / semantic-entropy family on free-form).

**MC (ARC, MMLU):** one forward pass scoring next-token likelihood of each option letter.

1. Build instruct prompt with question + choices.
2. Read logprobs (or logits) for letters `A`…`D` (`E` if present).
3. Softmax → renormalized probabilities \(p_i\) over options.
4. \(H = -\sum_i p_i \log p_i\) — **Shannon entropy of the option-letter distribution** (choice entropy; cf. MSP / first-token MC UQ literature).
5. Derive other fields from the **same** distribution:

| Field | Formula | Family |
|-------|---------|--------|
| `H` | \(-\sum_i p_i \log p_i\) | entropy |
| `p_max` | \(\max_i p_i\) | confidence |
| `pred` | \(\arg\max_i p_i\) | — |
| `margin` | \(p_{(1)} - p_{(2)}\) | margin |
| `top2_mass` | \(p_{(1)} + p_{(2)}\) | top-2 |
| `perplexity_H` | \(e^{H}\) | entropy (monotone) |
| `inv_p_max` | \(1/p_{\max}\) | confidence (monotone) |
| `surprisal` | \(-\log p_{\max}\) | confidence (monotone) |
| `probs` | \(\{L: p_L\}\) | full distribution |

**Literature link (MC):** Chuang et al. use **average token probability** of the chosen option letter; we compute **full distribution entropy** over renormalized option tokens — same token-probability family, closer to choice entropy (Biderman et al.; COLING 2025 MC uncertainty). Answer-space, not open-ended token-stream entropy (Sawant).

**Free-form (Hotpot):** no option letters.

1. Sample \(n\) short answers (\(T>0\), fixed in config).
2. Normalize text; cluster by exact match (v1).
3. **Discrete cluster entropy** = \(H(Y \mid q, M)\) (cluster-assignment entropy when answer likelihoods are unavailable; Kuhn et al.).
4. Store `pred` = mode cluster; letter-only fields `null`.

**Literature link (Hotpot):** sample-consistency / Jaccard-degree UQ (Chuang #6); **semantic-entropy family** (Kuhn et al.; Zhang et al.) — v1 uses exact-match clusters, not bidirectional entailment; upgrade to embedding/NLI clusters in v2 before claiming full semantic entropy.

### 3.2 Signal families

Compare **families**, not every monotone duplicate:

| Family | Members | One representative for S2 |
|--------|---------|---------------------------|
| Entropy | `H`, `perplexity_H` | `H` |
| Confidence | `p_max`, `surprisal`, `inv_p_max` | `p_max` |
| Margin | `margin` | `margin` |
| Top-2 | `top2_mass` | `top2_mass` |
| Paraphrase | `U` | `U` |

Chuang Finding ❷: **verbalized** confidence is a poor routing signal on small models — we do **not** use it.

### 3.3 Paraphrase uncertainty

Measures **input** stability (not answer self-consistency on identical prompt):

1. Paraphrase the **question text** only; keep choices fixed.
2. Frozen paraphraser (not \(M\)); default \(k=3\) surfaces (original + 2 rewrites).
3. Run the **same** probe on each surface → predictions \(\hat y_1,\ldots,\hat y_k\).
4. \(U = 1 -\) (fraction agreeing with mode prediction).

Primary `answer_scores` always use the **original** wording.

**Literature link:** distinct from Chuang’s same-prompt multi-sample consistency; closer to input-robustness / paraphrase disagreement (our contribution vs their eight UQ methods).

---

## 4. Model pool & corpus

From [`datasets/config.yaml`](../../datasets/config.yaml):

| Experiment | Weak | Strong | Role |
|------------|------|--------|------|
| **primary** | `llama-3.1-8b-instruct` | `llama-3.1-70b-instruct` | main tables |
| **ablation** | `qwen2.5-7b-instruct` | `llama-3.1-70b-instruct` | one cross-family row (S8) |

**Signal phase:** compute \((q,M)\) for **both** weak and strong on each query (enables S3, S5). **Routing phase (later):** may probe weak only and escalate — cascade is not implemented here.

**Sources:** `arc_challenge`, `hotpotqa`, `mmlu` — splits `fit` / `calib` / `eval` per corpus config.

**Inputs:** `datasets/processed/corpus_v1/queries_{fit,calib,eval}.jsonl`  
**Complexity join (analysis):** `signals/query/processed/{fit,calib,eval}.jsonl`

---

## 5. End-to-end pipeline

```text
queries_{fit,calib,eval}.jsonl
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE A — Load row + resolve model pool                   │
│   query_id, source, role, gold answer, prompt fields      │
│   experiment: primary | ablation                          │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE B — Paraphrase question text (if U enabled)         │
│   frozen T5 (or mock); k surfaces per query               │
│   cache: artifacts/paraphrases/{query_id}.json            │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE C — Probe each model M in {weak, strong}            │
│   MC: option-letter logprob pass → answer_scores + pred │
│   Hotpot: n samples → cluster entropy H → answer_scores   │
│   For each paraphrase surface: letter/sample → preds for U  │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE D — Assemble record + manifest                      │
│   signals/query_model/processed/{fit,calib,eval}.jsonl    │
│   artifacts/manifest.json (models, decoding, versions)    │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE E — Signal validation (no routing policy)           │
│   S1–S8 on fit; spot-check calib; eval held for paper     │
│   Metric: uncertainty–correctness AUROC (Chuang RQ1)      │
│   Join C(q) for S4                                        │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE F — Router (DEFERRED → §14)                         │
│   τ on calib; optional λ; routing curves; PGR/CPT on eval │
│   optional: calibration hold-out (Chuang §4 idea)         │
└───────────────────────────────────────────────────────────┘
```

**Out of scope in Stages A–E:** routing table, verifier, trained hidden-state probes, Arena labels.

---

## 6. Backend contract

### 6.1 `ModelBackend` interface

| Method | MC | Hotpot |
|--------|-----|--------|
| `score_letters(prompt) → {logprobs, letters}` | ✓ | — |
| `sample_answers(prompt, n, decoding) → [text]` | — | ✓ |

Implementations:

- **`mock`** — deterministic fake logprobs/samples (`--mock`, CI, schema smoke).
- **`hf`** — HuggingFace + chat template; 4-bit load for 70B; fixed seed/decoding in config.

### 6.2 Decoding defaults

| Setting | MC probe | Hotpot samples |
|---------|----------|----------------|
| temperature | 0 (greedy) | e.g. 0.7 |
| max_new_tokens | 1 (letter) or small | e.g. 32–64 |
| seed | fixed | fixed |

Match Chuang’s discipline where relevant: greedy scoring for MC option-letter probes; stochastic sampling only when estimating cluster entropy (Hotpot).

### 6.3 Paraphraser

- Frozen `google/flan-t5-base` (or config override).
- Prompt: rewrite question preserving meaning; **never** fine-tune on correctness labels.

---

## 7. Output record

One JSON line per `(query_id, model_id, experiment)`:

```json
{
  "query_id": "arc_train_0042",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "model_role": "weak",
  "experiment": "same_family_scaleup",
  "role": "fit",
  "source": "arc_challenge",
  "answer_scores": {
    "H": 0.42,
    "p_max": 0.81,
    "pred": "B",
    "margin": 0.55,
    "top2_mass": 0.92,
    "perplexity_H": 1.52,
    "inv_p_max": 1.23,
    "surprisal": 0.21,
    "probs": {"A": 0.05, "B": 0.81, "C": 0.10, "D": 0.04},
    "n_options": 4
  },
  "paraphrase": {
    "U": 0.33,
    "agreement": 0.67,
    "k": 3,
    "preds": ["B", "B", "C"]
  },
  "probe": {"kind": "mc_letter"}
}
```

`probe.kind`: `mc_letter` | `freeform_clusters`  
(`freeform_clusters` = discrete cluster entropy probe; semantic-entropy family with exact-match v1.)  
Hotpot `answer_scores` also stores raw **`samples`** (list of generated strings) so \(H\) can be recomputed offline if `normalize_answer` changes, without re-querying the model.

**Artifacts:** `processed/{fit,calib,eval}.jsonl`, `artifacts/manifest.json` — **gitignored**.

---

## 8. Leakage firewall

| Action | fit | calib | eval |
|--------|-----|-------|------|
| Compute \(H, U\) | ✓ | ✓ | ✓ (for reporting only) |
| Choose signal family / direction | ✓ analysis | confirm | — |
| Tune thresholds \(\tau\), weights \(\lambda\) | ✗ | ✓ | apply for metrics |
| Report PGR, CPT, routing curves | ✗ | diagnose | ✓ |
| Train paraphraser or probe on labels | ✗ never | ✗ | ✗ |

Gold labels: **evaluate** whether signals track `pred == gold`, `needs_strong`, weak-wrong — never to train a router.

---

## 9. Experiments (Stage E — before routing)

Run on **fit** first (per `source`, primary pool). Use **calib** only to confirm thresholds later.

| ID | Question | Primary metric |
|----|----------|----------------|
| **S0** | Schema + mock smoke (`--limit 8`) | pass/fail |
| **S1** | MC `pred` vs gold | accuracy |
| **S2** | Which family predicts **weak wrong**? | **AUROC** (Chuang alignment) |
| **S3** | Define `needs_strong` = weak wrong ∧ strong right; AUROC | AUROC |
| **S4** | Does \((H,U)\) add beyond \(C(q)\)? | ΔAUROC after join |
| **S5** | Weak−strong gap e.g. \(H_w - H_s\), \(p_{\max,w} - p_{\max,s}\) | AUROC / correlation |
| **S6** | \(U\) vs \(H\) — redundant or complementary? | correlation + AUROC |
| **S7** | Hotpot cluster entropy \(H\) vs `meta.level` / match | rank correlation |
| **S8** | Ablation: Qwen weak vs Llama weak (one table row) | same as S2 |

**Pass criterion before router (§14):** at least one model-dependent family beats chance on S2/S3 on pooled MC fit, or clear source-specific story documented.

**Deferred to router:** routing curves, calibration hold-out across benchmarks (Chuang §4), verifier (Sawant), gated multi-sample on all traffic.

---

## 10. Analysis joins

```text
signals/query_model/processed/fit.jsonl
        │
        ├─ join on query_id ─→ signals/query/processed/fit.jsonl  (C_query, φ)
        │
        └─ join on query_id + gold ─→ corpus queries  (correctness, needs_strong)
```

Example derived labels (eval only for final tables):

- `weak_correct` = (`pred_w == gold`)
- `needs_strong` = `¬weak_correct ∧ strong_correct` (when strong scores available)

---

## 11. Implementation layout

**Full end-to-end guide (repo walkthrough, packages, commands):** §15.

```text
signals/query_model/
  WORKFLOW.md          # this file
  config.yaml          # models, paths, decoding, k, n_samples  ✓
  features.py          # scores_from_probs, cluster_entropy, paraphrase_U  ✓
  backend.py           # ModelBackend: mock | hf  ✓ mock only
  paraphrase.py        # frozen rewrites  ✓ mock only
  build.py             # CLI  ✓
  requirements.txt     # ✓
  processed/           # gitignored
  artifacts/           # gitignored
```

### CLI (target)

```bash
# Smoke
./run.sh query-model --mock --limit 8

# Primary pool, fit split
./run.sh query-model --experiment primary --roles fit

# Ablation weak model
./run.sh query-model --experiment ablation --roles fit --limit 100
```

### Build order

1. `config.yaml` — pool pointers to `datasets/config.yaml`, decoding, paths.
2. `features.py` — pure math + unit tests on toy distributions.
3. `backend.py` — `mock` first.
4. `paraphrase.py`.
5. `build.py` + `./run.sh query-model` entry in `run.sh`.
6. **S0** → **S1–S6** on MC fit → Hotpot **S7** → ablation **S8**.
7. `research/method_model_dependent.md` stub aligned with this file.
8. **Router** workflow doc (only after S2/S3 pass; §14).

### Dependencies (`requirements.txt`)

`pyyaml`, `torch`, `transformers`, `accelerate`, `bitsandbytes` (70B), `numpy`; add `sentence-transformers` only if Hotpot clustering upgrades beyond exact match.

---

## 12. Paper paragraph (signal stage)

> We define model-dependent routing signals as properties of the query–model interaction. **Answer uncertainty** \(H(Y \mid q, M)\), where \(Y\) is the model’s answer, is the Shannon entropy of the **renormalized option-letter distribution** from a single probe on multiple-choice tasks (cf. choice entropy / token-probability UQ; Chuang et al., 2025), and **discrete cluster entropy** over \(n\) sampled answers on free-form tasks (a sample-based estimator in the semantic-entropy family; Kuhn et al., 2023; Zhang et al., 2025; exact-match clustering in v1). From the same MC probe we derive confidence summaries (`p_{\max}`, margin). **Paraphrase uncertainty** \(U(q \mid M)\) measures prediction stability under rewording of the question text. Unlike preference-trained routers (RouteLLM) or supervised hidden-state probes (Chuang et al.), we extract these signals without training a query-to-model classifier; we first test uncertainty–correctness alignment on held signal-analysis splits, then fix routing policy in a separate stage.

---

## 13. Related reading

| Source | Relevance |
|--------|-----------|
| [`research/confidence_routing_alignment.md`](../../research/confidence_routing_alignment.md) | Sawant, Chuang, Accredian positioning |
| [`research/related_work.md`](../../research/related_work.md) | RouteLLM, Hybrid, IRT, Zhang SE |
| [`signals/query/WORKFLOW.md`](../query/WORKFLOW.md) | \(C(q)\) stage |
| **§1 terminology** | Preferred terms: answer uncertainty, option-letter distribution, discrete cluster entropy |
| **§2.7 above** | Stage 1 audit: Chuang 8 + Sawant + RouteLLM → adapted / deferred / contrast |
| Chuang et al., [arXiv:2502.04428](https://arxiv.org/abs/2502.04428) | AUROC alignment; calibration data (router §14) |

---

## 14. Router stage preview (deferred — do not implement yet)

When signal validation passes:

1. **Policy sketch:** if \(C(q) \ge \tau_C\) → strong; elif \(H_w \ge \tau_H\) or \(U_w \ge \tau_U\) → strong; else serve weak `pred` from same probe.
2. **Calibration:** tune \(\tau_\*\) on **calib** only; optional \(\lambda\) weights on calib.
3. **Metrics on eval:** PGR, CPT, strong-call rate; optional routing curve vs offload rate (Chuang §3).
4. **Optional:** pooled calibration hold-out across ARC+Hotpot+MMLU calib (Chuang §4 analogy).

Do not implement §14 until S2/S3 pass on fit.

When ready, principles from alignment doc map to router as:

| Alignment idea | Router mechanism |
|----------------|------------------|
| Chuang offload | weak probe → if low confidence → strong |
| Sawant zones | high / mid / low bands on \(H_w\), \(U_w\) |
| Chuang calibration data | pooled calib CDF for \(\tau\) on new source |
| Sawant verifier | optional on lowest band |
| Accredian baselines | always-weak, always-strong, random, \(C\)-only |
| Our combination | \(C(q)\) OR \((H_w, U_w)\) thresholds; optional \(\lambda\) on calib |

---

## 15. End-to-end implementation guide

This section is the **operational runbook**: what exists in the repo today, what to build next, which libraries to install, and the exact command sequence from raw HuggingFace data → paper-ready signal tables.

### 15.1 Repository map

```text
routing/
├── run.sh                          # entry: corpus | complexity | query-model (planned)
├── README.md
├── professor.md                    # problem framing (unsupervised signals, binary pool)
├── datasets/
│   ├── config.yaml                 # sources, roles, model pool (weak/strong IDs)
│   ├── build.py                    # corpus builders + leakage checks  ✓
│   ├── requirements.txt            # datasets, pyyaml
│   └── processed/corpus_v1/        # gitignored outputs
│       ├── queries_{fit,calib,eval}.jsonl
│       ├── split_ids.json
│       └── manifest.json
├── signals/
│   ├── query/                      # Stage 0: C(q)  ✓ implemented
│   │   ├── WORKFLOW.md
│   │   ├── config.yaml
│   │   ├── features.py             # row-local φ(q) blocks
│   │   ├── build.py                # stages A–E pipeline
│   │   ├── requirements.txt
│   │   ├── processed/{fit,calib,eval}.jsonl
│   │   └── artifacts/              # geometry, zscore, manifest
│   └── query_model/                # Stage 1: H, U  ✗ not implemented yet
│       └── WORKFLOW.md             # this file
└── research/
    ├── confidence_routing_alignment.md
    ├── related_work.md
    ├── method_model_independent.md
    └── introduction.tex
```

**Status legend:** ✓ = code + outputs exist in tree · ✗ = spec only (implement per §11).

### 15.2 Full pipeline (three stages)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 0 — Corpus (datasets/)                                            │
│   HF datasets → unified query rows with gold, role, task_type           │
│   Command: ./run.sh corpus [--smoke]                                    │
│   Output:  datasets/processed/corpus_v1/queries_*.jsonl                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ query_id, prompt, gold, source, role
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 0 — Query complexity C(q) (signals/query/)  ✓ DONE              │
│   No LLM calls. Embeddings + geometry on fit only.                    │
│   Command: ./run.sh complexity [--limit N]                              │
│   Output:  signals/query/processed/{fit,calib,eval}.jsonl               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ join on query_id (analysis S4)
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — Answer uncertainty H, paraphrase U (signals/query_model/)     │
│   Probe weak + strong; MC option-letter H; Hotpot cluster entropy H   │
│   Command: ./run.sh query-model --mock|--experiment primary ...         │
│   Output:  signals/query_model/processed/{fit,calib,eval}.jsonl       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ S1–S8 validation on fit
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — Router (deferred §14)                                       │
│   τ on calib; PGR/CPT on eval                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 15.3 Command sequence (from clean checkout)

**One-time environment** (repo root):

```bash
cd /path/to/routing
python3 -m venv .venv
./run.sh --smoke          # creates venv, installs datasets deps, tiny corpus
```

**Stage 0a — full corpus:**

```bash
./run.sh corpus
# → datasets/processed/corpus_v1/queries_{fit,calib,eval}.jsonl
# Current counts (manifest): fit≈2278, calib≈499, eval≈2597
```

**Stage 0b — query complexity (no GPUs required):**

```bash
./run.sh complexity
# smoke: ./run.sh complexity --limit 64
# → signals/query/processed/*.jsonl + artifacts/
```

**Stage 1 — model-dependent signals (planned):**

```bash
# 1. Schema / CI smoke (no GPU, no HF weights)
./run.sh query-model --mock --limit 8

# 2. MC fit slice — primary pool, both models
./run.sh query-model --experiment primary --roles fit --sources arc_challenge,mmlu

# 3. Hotpot samples (more inference)
./run.sh query-model --experiment primary --roles fit --sources hotpotqa

# 4. Full calib/eval after fit analysis passes
./run.sh query-model --experiment primary --roles calib,eval
```

**Analysis (notebook or script — not yet in repo):**

```bash
# Join streams on query_id; compute AUROC for S2–S8
# Inputs:
#   signals/query_model/processed/fit.jsonl
#   signals/query/processed/fit.jsonl
#   datasets/processed/corpus_v1/queries_fit.jsonl  (gold)
```

### 15.4 Existing codebase walkthrough

#### `run.sh` — unified CLI

| Subcommand | Python entry | Installs |
|------------|--------------|----------|
| `--smoke` / `corpus` | `datasets/build.py` | `datasets/requirements.txt` |
| `complexity` | `signals/query/build.py` | `signals/query/requirements.txt` |
| `query-model` *(planned)* | `signals/query_model/build.py` | `signals/query_model/requirements.txt` |

Pattern to mirror for `query-model`: create venv if missing, `pip install -r signals/query_model/requirements.txt`, `exec` python with forwarded args.

#### `datasets/build.py` — corpus builder ✓

**Role:** Download ARC-Challenge, HotpotQA (distractor), MMLU subjects; emit leakage-safe `queries_*.jsonl`.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `row()` | Canonical schema: `query_id`, `source`, `role`, `prompt`, `gold`, `task_type`, `meta` |
| `build_arc` / `build_hotpot` / `build_mmlu` | Per-source HF loaders |
| `check()` | Asserts: no duplicate `query_id`, no fit/eval/calib overlap, no empty gold |
| `mc_prompt()` | Standard MC template ending in “Answer with the option letter only.” |

**Corpus row contract** (input to all signal stages):

```json
{
  "query_id": "ca76187619636336cce1e11f",
  "source": "arc_challenge",
  "role": "fit",
  "task_type": "science_mc",
  "prompt": "Question text...\n\nA. ...\nB. ...\n\nAnswer with the option letter only.",
  "gold": "C",
  "metric": "accuracy",
  "meta": { "n_choices": 4 }
}
```

**Config:** [`datasets/config.yaml`](../../datasets/config.yaml) — `model_pool.primary/ablation`, per-source `roles` and caps.

#### `signals/query/features.py` + `build.py` — C(q) ✓

**Role:** Model-independent complexity without calling candidate LLMs.

**`features.py`** — pure row-local math:

| Export | Blocks |
|--------|--------|
| `extract_row(row, cfg)` | `structural`, `linguistic_cues`, `task_form` |
| `parse_prompt()` | Split MC stem vs options (reused conceptually by paraphrase stage) |

**`build.py`** — five stages (see [`signals/query/WORKFLOW.md`](../query/WORKFLOW.md) §4):

| Stage | What | Fit-only? |
|-------|------|-----------|
| A | `extract_row` per corpus row | — |
| B | `SentenceTransformer` embed all prompts | — |
| C | `GeometryModel.fit` PCA + centroid + kNN + LOF | **fit only** |
| D | `ZScore.fit` + `compose_complexity` → `C_*` | **fit only** |
| E | Write `processed/*.jsonl` + artifacts | — |

**Output row** adds `complexity.C_query`, `embedding_geometry.*`, etc. Join key = `query_id`.

**Reuse for Stage 1:** `parse_prompt()` logic should be mirrored in `paraphrase.py` so rewrites touch **question text only**, not option lines.

#### `signals/query_model/` — H, U ✗ (spec only)

Mirror the `signals/query/` layout. Planned module responsibilities:

| File | Responsibility | Analog in `signals/query/` |
|------|----------------|----------------------------|
| `config.yaml` | Model IDs from `datasets/config.yaml`, decoding, `k` paraphrases, `n_samples` Hotpot | `signals/query/config.yaml` |
| `features.py` | `scores_from_probs`, `cluster_entropy`, `paraphrase_U` — pure numpy | `features.py` |
| `backend.py` | `ModelBackend` protocol: `score_letters`, `sample_answers`; `mock` + `hf` | *(new — needs torch)* |
| `paraphrase.py` | Frozen `google/flan-t5-base` rewrites; cache under `artifacts/paraphrases/` | *(new)* |
| `build.py` | Stages A–E from §5; CLI `--mock`, `--experiment`, `--roles`, `--limit` | `build.py` |

**`build.py` pipeline (to implement):**

```text
STAGE A  Load corpus rows + resolve model pool from config
STAGE B  Paraphrase question text (cache per query_id)
STAGE C  For each M ∈ {weak, strong}:
           MC     → backend.score_letters → answer_scores + pred
           Hotpot → backend.sample_answers → cluster_entropy → answer_scores
           For each paraphrase surface → preds → U
STAGE D  Write signals/query_model/processed/{role}.jsonl + manifest
STAGE E  (external) S1–S8 analysis scripts / notebook
```

### 15.5 Prerequisites

#### Knowledge (read before coding)

| Topic | Why | Where in repo |
|-------|-----|---------------|
| Unsupervised routing framing | What we claim vs RouteLLM | `professor.md`, `research/related_work.md` |
| Chuang UQ + AUROC alignment | Primary eval metric S2/S3 | `research/confidence_routing_alignment.md` §B |
| Letter-probe vs token-stream entropy | MC uses option-letter / answer-space scores | §3.1, §2.7 |
| Leakage splits | fit / calib / eval firewall | `datasets/build.py` `check()`, §8 |
| Model pool | weak 8B vs strong 70B | `datasets/config.yaml` `model_pool` |

#### Hardware & runtime

| Workload | GPU | Notes |
|----------|-----|-------|
| Corpus build | CPU | Downloads HF datasets; ~minutes |
| `signals/query` complexity | CPU (optional GPU for ST) | MiniLM embedder; ~2278 fit rows in minutes |
| `query-model --mock` | CPU | Deterministic fake logprobs; CI |
| `query-model` weak 8B | 1× GPU ≥16GB | 4-bit optional via bitsandbytes |
| `query-model` strong 70B | 1× GPU ≥40GB or multi-GPU | 4-bit load in `backend.py` plan |
| Paraphrase T5-base | CPU or small GPU | Frozen; cache rewrites |

#### Accounts & env vars

```bash
# HuggingFace model download (Llama gates)
export HF_TOKEN=...          # or: huggingface-cli login

# Optional: cache dir
export HF_HOME=/path/to/hf_cache
```

Llama 3.1 weights require Meta license acceptance on HuggingFace before first pull.

### 15.6 Python packages by stage

| Stage | File | Packages | Purpose |
|-------|------|----------|---------|
| Corpus | `datasets/requirements.txt` | `datasets`, `pyyaml` | HF `load_dataset` |
| Complexity | `signals/query/requirements.txt` | `sentence-transformers`, `scikit-learn`, `numpy`, `pyyaml` | Embeddings, PCA/LOF |
| Query×model *(planned)* | `signals/query_model/requirements.txt` | `pyyaml`, `numpy`, `torch`, `transformers`, `accelerate`, `bitsandbytes` | HF inference |
| Analysis *(planned)* | top-level or `analysis/requirements.txt` | `pandas`, `scikit-learn` | AUROC, joins, tables |
| Hotpot v2 clustering *(optional)* | add to query_model | `sentence-transformers` | Embedding clusters beyond exact match |

**Install pattern** (already used by `run.sh`):

```bash
.venv/bin/pip install -r datasets/requirements.txt
.venv/bin/pip install -r signals/query/requirements.txt
.venv/bin/pip install -r signals/query_model/requirements.txt   # when added
```

**Version guidance:** Pin `torch` to your CUDA build; `transformers>=4.40` for Llama 3.1 chat templates; `bitsandbytes` only needed for 4-bit 70B.

### 15.7 Data contracts & joins

All stages share **`query_id`** from corpus. Model-dependent rows add **`model_id`** + **`model_role`** + **`experiment`**.

```text
datasets/processed/corpus_v1/queries_fit.jsonl
  query_id, prompt, gold, source, role

signals/query/processed/fit.jsonl
  query_id, complexity.C_query, structural, embedding_geometry, ...

signals/query_model/processed/fit.jsonl   (one line per query × model)
  query_id, model_id, model_role, experiment,
  answer_scores.{H, p_max, pred, margin, ..., samples?},  # samples on Hotpot freeform only
  paraphrase.{U, agreement, k, preds},
  probe.kind
```

Offline recluster (when `samples` present): `cluster_entropy(row["answer_scores"]["samples"])`.

**Derived labels for S2/S3** (compute in analysis, not stored in signal build):

```python
weak_correct   = (pred_weak == gold)
strong_correct = (pred_strong == gold)
weak_wrong     = not weak_correct
needs_strong   = weak_wrong and strong_correct
```

**AUROC direction:** higher uncertainty (`H`, `U`, `surprisal`) should predict `weak_wrong` or `needs_strong` — verify sign per family on fit before reporting.

### 15.8 `.gitignore` additions (when implementing)

Add to repo `.gitignore`:

```text
signals/query_model/processed/
signals/query_model/artifacts/
```

(processed complexity paths are already ignored.)

### 15.9 Implementation checklist

| Step | Task | Status |
|------|------|--------|
| 1 | Corpus `queries_*.jsonl` | ✓ built |
| 2 | `signals/query` C(q) on all roles | ✓ built |
| 3 | `signals/query_model/config.yaml` | ✓ |
| 4 | `features.py` — unit-test toy softmax → H, margin | ✓ |
| 5 | `backend.py` — `MockBackend` (+ `HFBackend` later) | ✓ mock |
| 6 | `paraphrase.py` — question-only rewrites + cache | ✓ mock |
| 7 | `build.py` + `run.sh query-model` | ✓ |
| 8 | S0 smoke `--mock --limit 8` | ✓ |
| 9 | S1–S6 MC fit (real 8B/70B) | ✗ |
| 10 | S7 Hotpot cluster entropy | ✗ |
| 11 | S8 ablation Qwen weak | ✗ |
| 12 | Analysis notebook: joins + AUROC tables | ✗ |
| 13 | Router doc (only if S2/S3 pass) | deferred §14 |

`run.sh query-model` is wired (see §11 CLI).

### 15.11 What not to build yet

Per professor order and §2.5 firewall — **do not implement** until signal validation passes:

- Trained router (RouteLLM-style)
- Hidden-state probes (Chuang #7–8)
- Verifier model (Sawant)
- τ / λ tuning or PGR/CPT on eval
- Agent orchestration signals

Focus implementation on **HF backend (8B MC fit) → paraphrase U → fit AUROC tables**.
