# Workflow: model-dependent signals (query × model)

## 1. Research framing

### Core problem
Choose which LLM should answer query \(q\) to trade **quality vs cost**, using signals estimated **without** a large supervised map \(q \mapsto\) best model.

### What this stage does
Estimate **model-dependent** signals: information that comes from the **interaction** between a query and a candidate model \(M\).

This file covers **only** those signals. Routing rules, thresholds, and cost–quality tables come **later**, after we understand which signals carry information.

Model-independent complexity \(C(q)\) is already defined in [`signals/query/WORKFLOW.md`](../query/WORKFLOW.md).

### Professor order (keep this order)
1. Find what information we can extract from the **query** and from **query × model**.
2. Check which of those measures actually relate to mistakes / difficulty.
3. **Then** use the useful signals to route.

### Plain language
| Prefer saying | Avoid |
|---------------|--------|
| **question text** / **question part** | “stem” |
| **answer choices** / **options** | vague “MC items” alone |
| **score the letter tokens A/B/C/D** | “teacher-forced” without explaining |
| **group same answers together** | jargon-only “SE” with no gloss |

---

## 2. What we measure

For each pair \((q, M)\):

### 2.1 Answer-distribution scores (from one MC letter probe)

Give model \(M\) the multiple-choice prompt. Score how likely the **next token** is each option letter `A` / `B` / `C` / `D` (and `E` if present). Softmax those scores into probabilities \(p_i\). From **the same** probabilities, store:

| Field | Formula (idea) | Meaning |
|-------|----------------|---------|
| `H` | \(-\sum_i p_i \log p_i\) | Overall uncertainty (spread) |
| `p_max` | \(\max_i p_i\) | Confidence in the top choice |
| `pred` | \(\arg\max_i p_i\) | Letter the model would pick |
| `margin` | \(p_{(1)} - p_{(2)}\) | Gap between best and second-best |
| `top2_mass` | \(p_{(1)} + p_{(2)}\) | Mass on the two leading options |
| `perplexity_H` | \(e^{H}\) | Entropy on another scale |
| `inv_p_max` | \(1 / p_{\max}\) | Confidence on another scale |
| `surprisal` | \(-\log p_{\max}\) | Surprisal of the top letter |
| `probs` | \(\{L \mapsto p_L\}\) | Full letter distribution |

**Unified meaning of entropy:**  
\(H(q \mid M) = H(Y \mid q, M)\) = how uncertain \(M\) is about the **answer** \(Y\) to \(q\).

On MC, \(Y\) is a letter — so letter probabilities are the right distribution.

### 2.2 Signal families (for analysis — not eight independent claims)

| Family | Members | Note |
|--------|---------|------|
| Entropy | `H`, `perplexity_H` | Same ranking of queries |
| Confidence | `p_max`, `inv_p_max`, `surprisal` | Same ranking of queries |
| Margin | `margin` | Different story (close race) |
| Top-2 | `top2_mass` | Different story |
| Paraphrase | `U` | Needs rewordings; not a logprob transform |

**Store all fields.** When comparing “which signal informs,” compare **families** (e.g. `H`, `p_max`, `margin`, `top2_mass`, `U`).

### 2.3 Paraphrase uncertainty \(U(q \mid M)\)

1. Take the **question text** (question part of the prompt). Keep the same answer choices.
2. Make \(k-1\) alternative wordings with a **frozen** paraphraser (not the candidate model under test). Total surfaces = original + rewrites (\(k=3\) by default).
3. Run the **same** answer probe on each wording; collect predictions \(\hat y_1,\ldots,\hat y_k\).
4. \(U = 1 -\) (fraction that match the most common prediction).  
   High \(U\) ⇒ answers flip when the question is reworded (model is brittle).  
   Low \(U\) ⇒ stable under rephrase.

Primary `answer_scores` always come from the **original** wording. Paraphrase fields are only for \(U\).

### 2.4 Free-form path (Hotpot)

No A/B/C/D letters. Estimate the same \(H(Y \mid q, M)\) by:

1. Draw \(n\) short answers from \(M\).
2. Normalize text; **group same answers together**.
3. Entropy over group sizes = **discrete semantic entropy** (formal name for “uncertainty over meanings,” approximate).

Store `H`, mode `pred`, group sizes; set letter-only fields null when not applicable. Upgrade to entailment-based semantic entropy later if needed.

---

## 3. Model pool

From [`datasets/config.yaml`](../../datasets/config.yaml):

| Experiment | Weak | Strong |
|------------|------|--------|
| **primary** (main) | Llama-3.1-8B-Instruct | Llama-3.1-70B-Instruct |
| **ablation** | Qwen2.5-7B-Instruct | Llama-3.1-70B-Instruct |

For **signal understanding**, score **both** weak and strong on fit (so we can compare). Routing later may only probe the weak model.

---

## 4. Pipeline stages

```text
corpus queries_{fit,calib,eval}.jsonl
        │
        ▼
┌───────────────────┐
│ A  Load row + pool │
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────┐
│ B  Build paraphrases of the   │
│    question text (optional)   │
└─────────┬─────────────────────┘
          │
          ▼
┌───────────────────────────────┐
│ C  For each model M:          │
│    score letters (MC) or      │
│    sample answers (Hotpot)    │
│    → answer_scores            │
│    → paraphrase U             │
└─────────┬─────────────────────┘
          │
          ▼
┌───────────────────────────────┐
│ D  Write jsonl + manifest     │
└───────────────────────────────┘
```

**Out of scope here:** choosing weak vs strong for the user, \(\tau\), \(\lambda\), PGR/CPT.

---

## 5. Output record

One line per `(query_id, model_id)`:

```json
{
  "query_id": "...",
  "model_id": "llama-3.1-8b-instruct",
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

`probe.kind`: `mc_letter` | `freeform_se`.

Artifacts: `processed/{fit,calib,eval}.jsonl`, `artifacts/manifest.json`.

---

## 6. Firewall

| Allowed | Not in this stage |
|---------|-------------------|
| Compute signals on fit / calib / eval | Tune routing thresholds \(\tau\) |
| Join later to \(C(q)\) and gold for **analysis** | Report paper routing metrics (PGR, CPT) as if done |
| Freeze paraphraser + decoding in config | Learn paraphraser from correctness labels |

Gold labels: use only to **evaluate** whether signals track mistakes — not to train a preference router.

---

## 7. Honest cost note

Getting \((q, M)\) scores **requires calling** \(M\) (at least scoring letters or short samples). That is required for “interaction.” It is **not** the same as RouteLLM (query text only, no model call).

On MC, one letter-scoring pass yields `pred` and all `answer_scores`. Comparing `pred` to gold gives correctness **without** a second long generation.

---

## 8. Experiments (signals only — before routing)

Run on **fit** first, per `source`, primary pool. Then spot-check calib.

| ID | Question |
|----|----------|
| **S0** | Smoke / schema (`--mock`) |
| **S1** | Does MC `pred` match gold? (cheap correctness) |
| **S2** | Which family best predicts **weak wrong**? (AUROC / Spearman) |
| **S3** | Same for strong; define **needs_strong** = weak wrong and strong right |
| **S4** | Do answer_scores add anything beyond \(C(q)\)? (join complexity jsonl) |
| **S5** | Does weak−strong **gap** (e.g. \(H_w - H_s\)) help? |
| **S6** | Is paraphrase \(U\) different from \(H\)? (correlation + AUROC) |
| **S7** | Hotpot: discrete SE vs `meta.level` / answer match (after MC looks sane) |
| **S8** | One cross-family ablation row (Qwen weak) |

**Skip until signals are understood:** full routing curves, \(\lambda\) fitting, always calling both models with heavy sampling for every query.

---

## 9. Implementation layout (target)

```text
signals/query_model/
  WORKFLOW.md       # this file
  config.yaml
  features.py       # scores_from_probs, discrete SE, paraphrase U
  backend.py        # mock | HuggingFace (score letters / sample)
  paraphrase.py     # frozen rewrites of question text
  build.py          # CLI
  requirements.txt
  processed/        # gitignored
  artifacts/        # gitignored
```

Suggested CLI (when coded):

```bash
./run.sh query-model --mock --limit 8
./run.sh query-model --experiment primary --roles fit
```

---

## 10. Paper statement (this stage)

> We define model-dependent routing signals as properties of the query–model interaction: answer-distribution scores \(H(Y \mid q, M)\) (closed-set letter probabilities on multiple-choice; grouped-answer entropy on free-form) and related confidence summaries from the same distribution, plus paraphrase uncertainty \(U(q \mid M)\) measuring stability under rewording of the question text. We extract these signals without training a supervised query-to-model router; analysis asks which signal families predict model error before any routing policy is fixed.

---

## 11. Build order

1. `config.yaml` aligned with this workflow.  
2. `features.py` (math only; unit-testable).  
3. `backend.py` mock, then HuggingFace.  
4. `paraphrase.py`.  
5. `build.py` + `run.sh query-model`.  
6. S0 smoke → S1–S6 on fit MC → Hotpot / ablation as needed.  
7. **Then** design routing (separate workflow / docs).







Model-dependent signals (query–model interaction)

What the professor requires

From professor.md + research/introduction.tex:





Model-independent (done): \phi(q), C(q) — query alone.



Model-dependent (this stage): for each pair (q, M):





Entropy H(q \mid M) — how uncertain M is on q.



Paraphrase uncertainty U(q \mid M) — whether M’s answer stays stable when the same ask is rephrased (UKG vs robust student analogy).

Routing uses these as unsupervised features (rules or later \lambda), not preference-trained q \to model maps. Agents stay out of scope.

Honest framing (vs RouteLLM): these need a lightweight probe of M (logprobs and/or short generations). That is still pre-commitment to a full cascade/retry loop, but not “query-text only.” Name this clearly in WORKFLOW / method notes.

flowchart LR
  q[Query q] --> C[C_query done]
  q --> Probe[Probe each M in pool]
  Probe --> H["H(q|M)"]
  Probe --> U["U(q|M)"]
  C --> Score["score(q,M)"]
  H --> Score
  U --> Score
  Score --> Route[Pick weak or strong]

Signal definitions (concrete, corpus-aware)

Corpus = ARC/MMLU (MC) + Hotpot (free-form). One contract, two compute paths:

1. Entropy H(q \mid M)







Task



Probe



Formula





MC (ARC/MMLU)



Teacher-forced logprobs of option letters A/B/C/D (and E if present) given the prompt



Softmax over option logprobs → Shannon entropy H = -\sum_i p_i \log p_i; also store p_{\max}, predicted letter





Free-form (Hotpot)



n short samples (temp > 0)



Semantic / answer entropy: normalize answers (lower/strip), cluster exact-match groups (v1; SE embedding clusters can come later), H = -\sum_c (

High H ⇒ M is unsure on q ⇒ prefer escalate / other model under rules.

2. Paraphrase uncertainty U(q \mid M)





Build k paraphrases of the question stem with a frozen paraphraser (not the candidate under test): default google/flan-t5-base rewrite prompt, plus the original (k=3 total including original in v1).



Run the same answer probe as above on each surface form.



MC: U = 1 - fraction of paraphrases whose argmax letter matches the mode letter (or mean pairwise disagreement).



Free-form: U = 1 - fraction of paraphrases whose normalized answer matches the mode answer.

Low U ⇒ robust under rephrase ⇒ favor that M for q.

3. What we do not put in v1





Full Zhang-style semantic-entropy embedding clusters (add later if exact-match is too coarse on Hotpot).



Preference / correctness labels as training targets (oracle accuracy is for eval only, same firewall as complexity).



Agent signals.

Model pool (already frozen)

From datasets/config.yaml:





Primary: llama-3.1-8b-instruct ↔ llama-3.1-70b-instruct



Ablation: qwen2.5-7b-instruct ↔ llama-3.1-70b-instruct

Compute (H, U) for both models on each query for the active experiment; rule example: prefer the model with lower U (or lower H) among weak/strong, or combine with C(q) later.

Inference default: HuggingFace transformers + 4-bit load for 70B (bitsandbytes), Instruct chat templates, fixed seed/decoding in config. Smoke path: --limit + optional --mock backend (deterministic fake logprobs) so CI/dev does not need GPUs.

Layout (mirror signals/query, minimal files)

signals/query_model/
  WORKFLOW.md      # concepts, firewall, formulas, experiments
  config.yaml      # models, n_samples, k_paraphrases, decoding, paths
  features.py      # entropy + disagreement math (no HF)
  backend.py       # ModelBackend: mock | hf (logprobs MC + generate)
  paraphrase.py    # frozen T5 paraphrases of stem
  build.py         # CLI: corpus → per (query_id, model_id) jsonl
  requirements.txt

Wire: ./run.sh query-model [--limit N] [--experiment primary|ablation] [--mock]

Output record (one line per query–model)

{
  "query_id": "...", "model_id": "...", "role": "fit|calib|eval",
  "source": "arc_challenge|...", "experiment": "same_family_scaleup",
  "entropy": {"H": 0.0, "p_max": 0.0, "pred": "A"},
  "paraphrase": {"U": 0.0, "n": 3, "agreement": 0.67},
  "probe": {"kind": "mc_logprob|sample_entropy"}
}

Artifacts: signals/query_model/processed/{fit,calib,eval}.jsonl, artifacts/manifest.json. Gitignore processed (same pattern as query).

Firewall (same as complexity)







Use



Allowed





Build H,U on fit/calib/eval



yes (features of q,M)





Tune rule thresholds / \lambda



calib only





Paper metrics (PGR, CPT, AUROC)



eval only





Fit paraphraser or entropy on eval labels



never

Paraphraser and decoding hyperparams are frozen in config; not learned from correctness.

Experiments this unlocks (paper tables later)





E_H: does high H(q \mid M_{\mathrm{weak}}) predict needs_strong (oracle)?



E_U: does high U(q \mid M_{\mathrm{weak}}) predict needs_strong?



E_rule: prefer \arg\min_M U(q|M) (or H) vs always-weak / always-strong / C(q)-only.



Primary vs ablation = one extra row.

Oracle answers (correct/incorrect per model) are a separate eval artifact when you run full generations for scoring—not required to define H,U.

Implementation order





Write WORKFLOW.md + config.yaml (formulas + pool pointers).



Implement features.py (pure math) + unit checks on toy distributions.



Implement backend.py (mock first) + paraphrase.py.



Implement build.py CLI; smoke with --mock --limit 8.



Document real run: primary experiment on GPU when weights available.



Short research/method_model_dependent.md (+ .tex stub) aligned with WORKFLOW.

Dependencies

Add to signals/query_model/requirements.txt: transformers, accelerate, bitsandbytes, torch (and reuse pyyaml / sentence stack only if we later upgrade Hotpot clustering).