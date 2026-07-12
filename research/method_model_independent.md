# Method: Model-Independent Routing Signals

We first describe the **model-independent** component of our routing signal inventory: a query complexity representation computed from the query text alone, without calling any candidate LLM. Model-dependent signals (model-conditional entropy and paraphrase-based uncertainty) are defined later and composed with the present signals at decision time.

## Problem and Operational Definition

Let \(q\) denote an input query (the prompt string, including multiple-choice options when present). Let \(\mathcal{M}=\{M_1,\ldots,M_K\}\) be a pool of candidate LLMs. A **model-independent signal** is any feature of \(q\) that does not depend on a particular \(M\in\mathcal{M}\). Our goal at this stage is to estimate a feature vector \(\phi(q)\) and an optional scalar complexity score \(C(q)=g(\phi(q))\) that can be used in a **pre-inference** routing rule—for example, escalating to a stronger model when \(C(q)\) is large—before generating a full answer from a candidate LLM.

**Operational definition for routing:** A query is treated as more complex if a weaker or cheaper model is more likely to fail on it, so the system should escalate to a stronger or costlier model.

We do *not* equate complexity with length alone. Instead we approximate complexity with four complementary dimensions of \(\phi(q)\), then compose normalized scalars into \(C(q)\). Because a frozen sentence embedding model is used for one dimension, we describe these signals as **candidate-independent** (independent of the routing pool \(\mathcal{M}\)), not as tokenizer-free or embedding-free.

## Data Splits and Leakage Control

Queries are partitioned into three roles: **fit**, **calib**, and **eval**.

| Transform | Fit on | Apply to |
|-----------|--------|----------|
| Geometry (PCA / kNN / LOF) | fit | fit, calib, eval |
| Z-score | fit | all |
| Complexity weights / threshold \(\tau\) | calib | eval for reporting |

No eval query identifier participates in geometry fitting.

## Feature Vector \(\phi(q)\)

\[
\phi(q)=\bigl(\phi_{\mathrm{str}}(q),\;\phi_{\mathrm{ling}}(q),\;\phi_{\mathrm{form}}(q),\;\phi_{\mathrm{geo}}(q)\bigr)
\]

### Surface load (\(\phi_{\mathrm{str}}\))

After splitting a multiple-choice prompt into a question stem and option texts when option markers are present:

- token lengths of the full prompt and of the question stem
- mean / std of option lengths; question-to-option length ratio (when options exist)
- moving-average type–token ratio (MATTR) over a fixed window
- zlib compression ratio \(|\mathrm{compress}(q)|/|q|\) (lower ⇒ denser)

### Linguistic and cognitive cues (\(\phi_{\mathrm{ling}}\))

Using fixed lexicons and surface markers:

- **requirement** markers (multi-step connectors, enumerated steps)
- **reasoning depth**: max revised Bloom level from process-verb cues on the stem (Understand=2 … Create=6; Remember/interrogatives omitted; bare *how* excluded)
- **multi-hop** cues (both, as well as, according to, …)
- **domain breadth**: # distinct domain buckets with ≥1 keyword hit
- # question marks; # sentences in the stem

Lexicons are frozen in configuration; not learned from preference or correctness labels.

### Task form (\(\phi_{\mathrm{form}}\))

- `is_mc`, `n_choices` (0 if free-form)
- `source`, `task_type` (stratified analysis)

### Semantic atypicality (\(\phi_{\mathrm{geo}}\))

Embed each prompt with a frozen Sentence-Transformers encoder (L2-normalized). On **fit** embeddings only, estimate PCA (≤3), centroid, \(k\)-NN (cosine), and LOF (novelty mode; LOF standardized on fit). For any \(q\): centroid distance, mean \(k\)-NN similarity, LOF score, PCA coordinates.

## Normalization and Scalar Complexity \(C(q)\)

Numeric features are z-scored on **fit**. Intermediate scalars:

\[
\begin{aligned}
C_{\mathrm{length}}(q) &= z(\mathrm{prompt\_token\_len}), \\
C_{\mathrm{density}}(q) &= z(\mathrm{MATTR}) - z(\mathrm{compression\_ratio}), \\
C_{\mathrm{atypical}}(q) &= \tfrac{1}{2}\,z(\mathrm{centroid\_distance}) + \tfrac{1}{2}\,z(\mathrm{LOF}), \\
C_{\mathrm{linguistic}}(q) &= \mathrm{mean}\bigl\{
  z(n_{\mathrm{req}}),\,
  z(\mathrm{reasoning}),\,
  z(\mathrm{multihop}),\,
  z(\mathrm{domain\_breadth}),\,
  z(n_{\mathrm{sentences}})
\bigr\}.
\end{aligned}
\]

Primary scalar (equal weights by default; tunable on calib):

\[
C(q)\;\equiv\;C_{\mathrm{query}}(q)
=\sum_{s} w_s\,C_s(q)
\Big/\sum_{s} w_s.
\]

## Use in Routing

Stage-1 rule using only the model-independent score:

\[
\mathrm{route}(q)=
\begin{cases}
M_{\mathrm{strong}} & \text{if } C(q)\ge \tau,\\
M_{\mathrm{weak}} & \text{otherwise,}
\end{cases}
\]

with \(\tau\) chosen on **calib** and frozen before **eval**. When model-dependent signals are available, \(C(q)\) is one coordinate of the unsupervised feature vector under rules or parameter weighting. Computing \(\phi(q)\) and \(C(q)\) requires no generation from \(\mathcal{M}\).
