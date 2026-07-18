# Problem Formulation

We formalize **unsupervised LLM routing**: selecting a model for a query from estimated routing signals, in contrast to supervised routers that learn a mapping from queries to models using preference, quality-gap, or correctness labels.

## Setup and Notation

Let \(\mathcal{Q}\) denote a space of input queries. Each query \(q\in\mathcal{Q}\) is a prompt string (including multiple-choice options when present). Let \(\mathcal{M}=\{M_1,\ldots,M_K\}\) denote a fixed pool of candidate LLMs, ordered by increasing capability and cost. We often write \(M_{\mathrm{weak}}\) and \(M_{\mathrm{strong}}\) for a binary pair; the formulation generalizes to \(K\ge 2\).

Each model \(M\in\mathcal{M}\) incurs a nonnegative **cost** \(c(M)\) per query. Let \(y_M(q)\) denote the response when \(M\) answers \(q\), and \(\mathrm{qual}(q,y)\) a task-appropriate quality score when ground truth is available.

A **router** is a function \(R:\mathcal{Q}\to\mathcal{M}\). At deployment, only \(R(q)\) produces the final answer (single-model routing).

Queries are partitioned into **fit**, **calib**, and **eval**. Transforms and thresholds are fit on fit/calib only; eval is held out for reporting.


## Unsupervised LLM Routing

We study **unsupervised LLM routing**: given \(q\), estimate routing signals and decide which model to call without large-scale preference or outcome supervision.

### Model-independent signals

\(s(q)\) depends only on the query. Feature vector \(\phi(q)\) and scalar complexity \(C(q)=g(\phi(q))\) (Section: model-independent method). \(C(q)\) is computed **before** calling any candidate LLM.

### Model-dependent signals

\(s(q,M)\) depends on query–model interaction. Feature vector \(\psi(q,M)\):

- **Answer uncertainty** \(H(q\mid M)\equiv H(Y\mid q,M)\): entropy of the model’s answer on \(q\). MC: option-letter distribution from one probe. Free-form: discrete cluster entropy over sampled answers.
- **Paraphrase-based uncertainty** \(U(q\mid M)\): prediction disagreement when the question is paraphrased; \(M\) probed on each surface form.

Additional probe-derived scores (e.g. \(p_{\max}\), margin) are coordinates of \(\psi\). Computing \(\psi\) requires probing \(M\); we do not claim strict query-only routing, but we do not require preference-scale supervision to define signals.

### Joint signal vector

\[
\mathbf{z}(q,M)=\bigl[\,\phi(q),\;\psi(q,M)\,\bigr].
\]

**Operational complexity:** A query is more complex if a weaker model is more likely to fail, so the system should escalate to a stronger model. Signals proxy this need; predictive power is evaluated on fit and reported on eval.

## Routing Decision Rules

### Rule-based routing

Query complexity only:
\[
R_{\mathrm{C}}(q)=
\begin{cases}
M_{\mathrm{strong}} & \text{if } C(q)\ge \tau_C,\\
M_{\mathrm{weak}} & \text{otherwise.}
\end{cases}
\]

Model-dependent escalation:
\[
R_{\mathrm{H}}(q)=
\begin{cases}
M_{\mathrm{strong}} & \text{if } H(q\mid M_{\mathrm{weak}})\ge \tau_H \;\lor\; U(q\mid M_{\mathrm{weak}})\ge \tau_U,\\
M_{\mathrm{weak}} & \text{otherwise.}
\end{cases}
\]

Thresholds on **calib**; frozen for **eval**.

### Parameter-weighted routing

\[
S(q; \boldsymbol{\lambda})=\sum_{j=1}^{d} \lambda_j\, z_j(q, M_{\mathrm{weak}}),
\qquad
R_{\boldsymbol{\lambda}}(q)=
\begin{cases}
M_{\mathrm{strong}} & \text{if } S\ge \tau,\\
M_{\mathrm{weak}} & \text{otherwise.}
\end{cases}
\]

\(\boldsymbol{\lambda}\) and \(\tau\) may be tuned on calib; **signals are defined a priori**, not trained from Arena/preference labels. This is the split: unsupervised signal extraction → optional weight learning.

## Evaluation Objectives

**PGR** (performance gap recovered):
\[
\mathrm{PGR}(R)=\frac{r(R)-r(M_{\mathrm{weak}})}{r(M_{\mathrm{strong}})-r(M_{\mathrm{weak}})}.
\]

**CPT**(\(x\%\)): minimum fraction of strong-model calls to reach target PGR \(x\%\).

**Signal validation** (on fit, before policy): ROC-AUC of each signal vs weak-model failure / need for strong model. Weak or negative correlations are reported when present.

