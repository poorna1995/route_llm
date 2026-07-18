# LLM routing literature ↔ our research alignment

**Sources (verified 2026-07-15):**
1. Udayan Sawant, *Confidence-Based Routing in LLM Systems* (Medium, May 2026) — summarized from chat excerpt; **not re-fetched**.
2. Yu-Neng Chuang et al., **[arXiv:2502.04428](https://arxiv.org/abs/2502.04428)** — facts from **arXiv PDF/HTML** (see version warning below).
3. Accredian / Pankaj Tiwari, *LLM Routing: Optimizing Pathways in Language Processing* — [Medium](https://medium.com/accredian/llm-routing-optimizing-pathways-in-language-processing-c52c2adf7c4e); summarized from **user-pasted excerpt**; **not re-fetched** in this pass. Survey of **learned routers** (RouteLLM, RouterBench, Hybrid LLM, etc.).

**Version warning:** An **extended workshop PDF** (OpenReview / NeurIPS 2025 workshop copy) uses different scale (**12 SLMs, 4 LLMs, 15 datasets, 5000+ settings**), renames “calibration data” → **“proxy routing data”**, adds **DeepSeek-R1 / Qwen3-32B / GPT-4.1 mini**, theory (Theorems 1–3), and an RMS-vs-random table. **Do not mix those numbers with arXiv:2502.04428.** This file cites the **arXiv** version unless a line is explicitly marked `[extended workshop]`.

**Purpose:** Understand both pipelines step by step and map them to our ACL unsupervised signal-based routing work.

**Our repo anchors:** [`professor.md`](../professor.md), [`research/introduction.tex`](introduction.tex), [`signals/query/WORKFLOW.md`](../signals/query/WORKFLOW.md), [`signals/query_model/WORKFLOW.md`](../signals/query_model/WORKFLOW.md).

---

# Part A — Industry blog (Sawant 2026)

## 1. What the blog is solving (in one paragraph)

Production routers often output a **single score** (e.g. complexity 0.38) and **hard-switch** to a tier. Borderline scores are not truly “Tier 2” — they are **mild preferences with hidden uncertainty**. The blog’s fix: output **(decision, confidence)** and treat low confidence as a trigger for **extra checks or escalation**, not silent misrouting. That is **confidence-based routing** — a probabilistic layer on top of a complexity router.

---

## 2. Blog pipeline (five layers)

```text
Request
   │
   ▼
┌─────────────────────┐
│ Complexity / tier   │  ← Post 3.1: heuristic or classifier → tier guess
│ score (point est.)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Signal extraction   │  ← entropy, logprobs, self-consistency
│ from model output   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Calibration         │  ← confidence should match empirical accuracy
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Threshold policy    │  ← zones: dispatch / sample / verify / escalate
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Escalation router   │  ← Tier 1 → verifier → Tier 2 if needed
└─────────────────────┘
```

### 2.1 Signal extraction (blog)

| Signal | What it measures | When used | Cost |
|--------|------------------|-----------|------|
| **Token entropy** | Spread of next-token distribution over generation | Logprobs available; mean over first 20–50 tokens | One generation with logprobs |
| **Logprob thresholds** | Confidence on **critical tokens** (e.g. first content token) | Targeted; API returns top logprobs | Same call |
| **Self-consistency** | Agreement across **N samples** of same prompt | “Uncertain zone” only | **N×** generation |
| **Verifier model** | Accept/reject (request, answer) pair | Low confidence; no logprobs | Extra small-model call |

Blog insight: **do not treat one heuristic score as certainty**. Overconfidence → cheap tier fails silently; underconfidence → wasted spend on strong tiers.

### 2.2 Calibration (blog)

A confidence of 0.70 is useful only if ~70% of such decisions are actually correct. Miscalibration is common. Fix: labeled **routing outcomes** (accepted / rejected / escalated) → reliability diagram → temperature scaling or **isotonic regression**. Recalibrate when distribution drifts (ECE monitoring).

### 2.3 Threshold zones (blog)

| Zone | Typical behavior |
|------|------------------|
| High confidence | Dispatch to chosen tier |
| Uncertain | Dispatch + async quality sampling |
| Low confidence | **Synchronous verify** or escalate before serve |

### 2.4 Failure modes (blog)

1. **Overconfidence** — worst for routing (silent quality loss).  
2. **Underconfidence** — over-escalation, cost blow-up.  
3. **Verifier blind spots** — verifier cannot catch domain errors it does not know.  
4. **Self-consistency cost** — never on full traffic; gate behind uncertain band.

---

# Part B — Academic paper (Chuang et al., arXiv:2502.04428)

## B.0 What the paper actually claims (checklist)

| Item | Stated in arXiv:2502.04428 |
|------|----------------------------|
| Settings scale | **1500+** (abstract & conclusion) |
| SLMs | **8** open-source (1B–8B): Llama-3.2-1B/3B, Phi-3.5-mini, **danube3.1-4b-chat**, Mistral-7B, Qwen2.5-7B, Llama-3.1-8B, granite-3.1-8b |
| LLMs | **2**: Llama-3.1-70B-Instruct, **GPT-4o mini** (API) |
| Datasets | **14** (math×4, commonsense×8, CoQA, MMLU) — **no MATH-500** |
| UQ methods | **8** (Table 1) |
| Alignment metric | **ROC AUC** — confidence ranks binary correctness |
| Generalization object | **Calibration data** (hold-out set), **not** target-task labels |
| Generalization eval | **Leave-one-out**: calibration from **13** datasets, evaluate on held-out 14th |
| UQ used for calibration experiments | **OOD Probe** and **Perplexity** only (§4.2) |
| Free-form grading | **GPT-4o mini** judges answer equivalence (Zheng et al., 2023) |
| Code repo in arXiv PDF | **Not cited** in the arXiv abstract/PDF we read |

## B.1 Problem and principle

**Setting:** On-device **SLM** for latency/energy; remote **LLM** for hard queries.

**Principle (paper wording):** offload when the SLM gives a **low-confidence** response — *“If you lack confidence, seek stronger support.”*

**Intro framing:** uncertainty is taken from the **SLM itself**, “without the aid of extra routers” for the *uncertainty-based* approach. **Caveat:** two of their eight UQ methods (**Trained Probe**, **OOD Probe**) *are* trained MLP classifiers on hidden states — so “no extra router” is **not** literally true for all methods they benchmark.

**Two research questions (paper §1):**

| RQ | Question | What they do |
|----|----------|--------------|
| **RQ1** | Best UQ practice for SLM→LLM routing? | Benchmark **8 UQ × 8 SLM × 14 datasets**; ROC-AUC alignment; **overall accuracy vs routing ratio** curves |
| **RQ2** | How to initialize routing on a **new dataset** without new downstream data? | **Calibration data** pipeline; predict routing curves on held-out dataset |

**Contrast with learned routers (their §2.2):** RouteLLM, RouterBench, etc. train query routers from historical outputs; struggle on new downstream tasks. They position **SLM self-uncertainty** as more portable — with the probe caveat above.

---

## B.2 Runtime story (as stated + what is **not** spelled out)

**Explicit in paper:**
1. SLM answers the query.
2. A UQ method yields a **confidence** score (higher = more certain; they equate low uncertainty with high confidence).
3. **Low-confidence** queries are candidates to **offload** to the LLM (abstract, §1).
4. **Routing curves** plot **overall accuracy vs routing ratio** (Figure 2 caption).
5. For generalization, **thresholds for different routing ratios** are chosen using **calibration data** (§4.3, Insights ❷).

**Not fully specified in main text:** the paper does **not** give a step-by-step pseudocode for building the routing curve (e.g. exact definition of “routing ratio” as a formula). Treat the curve as an **empirical protocol** reported in figures, not something we should invent. Figure 3 *does* define a related analysis: progressively **exclude lowest-confidence** queries and plot SLM accuracy relative to LLM on the remaining top fraction.

```text
Query q
   │
   ▼
SLM → answer + confidence (UQ method)
   │
   ▼
If confidence low enough → offload to LLM
Else → keep SLM answer
```

**Routing ratio (use paper’s term only):** x-axis in Figure 2; paper uses it with **overall accuracy** but does not define the ratio algebraically in the sections we verified. Do not cite a specific quantile formula unless you check their released code.

**Decoding note (calibration construction only, §4.2):** temperature **0**, top-p **1.0**, **greedy** search, seed **50** — stated for building calibration data, not necessarily for every benchmark run.

---

## B.3 Step-by-step: eight UQ methods (what they actually compute)

All methods output a scalar **confidence** (higher = more certain). Paper uses “uncertainty” and “confidence” interchangeably (low uncertainty = high confidence).

| # | Method | Access | Training? | Computation (step by step) |
|---|--------|--------|-----------|----------------------------|
| 1 | **Average token prob** | White-box | No | MC: prob of chosen option token (e.g. “A”). Free-form: mean token prob over generated sequence. |
| 2 | **Perplexity** | White-box | No | On N output tokens: \(\text{PPL} = \exp(-\frac{1}{N}\sum \ln p(y_i))\). Confidence = **inverse perplexity**. |
| 3 | **p(True)** | White-box | No | SLM generates answer; follow-up asks “True/False — is your answer correct?”; confidence = normalized prob of **True** token. |
| 4 | **Verbalization-1s** | Black-box | No | Single prompt: answer + numeric confidence in one shot. |
| 5 | **Verbalization-2s** | Black-box | No | Round 1: answer. Round 2: separate confidence query. |
| 6 | **Jaccard degree** | Black-box | No | Sample **m=5** answers (temp=1.0); pairwise Jaccard similarity matrix **D**; confidence = \(\text{trace}(mI - D) / m^2\). |
| 7 | **Trained probe** | White-box | **Yes** | 4-layer MLP, LeakyReLU; hidden states from **8th-to-last** layer; trained on **in-domain training subsample** per dataset; 20 epochs, lr **5×10⁻⁴** |
| 8 | **OOD probe** | White-box | **Yes** | Same architecture; trained on **all other datasets** (e.g. evaluate AQuA → train on other **13**); 20 epochs, lr **1×10⁻⁴** |

**MC detail (verified §3.1):** Average token probability uses the probability of the **chosen option token** (e.g. “A”) — not necessarily the same as scoring all letters and taking `argmax` (our MC probe records full letter distribution).

**Relation to our MC probe:** our `p_max` / `H` / `margin` use the **full letter distribution** from one scoring pass — **closely related** to their token-probability family, not guaranteed identical to “chosen option token” if generation and argmax differ.

---

## B.4 Benchmarking phase (Section 3)

```text
FOR each SLM (8 models)
  FOR each dataset (14)
    FOR each UQ method (8)
      → SLM outputs + confidence
      → correctness labels (task-dependent; GPT-4o mini for free-form)
      → ROC-AUC( confidence vs correctness )
      → plot overall accuracy vs routing ratio (to Llama-3.1-70B or GPT-4o mini)
```

**Hardware (§3):** four **80GB NVIDIA A100** GPUs.

**Datasets (14, §3.1):** GSM8K, AQuA, MultiArith, SVAMP; CommonsenseQA, HellaSwag, OpenBookQA, PIQA, TruthfulQA, WinoGrande, BoolQ, Social IQa; CoQA; MMLU.

**Example figures (verified captions):** Figure 2 — routing to **Llama-3.1-70B on OpenBookQA** and **GPT-4o mini on Social IQa**; Figure 1 — ROC AUC on **CommonsenseQA**.

---

## B.5 Four benchmark findings (what matters for routing)

**Finding ❶ — Calibration ≠ alignment.**  
A score can be well-calibrated yet poorly **rank** wrong vs. right answers. Routing needs **uncertainty–correctness alignment** (ROC-AUC), not calibration alone. SLMs can be **confidently wrong**.

**Finding ❷ — Verbalization fails for routing.**  
Verbalization-1s / verbalization-2s consistently show **low ROC-AUC** on SLMs → bad routing curves → unnecessary LLM calls (cost) or missed escalations.

**Finding ❸ — Top UQ methods for routing.**  
**Trained probe**, **OOD probe**, and **Perplexity** rank top-3 consistently. **Before deploying**, benchmark UQ alignment on your SLM — method choice dominates routing quality.

**Finding ❹ — High-confidence SLM ≈ LLM (example from paper).**  
Figure 3: exclude lowest-confidence queries; plot **relative accuracy** = SLM accuracy / LLM accuracy on the remainder. **Example cited in paper:** on **PIQA**, **Danube-3.1-4B** is nearly equal to **GPT-4o mini** on the **top 20%** highest-confidence queries. (Do **not** cite Qwen3-0.6B on GSM8K — that is from the `[extended workshop]` PDF, not arXiv.)

---

## B.6 Calibration data pipeline (Section 4) — arXiv wording

**Terminology:** arXiv uses **calibration data** (abstract: “calibration data construction … pipeline and open-source a constructed hold-out set”). `[extended workshop]` renames this **proxy routing data**.

**Motivation (Insights ❶, §4.3):** for a fixed (SLM, UQ), confidence histograms look similar across **14** tasks (Figure 4) — shape driven by **SLM + UQ**, not downstream dataset identity.

**Construction (§4.1–4.2, verified):**
1. Pool diverse datasets \(\mathbb{D} = \{\mathcal{D}_i\}_{i=1}^{N}\).
2. Run selected UQ → uncertainty distributions \(\{\mathcal{F}_{\mathbb{D}}\}\) binned into **M = 30** bins.
3. **Weighted-sample** instances from each bin; take **10%** of instances **per bin** → final **calibration data** \(\boldsymbol{X}\).
4. Decoding for this construction: **temp 0**, **top-p 1.0**, **greedy**, seed **50**.

**Generalization eval (§4.2, verified):**
- **Leave-one-out / cross-validation:** pick target dataset; build calibration data from the other **13**; target has **no** instances in calibration set.
- Compare routing curves from **calibration data** vs **full target dataset** (oracle ground truth for the experiment).
- Reported UQ for this section: **OOD Probe** and **Perplexity** only.
- **3** runs averaged.

**What arXiv does *not* include (present only in `[extended workshop]` PDF):** Theorems 1–3, explicit “fully OOD domain” split (math→commonsense), RMS distance vs random routing table, **quodlibeta** GitHub link, **12/4/15** scale.

**Our mapping:** their calibration hold-out ≈ our future use of **calib split** across ARC/Hotpot/MMLU to set τ — analogous idea, **not yet implemented**.

---

## B.7 How their routing differs from learned routers

| Aspect | Chuang et al. (UQ routing) | RouteLLM / Hybrid / RouterBench |
|--------|---------------------------|----------------------------------|
| Router input | SLM’s own generation signals | Query text (+ sometimes prefs) |
| Training | Probes optional; routing threshold from proxy stats | Supervised router training |
| New task | Proxy CDF transfer | Needs new training data |
| Inference cost | SLM run + optional LLM offload | Query-only router (RouteLLM etc.) |
| Core metric | ROC AUC alignment; accuracy vs routing ratio | Win-rate / cost-quality |

---

## B.8 Limitations (grounded)

- **Sequential cost:** SLM run always; LLM on offload path (same structural cost class as blog cascade).
- **Probe methods need labels** on source data to train — strongest alignment in their benchmark, but **supervised**.
- **Verbalization** ranks poorly for SLM routing in their experiments despite other UQ literature on calibration.
- **Eval cost:** GPT-4o mini as judge on free-form tasks.
- **Routing-ratio protocol** under-specified in prose — rely on their figures/code for reproduction.
- **No** query-complexity signal, **no** paraphrase-input uncertainty, **no** multi-signal fusion in their eight methods.
- **Two paper versions** in the wild — cite **arXiv:2502.04428** with correct scale (8/2/14/1500+).

---

# Part D — Survey blog: learned routers (Accredian / Tiwari)

**What this article is:** A **high-level survey** of LLM routing as cost–quality optimization — not a new method. It explains the generic router loop, then focuses on **RouteLLM** and points to **RouterBench**, **Hybrid LLM**, and two other papers. It does **not** discuss on-device UQ cascades (Chuang) or production confidence zones (Sawant).

**Source status:** Content below is taken only from the **paste you provided**. RouteLLM result numbers are **blog-reported**; for citations in our paper, prefer **Ong et al. (RouteLLM, ICLR 2025)** and [`research/related_work.md`](related_work.md) / [`literature.md`](../literature.md).

## D.1 Generic routing loop (blog’s five steps)

```text
1. Query analysis     — content, intent, domain, complexity, user prefs
2. Model selection    — capabilities, past performance, load, cost
3. Query forwarding   — send to chosen model(s)
4. Response aggregation (optional) — merge if multiple models
5. Performance monitoring — feedback to improve future routing
```

**Our read:** This is the **supervised / learned-router** mental model: a **router module** decides *which model* before or instead of reading generator uncertainty. Matches RouteLLM, Hybrid LLM, RouterBench baselines — **not** our signal-first story.

## D.2 RouteLLM (main example in blog)

| Blog claim | Notes |
|------------|--------|
| **Problem** | Balance GPT-4-class quality vs cost | Same framing as our weak↔strong pool |
| **Training data** | Chatbot **Arena** preference battles | Supervised pairwise labels |
| **Router families** | Similarity-weighted (SW) ranking; **matrix factorization**; **BERT** classifier; **causal LLM** classifier | All predict preference / win from **query text** |
| **Augmentation** | Arena + **LLM-judge** augmented pairs | Blog says augmentation helps all routers |
| **Arena-only results (blog)** | Matrix factorization → **95% of GPT-4 performance** with **26%** GPT-4 calls (~**48%** cheaper vs random) | Blog-reported; cite RouteLLM paper to reproduce |
| **Augmented results (blog)** | MF → **95%** GPT-4 perf with **14%** GPT-4 calls (~**75%** cost reduction vs random) | Blog-reported |
| **Inference** | Router sees **query**; routes to strong or weak | **No** live entropy / logprob from candidate model at serve time |

**Repo cross-ref:** [`research/related_work.md`](related_work.md) § Binary Supervised Routing — Bradley–Terry win probability, threshold vs cost.

## D.3 Challenges named in blog

| Challenge | Blog wording | Relevance to us |
|-----------|--------------|-----------------|
| **Query complexity** | Hard to know if “capital of France” vs multi-step reasoning | Our **\(C(q)\)** targets this **without** preference training |
| **Latency** | Fast routing vs slow strong model | We defer routing; honest about weak-model probe cost |
| **Cost–quality tradeoff** | Cheap model risk on hard queries | Same as professor brief; we use **signals** not learned \(q\to M\) |

## D.4 Evaluation & benchmarks (blog)

| Benchmark | Blog use |
|-----------|----------|
| **GSM8K** | Math reasoning for routers |
| **MT-Bench** | RouteLLM figure (router performance) |
| **MBPP** | Code generation routing |
| **RouterBench** | Hu et al. — blog cites **405k+** inference outcomes; systematic multi-LLM routing eval; [withmartian/routerbench](https://github.com/withmartian/routerbench) |

Blog frames RouterBench as filling the lack of a **standardized** router evaluation harness.

## D.5 Other papers mentioned (titles only in blog)

| Paper (blog title) | Blog one-liner | Our lane |
|--------------------|----------------|----------|
| **Hybrid LLM** (Ding et al.) | Router assigns small vs large from predicted difficulty; up to **40%** fewer large-model calls | Supervised quality-gap labels; query-only at inference — see `related_work.md` |
| **Large Language Model Routing with Benchmark Datasets** | Learn router from benchmark datasets as binary classification | Supervised router-from-benchmarks (blog does not give arXiv id in paste) |
| **Harnessing the Power of Multiple Minds: Lessons Learned from LLM Routing** | Routing promising but **not universally feasible** | Blog caution — supports professor “pick one problem, validate signals first” |

## D.6 Three-way positioning (for our paper)

```text
                    LEARNED ROUTER              GENERATOR UNCERTAINTY
                    (query → model)             (weak model signals)
                    ─────────────────           ─────────────────────
Accredian survey    RouteLLM, RouterBench,     (not covered)
                    Hybrid LLM

Chuang arXiv        contrasts with extra         SLM UQ → offload;
                    routers in §2.2              calibration data

Sawant blog         complexity classifier +      entropy, logprobs,
                    then confidence zones          verifier cascade

OUR WORK            contrast case (not           C(q), H, U signals;
                    training q→M map)            rules/λ later; no Arena labels
```

**Key sentence for ACL:** Accredian-style routing **learns** \(q \to M\) from preferences or benchmark outcomes; we **define** interpretable signals and test which predict error **before** fixing a policy.

---

# Part C — Mapping all sources to **our** research

## C.1 Same high-level story

| Idea | Sawant | Chuang | Accredian survey | Our research |
|------|--------|--------|------------------|--------------|
| Cheap-first / binary pool | Tier 1→2→3 | SLM→LLM offload | weak vs strong (RouteLLM) | 8B↔70B (later route) |
| Decision signal | Entropy, logprobs, verifier | 8 UQ scores on SLM output | **Learned router** on query text | \(C\), \(H\), \(U\) |
| Query complexity | Heuristic classifier score | (not primary) | “Infer complexity” challenge | **\(C(q)\)** unsupervised |
| Training from labels | Historical outcomes | Probes + calibration pool | **Arena preferences**, RouterBench outcomes | Labels for **eval** only |
| Eval harness | Production zones | ROC AUC, routing curves | MT-Bench, GSM8K, **RouterBench** | S1–S6 then PGR/CPT |

## C.2 Closest academic neighbor: Chuang et al.

This paper is the **most directly comparable** prior work to our model-dependent phase:

| Their component | Our analogue | Notes |
|-----------------|--------------|-------|
| Average token prob (MC letter) | `p_max`, `margin`, `H` on A/B/C/D | **Closely related** — we score full letter distribution; they use prob of **chosen** option token |
| Perplexity | `perplexity_H` from letter distribution | Related family; we prefer answer-space \(H\) |
| Jaccard degree (5 samples) | Hotpot discrete semantic entropy | Both consistency-based; different clustering |
| p(True) | Not planned v1 | Extra generation round |
| Trained / OOD probe | **Not us** | Supervised on hidden states — strongest in their benchmark but violates unsupervised signal story |
| Verbalization | **Avoid** | They show it fails for SLM routing |
| Proxy routing data | **Partially relevant** | Analogous to pooled **calibration data** on our calib benchmarks — test later |
| Uncertainty–correctness AUC | **S2 experiment** | Our `needs_strong` / weak-wrong AUROC is the same scientific question |
| Routing curve | **Later** | After S1–S6 signal validation |

**Critical distinction for the paper:** They **benchmark eight UQ estimators** and propose **calibration-data threshold transfer** (arXiv §4). We define **unsupervised** \((C(q), H(q|M), U(q|M))\) and test signal families **before** routing.

## C.3 Where we **differ** from both (ACL framing)

| Blog / Chuang / Accredian | Our ACL framing |
|---------------------------|-----------------|
| Supervised router (RouteLLM, Hybrid, probes) | **Unsupervised** named signals; no Arena-trained \(q\to M\) |
| Router learns from preferences / benchmark outcomes | Labels for **signal validation** and optional \(\lambda\) on calib |
| Query-only at inference (RouteLLM family) | **\(C(q)\)** query-only **plus** \((q,M)\) probe for \(H,U\) |
| Generator uncertainty only at train label time (Hybrid) | Generator uncertainty at **inference** for routing signals |
| Token entropy / verifier (Sawant) | MC letter distribution; verifier deferred |
| RouterBench multi-LLM scale | Binary 8B↔70B first (professor scope) |

## C.4 Layer-by-layer alignment (unified)

```text
OUR STACK                          BLOG ANALOGUE
─────────────────────────────────────────────────────────────
C(q)  model-independent            Complexity classifier (Post 3.1)
       complexity φ(q)              (we: unsupervised, not learned router)

H, p_max, margin, …                Entropy + logprob confidence
       answer_scores on MC          (we: answer-space, not token stream)

U(q|M)  paraphrase disagreement    Related to self-consistency /
                                    input-robustness (different mechanism)

(discrete SE on Hotpot)            Self-consistency / semantic entropy

[calib: tune τ, optional λ]        Calibration + threshold zones

[routing: escalate if uncertain]   Escalation router + verifier

NOT IN V1                          Verifier, probes, speculative parallel tiers

CHUANG ET AL.                      OUR RESEARCH
─────────────────────────────────────────────────────────────
8 UQ benchmark                     S2: AUROC families on fit
Proxy routing CDF                  Optional: calib hold-out (Chuang §4 calibration data)
Trained/OOD probe (best AUC)       Explicitly NOT our method
Jaccard / sample consistency       Hotpot discrete SE
Perplexity / avg token prob        MC H, p_max, margin (answer_scores)
No query complexity signal         C(q) model-independent (unique to us)
No paraphrase uncertainty          U(q|M) paraphrase disagreement (unique to us)
```

---

## C.5 Practical mapping: their UQ → our `answer_scores`

Blog measures uncertainty on **generated tokens**. We measure uncertainty on **the answer choice** (MC) or **answer meaning clusters** (Hotpot). Same *role* (confidence for escalation), different *estimator*.

| Source signal | Closest in our plan | Same? |
|---------------|---------------------|-------|
| Chuang: avg token prob (MC letter) | `p_max`, `H`, `margin` | **Closely related** — not identical if chosen token ≠ argmax letter |
| Chuang: perplexity | `perplexity_H` | **Related** — monotone with \(H\) on letters |
| Blog: mean token entropy | `H` over A/B/C/D | **Concept** same; answer-space not token stream |
| Blog: first-token logprob | `p_max`, `surprisal` | **Analogous** |
| Chuang: jaccard degree (5 samples) | Hotpot discrete SE | **Partial** — semantic clusters vs Jaccard |
| Blog / Chuang: self-consistency | Hotpot samples | **Partial** |
| Chuang: trained/OOD probe | — | **Not us** — supervised hidden-state classifier |
| Chuang / blog: verbalization | — | **Avoid** — they show SLM verbalization misaligns |
| Our unique: \(C(q)\) | `signals/query/` | **Not in either source** |
| Our unique: \(U(q|M)\) paraphrase | `signals/query_model/` | **Not in either source** |

**Takeaway:** On ARC/MMLU, our MC letter probe is in the same **token-probability family** as their average token prob / perplexity — with explicit **family ablation** (S8). Their top alignment methods include **supervised probes**, which we do not adopt.

---

## C.6 Process alignment: what we do now vs later

### Phase A — model-dependent signals (this workflow)

Adapted from alignment doc **Part C.6** — signals only:

1. Extract \(C(q)\) — done (`signals/query/`).  
2. Extract \((q,M)\) **answer_scores** + **\(U\)** — §3–§7 here.  
3. **S1–S8** on fit; **AUROC** alignment (Chuang RQ1).  
4. **No** router, τ, routing curves, verifier.

See [`signals/query_model/WORKFLOW.md`](../signals/query_model/WORKFLOW.md) §2 for full adaptation table.

### Phase B — router (deferred)

1. **Alignment check on calib** (Chuang Finding ❶): reliability / AUROC per signal family — not just one \(\tau\).  
2. **Threshold policy:** e.g. if \(C(q)\ge\tau_C\) → strong; elif \(H_w\ge\tau_H\) or \(U_w\ge\tau_U\) → strong; else keep weak answer from same probe.  
3. **Calibration-style thresholding (optional):** pooled calib split (Chuang §4 **calibration data** idea) — only after signals validated.
4. **Routing curve + PGR/CPT** on eval — compare to always-weak / always-strong.  
5. **Optional \(\lambda\):** combine \(C, H, U\) on calib.  
6. **Extensions:** verifier (blog); gated multi-sample on Hotpot only.

---

## C.7 What we should **borrow**

| Borrow | From | Why |
|--------|------|-----|
| **Uncertainty–correctness alignment (AUROC)** | Chuang RQ1 | Primary metric for S2; cite as established routing criterion |
| **Routing curve (accuracy vs offload rate)** | Chuang §3 | Standard eval when routing phase starts |
| **Proxy / pooled CDF for \(\tau\)** | Chuang §4 (calibration data) | Hold-out calibration pool; leave-one-out evaluation pattern |
| **Reject verbalization for SLM routing** | Chuang Finding ❷ | Justifies logprob/entropy families over verbalized confidence |
| **Two-output framing** (decision + confidence) | Blog | \(C(q)\) + \(H,U\) |
| **Confidence zones** | Blog | Report by quantile bands on calib |
| **Cost gating** | Both | Paraphrase \(U\) / Hotpot samples only in mid-confidence band |
| **Failure modes** | Both | Overconfidence, probe supervision trap, verifier blind spots |

## C.8 What we should **not** copy

| Skip or defer | Why |
|---------------|-----|
| Trained / OOD probe as our method | Supervised; they win benchmark but break unsupervised claim |
| Verbalization UQ | Empirically weak for SLM routing in Chuang et al. |
| Full 12-SLM × 15-dataset scale v1 | Professor: one problem; our 3-benchmark pool first |
| Verifier as core (blog) | Product cascade; not signal paper |
| Token entropy on MC as primary | Letter distribution is correct estimator |
| Claiming query-only routing | Both sources require SLM/weak inference for UQ |

---

## C.9 Suggested related-work sentences (draft)

**Chuang et al. (arXiv:2502.04428):**
> Chuang et al. (2025) benchmark eight UQ estimators for on-device SLM→LLM routing over **1500+** settings (**8 SLMs, 2 LLMs, 14 datasets**), showing that **uncertainty–correctness alignment** (ROC AUC) predicts routing utility better than assuming calibration alone, and that confidence histograms are largely determined by **(SLM, UQ)** rather than downstream task identity. They propose **calibration data** — a binned hold-out set — to set routing thresholds on a new dataset without target-task instances. We adopt their alignment metric and routing-curve evaluation framing, but study an **unsupervised signal inventory** adding \(C(q)\) and paraphrase \(U(q\mid M)\), and we do not use their supervised hidden-state probes.

**Accredian survey + RouteLLM:**
> Survey and systems posts (Tiwari, Accredian) popularize **learned** LLM routers trained on preference data (RouteLLM) and benchmarked on RouterBench. We treat this line as the main **supervised** contrast: they learn \(P(\text{strong wins} \mid q)\) or quality-gap labels from past outcomes; we extract **unsupervised** complexity and generator-side signals and evaluate whether they predict weak-model failure before any router is trained.

**Blog + combined:**
> Industry and systems work (Sawant, 2026) and recent benchmarking (Chuang et al., 2025) converge on **confidence-gated escalation** after a cheap model attempt. We sit between query-only routers (RouteLLM) and full probe-trained cascades: generator-side signals without training a router from preferences, with signal validation preceding any fixed policy.

---

## C.10 Alignment checklist (for next implementation)

- [ ] `answer_scores` include `H`, `p_max`, `margin` (Chuang: avg token prob + perplexity family).  
- [ ] S2 reports **AUROC** uncertainty–correctness alignment (Chuang metric).  
- [ ] Document MC probe **is** the weak answer when confident (no double inference).  
- [ ] Signal analysis by **family** — avoid duplicate monotone metrics (S8).  
- [ ] Related work: cite Chuang et al.; contrast trained probe vs our unsupervised signals.  
- [ ] Routing phase: routing curve + optional **calibration hold-out** (Chuang §4; arXiv term).
- [ ] Do **not** use verbalization UQ; cite Chuang Finding ❷ if discussed.  
- [ ] Paper: \(C(q)\) and \(U\) as contributions absent from Chuang’s 8 methods.  
- [ ] Verifier / speculative execution = future work (blog).

---

---

## Verification log (anti-hallucination)

| Claim | Status | Source |
|-------|--------|--------|
| 8 SLM / 2 LLM / 14 datasets / 1500+ | ✅ | arXiv abstract, §3.1, conclusion |
| Term **calibration data** | ✅ | arXiv abstract, §4 |
| 30 bins, 10% per bin, leave-one-out 13+1 | ✅ | §4.2 |
| ROC AUC alignment; top-3 UQ: Trained Probe, OOD Probe, Perplexity | ✅ | Observations ❶–❸ |
| Danube-3.1-4B / PIQA / top 20% example | ✅ | Observation ❹, Figure 3 caption |
| GPT-4o mini judge & offload LLM | ✅ | §3.1 |
| 12 SLM / 4 LLM / 15 datasets / 5000+ | ❌ in arXiv | `[extended workshop]` PDF only |
| **Proxy routing data** name | ❌ in arXiv | workshop rename |
| RMS 0.001 vs 0.029 | ❌ in arXiv | workshop appendix |
| quodlibeta GitHub | ❌ in arXiv PDF | workshop abstract only (not verified here) |
| NeurIPS 2025 workshop venue | ❌ on arXiv page | workshop PDF footer |
| Exact routing-ratio formula | ⚠️ not in prose | figures only — do not invent |
| Sawant blog thresholds (0.78/0.55) | ✅ | user-pasted excerpt |
| RouteLLM MF 26% / 14% GPT-4 calls @ 95% perf | ⚠️ blog-reported | Accredian paste; verify in Ong et al. |
| RouterBench 405k outcomes | ⚠️ blog-reported | matches RouterBench paper claim; not re-checked here |
| Accredian five-step router loop | ✅ | user-pasted excerpt |
| Hybrid 40% fewer large calls | ⚠️ blog-reported | Ding et al.; see `related_work.md` |
| Our \(C(q)\), \(U\), S1–S6 plan | ✅ | repo `WORKFLOW.md` files |

**How to cite in our paper:** prefer **arXiv:2502.04428** with numbers from **B.0** table. If you read the workshop PDF, say so explicitly and use its scale.

---

## C.11 Log

| Date | Note |
|------|------|
| 2026-07-15 | Initial alignment doc from Sawant (May 2026) blog excerpt. |
| 2026-07-15 | Added Chuang et al. summary — **first draft mixed arXiv with extended workshop PDF** (wrong scale/models/terms). |
| 2026-07-15 | **Recheck:** corrected to **arXiv:2502.04428 only** — 8 SLM / 2 LLM / 14 datasets / 1500+ / **calibration data** / GPT-4o mini; removed unverified routing-ratio formula, RMS table, NeurIPS+github as primary refs; flagged workshop PDF separately. |
| 2026-07-15 | Added **Part D** — Accredian/Tiwari survey (RouteLLM, RouterBench, Hybrid); learned-router lane; blog-reported numbers flagged ⚠️. |
