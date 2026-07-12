
### Routellm

**Paper: RouteLLM: Learning to Route LLMs with Preference Data (Ong et al., ICLR 2025)**

**1. Core problem — technical and application-wise**

Technically: binary routing — learn a function R: Q → {M_weak, M_strong} that decides, per query, whether a weak/cheap or strong/expensive LLM should answer it, using only the query text (no answer generation from either model at inference time). This is framed as a win-prediction problem: estimate P(strong wins | query), then threshold it against a cost parameter α.

Application-wise: cost/latency reduction in production LLM serving. If you always call GPT-4-class models you get top quality but pay 50-100x more than a small open model per token. The paper's goal is to recover most of the strong model's quality while sending the bulk of easy/simple traffic to a cheap model — directly a deployment cost-optimization problem, not just an academic accuracy problem.

This is close to your SAAR/PF-AAR framing except RouteLLM is binary (2-way, quality-only signal from preference labels) whereas your gates use solvability + agent ranking + entropy/uncertainty signals across N agents.

**2. Training dataset format and openness**

Primary dataset: D_arena — 80K battles from Chatbot Arena (open, human preference triplets: query, response_A, response_B, human vote). They filter to 65K after removing prompts <16 chars, spanning 64 models, clustered into 10 tiers by Elo score (dynamic programming to minimize intra-tier variance). Only (query, win_label) pairs are kept — response text is discarded for training the router itself.

Two augmentation sets, both used to fix label sparsity:
- D_gold: ~1,500 samples from MMLU validation split. Golden-label correctness used to derive s/w comparison label directly (open, public benchmark data).
- D_judge: ~120K samples from the Nectar dataset, with GPT-4 as judge producing pairwise labels between GPT-4 response (as strong) and a newly-generated Mixtral-8x7B response (as weak). This cost ~$700 in GPT-4 API calls. Nectar is open source; the judge labels are their own synthetic addition.

So: training data is fundamentally preference triplets (query, label ∈ {strong_wins, tie, weak_wins}), not query-only labels, and not full generation-based signals like your entropy/paraphrase-uncertainty approach. All source datasets (Arena, MMLU, Nectar) are open; the GPT-4-judge labels they generated are open-sourced via their released framework but represent a reproducibility cost (~$700 + rate limits) if you wanted to redo it.

**3. Central innovative idea**

Formulate routing as a Bradley-Terry-style win-probability estimation problem learned purely from human pairwise preference data (Chatbot Arena), rather than from synthetic quality scores (like Hybrid-LLM's BARTScore-derived labels) or reward-model outputs (like Zooter's QwenRM labels). The router never needs the actual model responses at train or inference time — only the query and a scalar win probability.

They test four different parameterizations of the same win-prediction objective (similarity-weighted BT ranking, matrix factorization, BERT classifier, causal LLM classifier) under one unified framework, then show that augmenting sparse human preference data with golden-labeled and LLM-judge-labeled data is what actually makes the higher-capacity models (BERT, causal LLM) usable — without augmentation they perform at or below random.

**4. What's novel and why it works**

- Human-preference-grounded training signal instead of synthetic/proxy labels — arguably more calibrated to what "better response" means to real users, and avoids compounding a proxy metric's bias into the router.
- Tiering models into 10 Elo-based buckets solves the sparsity problem (raw pairwise coverage <0.1%) without collapsing to a single strong/weak pair, so the router learns query-level separability signal that's more general across model pairs.
- Data augmentation as the key lever: this is the real "why it works" — the router's capacity (BERT, causal LLM) only pays off once the label distribution is dense/matched to the eval distribution. This is shown quantitatively via the benchmark-dataset similarity score (Section 5.3), which is a genuinely nice diagnostic: it correlates predicted APGR-improvement with embedding similarity between the training and eval distribution, giving an a priori explanation for why routers trained on D_arena flop on MMLU/GSM8K (low similarity, ~0.48-0.49) but do fine on MT Bench (0.60+).
- Generalizes to unseen model pairs (Claude 3 Opus/Sonnet, Llama 3.1 70B/8B) with zero retraining — because the router learns query-level complexity signal rather than model-pair-specific patterns. That's the strongest empirical claim in the paper.

**5. Limitations and how you might address them architecturally**

Their own stated limitations:
- Binary-only (strong/weak); no N-way multi-model routing — extending to N models is future work. This is directly where SAAR/PF-AAR already goes further with N-tier and multi-agent routing.
- No real generation-based signal — router only sees the query, never any model's actual behavior on it (no entropy, no self-consistency, no calibration signal). This is exactly the gap your entropy + paraphrase-uncertainty signals fill, at the cost of needing a forward pass (your "pre-inference" framing tension).
- In-domain assumption: performance depends heavily on train/eval distribution similarity (Section 5.3) — routers trained on Arena chat data generalize poorly to structured reasoning benchmarks (MMLU, GSM8K) unless augmented. A fix: train on a broader, stratified mixture of task types from the start rather than post-hoc augmentation, or use a task-type/domain classifier as an auxiliary signal (your query-complexity gate does something like this).
- No handling of tie labels explicitly discussed as a separate modeling target — ties are folded into the BT loss but the paper doesn't analyze how tie-heavy queries behave differently.
- Fixed threshold α is global, not query-adaptive — a single cost threshold doesn't account for query-specific value of being right (e.g., safety-critical vs. casual queries could warrant different α). A calibration-aware or per-query risk-adjusted threshold (closer to your solvability-aware framing) would be a natural extension.
- Router overhead reported but SW ranking needs a full corpus similarity computation at inference (Table 7 shows it's the slowest, ~2.9 req/s) — doesn't scale as well as the parametric routers (matrix factorization, BERT) for high-throughput serving.

**6. Training procedure — inputs and outputs, step by step**

Step 1 — Data collection: gather (query, response_A, response_B, human_vote) triples from Chatbot Arena; discard response text, keep (query, model_A_id, model_B_id, vote).

Step 2 — Tiering: cluster 64 models into 10 tiers by Arena Elo score using dynamic programming to minimize intra-tier variance. Define M_strong = top 2 tiers, M_weak = tier 3. Relabel each triple as (query, l_{s,w}) where l ∈ {wins, tie, win_w}.

Step 3 — Augmentation (optional per experiment): add D_gold (query, golden-label correctness comparison from MMLU) or D_judge (query, GPT-4-judged win label between a real GPT-4 response and a newly-generated Mixtral response on Nectar prompts).

Step 4 — Model-specific training, four parallel architectures sharing the same win-probability objective P_θ(win_s | q):
- SW Ranking: no training — computes similarity-weighted Bradley-Terry coefficients at inference time using cosine similarity between the query and training queries (text-embedding-3-small embeddings).
- Matrix Factorization: learns bilinear scoring function δ(M,q) = w2ᵀ(v_m ⊙ (W1ᵀv_q + b)) over model and query embedding vectors; trained on 8GB GPU, ~10 epochs, batch 64, Adam lr=3e-4.
- BERT classifier: full fine-tune of BERT-base, CLS token → sigmoid logistic head; 2×L4 GPUs, ~2000 steps, batch 16, seq len 512, lr=1e-5.
- Causal LLM classifier: Llama 3 8B fine-tuned in instruction-following format; appends comparison labels as new vocabulary tokens, computes win probability via softmax over label classes in next-token prediction; 8×A100, ~2000 steps, batch 8, seq len 2048, lr=1e-6.

Step 5 — Inference: given a new query, compute P(win_s | q); apply threshold α to route to weak or strong model; the chosen model alone generates the final answer (single LLM call, addressing latency).

**7. Metrics and evaluation per stage**

Primary metrics:
- c(R^α) — cost efficiency = fraction of calls routed to the strong model.
- r(R^α) — average response quality (accuracy on MMLU/GSM8K, LLM-judge score on MT Bench).
- PGR (Performance Gap Recovered) = (r(router) − r(weak)) / (r(strong) − r(weak)) — how much of the quality gap between weak and strong is closed.
- APGR (Average PGR) — integral of PGR over the full cost curve (0-100% strong calls), computed by discretizing into 10 buckets — the single scalar "how good is this router across all cost budgets" summary.
- CPT(x%) — call-performance threshold — minimum % of strong-model calls needed to reach a target PGR (e.g., CPT(50%), CPT(80%)) — practically the most deployment-relevant metric ("how cheap can I be at X% quality recovery").
- Benchmark-dataset similarity score (Eq. 14) — average max cosine similarity between eval prompts and training prompts — used post-hoc to explain cross-benchmark generalization gaps, not a training-time metric.

They evaluate on three held-out academic benchmarks (MMLU 5-shot, MT Bench, GSM8K 8-shot) with a data-contamination check via embedding similarity threshold (0.95) against training data, and report each router × each training-data variant in separate tables (Tables 1-3), plus adaptability tests on two unseen model pairs (Table 4) and cost/inference overhead (Tables 6-7).

**8. Production/deployment use — dataset format, evaluation**

They benchmark deployment cost directly: routing overhead per router type (Table 7) — cost per million requests, requests/sec throughput, hourly VM cost, using Google Cloud g2-standard-4 (single L4 GPU) for GPU-based routers and n2-standard-8 (CPU) for SW ranking. They frame the router as a lightweight pre-processing service sitting in front of the actual LLM serving stack — the router's own inference cost is shown to be <0.5% of the cost of a single GPT-4 call, which is the argument for why it's viable in production.

They also benchmark against two real commercial routing products (Unify AI, Martian) on MT Bench (Appendix E), using the same routing-performance-vs-cost curve format, showing their open router beats both by requiring up to 40% fewer GPT-4 calls at equivalent quality. This is the closest thing in the paper to a "production-readiness" comparison, though it's still an offline benchmark rather than live traffic.

They open-sourced the training/serving/eval framework itself so others can plug in their own model pairs and preference datasets — this is their answer to the production-deployment question rather than a live production case study.

---

A few things worth flagging for your related-work section specifically: their "pre-inference" framing is much cleaner than yours will need to be, precisely because they never use entropy or generation-based uncertainty — the router only ever sees the query text. That's the tension you'll want to name explicitly and differentiate against: your signals require partial/full generation passes, so your paper needs its own honest framing of what "unsupervised" and "query routing" mean when your later gates aren't purely pre-inference. RouteLLM gives you a clean contrast case to cite for that distinction.



### Hybrid LLM

**Paper: Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing (Ding et al., Microsoft, arXiv 2404.14618)**

**1. Core problem — technical and application-wise**

Technically: binary routing r: X → {0,1} between a small model S and large model L, where the decision is based on predicting the *quality gap* H(x) = q(S(x)) − q(L(x)) for a query x, rather than just a win/loss preference label. The router's job is to find "easy" queries — defined purely by outcome (small model matches or beats large model), not by any intuitive notion of query difficulty — and send those to S.

Application-wise: same MLaaS cost-reduction framing as RouteLLM, but with two additional angles emphasized here — (a) edge/cloud routing (small model runs locally on a laptop/phone, large model is the paid cloud API call — so a user-side cost reduction, not just provider-side), and (b) a tunable quality-cost dial at test time via a threshold, framed explicitly around a target performance-drop tolerance rather than a fixed α.

**2. Training dataset format and openness**

Dataset: MixInstruct (Jiang et al. 2023) — open source, ~20K real-world instructions aggregated from 4 public sources (Alpaca-GPT4, Dolly-15K, GPT4All-LAION, ShareGPT). They sample 10K training examples uniformly, and for each generate **10 responses from every LLM under consideration** (this is the critical difference from RouteLLM — they need actual generations, not existing preference labels).

Label format is a scalar/probabilistic quality-gap estimate rather than a discrete win/tie/lose category:
- y_det: binary label = 1[q(S(x)) ≥ q(L(x))], computed from a single sampled response pair per model.
- y_prob: soft label = empirical Pr[q(S(x)) ≥ q(L(x))], estimated from 10 sampled responses per model (so 10×10=100 implicit pairwise comparisons per query, though they describe it as averaging the indicator over the 10 samples).
- y_trans(t): relaxed tail-probability label Pr[q(S(x)) ≥ q(L(x)) − t] for a calibrated t* chosen via grid search to maximize label spread.

Quality function q(·) = BART score (Yuan et al. 2021) — computed automatically, not human-labeled, which is what makes generating 10 responses × multiple model pairs tractable without human annotation cost (unlike RouteLLM's Chatbot Arena human votes or GPT-4-judge cost).

Router itself: DeBERTa-v3-large (300M params), trained via standard fine-tuning, 5 epochs, single A100. Code and data pipeline open-sourced at the repo listed in the paper.

**3. Central innovative idea**

Model the routing decision as **estimating a quality-gap random variable** H(x), not just a static win/loss label — and explicitly model the *distributional* nature of that gap because LLM generations are stochastic (Figure 3 shows visibly different BART-score distributions per sample). This is the paper's core conceptual move: routing-as-classification (RouteLLM's framing) is a special/degenerate case of routing-as-estimating-a-distribution-over-quality-differences.

The second innovative piece is the **data transformation trick (Section 3.3)**: when the large model dominates almost everywhere (large capability gap), Pr[H(x) ≥ 0] collapses to ~0 for 90% of queries, which starves the training signal (severe class imbalance). Rather than reweighting or resampling (standard imbalance fixes), they shift the decision boundary itself — relax to Pr[H(x) ≥ −t] for a learned threshold t* chosen to maximize label spread via a simple pairwise-distance objective. This sidesteps the standard imbalanced-learning toolbox entirely by redefining what counts as "easy" in a way that still serves the deployment goal (cost reduction with bounded quality drop), even though it deviates from the literal question "when does S beat L."

**4. What's novel and why it works**

- Treating query difficulty as inherently **probabilistic** (soft labels from repeated sampling) rather than a point estimate is a genuinely different modeling choice from RouteLLM, and their ablation (r_det vs r_prob vs r_trans) is a clean isolation of this idea's contribution — r_prob consistently beats r_det across all performance-gap regimes (Table 1), which supports the claim that capturing generation stochasticity, not just the outcome, is what improves routing.
- The t*-relaxation is why this method (unlike RouteLLM, unlike naive win-rate routers) still works when the small model is dramatically weaker — their large-gap experiment (FLAN-T5-800M vs Llama-2-13B) shows r_trans achieving 40% cost advantage at 10.3% quality drop, clearly beating r_det/r_prob (13.8%/13.1%) in that regime specifically (Table 1). This is the paper's strongest empirical differentiator.
- BART score as training signal is deliberately chosen for being cheap to compute at scale (no GPT-4 judge cost) — but they's careful to validate it's not just a proxy of convenience: Section 4.6 explicitly checks correlation between BART score and GPT-4-based evaluation, and shows routing performance degrades gracefully with weakening correlation (Figure 7) — a legitimate validity check rather than an unexamined assumption.
- Query-pair generalization test (Section 4.7) — correlating quality-gap distributions of a training pair vs. a novel test pair to predict whether a router transfers — is a more principled generalization diagnostic than RouteLLM's "we tried it and it worked" approach; it gives you an a priori, checkable criterion for whether a router will transfer to a new model pair.

**5. Limitations and how you might address them**

Authors' own stated limitations (Section 5):
- Purely query-input-based routing, no task-awareness — they explicitly flag task labels as an unexplored signal. Your query-complexity gate + agent-tier framing in SAAR is a direct answer to this gap.
- Binary-only, no N-model routing — same limitation as RouteLLM; both papers converge on this as future work, which is exactly your PF-AAR/SAAR's stated contribution vs. this line of prior work.
- Fixed model pair + fixed data distribution at train and test time — no OOD generalization mechanism beyond the correlation-based heuristic in 4.7, which is diagnostic, not adaptive (it tells you *whether* a router will fail, not how to fix it if it does).
- Reliance on automatic quality metrics (BART score) that require correlation-checking against human/GPT-4 judgment per-domain before trusting them; this is a manual, offline calibration step, not something the framework does automatically.

Additional limitation not explicitly named but visible in the data: the whole framework requires generating **10 responses per query per model at training time** — this is expensive relative to RouteLLM's approach (which reuses existing preference labels without generating anything itself). For your entropy/uncertainty framing, this is actually closely related — you're both paying a "many-samples-per-query" cost, and it's worth explicitly contrasting: their 10 samples are used to estimate an outcome-quality gap; yours are used to estimate model uncertainty/entropy directly. That's a meaningful signal-source distinction to draw out in related work — they sample to measure *comparative quality*, you sample to measure *self-uncertainty*, and it's worth asking in your paper whether these are correlated or genuinely orthogonal signals.

A concrete architectural extension: their t* is chosen via grid search on a single scalar objective (maximize average pairwise label spread) — this is a coarse, non-adaptive calibration. A learned or per-query-adaptive relaxation (e.g., predicting t as a function of query features, or calibrating t differently across capability-gap regimes automatically rather than per-model-pair by hand) would remove a manual tuning step and could plausibly be framed as a refinement worth citing/building on.

**6. Training procedure — inputs and outputs, step by step**

Step 1 — Sample 10K queries from MixInstruct train split.

Step 2 — For each query, generate 10 responses from S and 10 from L (stochastic decoding, not greedy) — this is the expensive step relative to RouteLLM, since it requires actual inference from both models across the full training set, 20 generations per query.

Step 3 — Score every response with BART score q(·).

Step 4 — Compute labels per query depending on router variant:
- r_det: single-sample indicator, y_i = 1[q(S(x_i)) ≥ q(L(x_i))] using just one response pair.
- r_prob: soft label y_i = mean indicator over all 10-sample pairs (empirical estimate of Pr[H(x)≥0]).
- r_trans: grid-search t* by maximizing Σ|y_trans_i(t) − y_trans_i'(t)| pairwise spread over the training set, then recompute soft labels as Pr[H(x_i) ≥ −t*].

Step 5 — Fine-tune DeBERTa-v3-large with binary cross-entropy loss against the chosen label type, 5 epochs, single A100, batch details not fully specified beyond that.

Step 6 — Validation-based checkpoint selection (best epoch on val split).

Step 7 — Inference: single forward pass of DeBERTa over the query → score p_w(x) ∈ [0,1]. Threshold τ (chosen empirically, Section 4.5) determines routing: score > τ → route to small model, else large model. τ is chosen via grid search on a 500-sample calibration set to hit a target performance-drop ceiling (e.g., ≤1% BART score drop).

**7. Metrics and evaluation**

- BART score (Yuan et al. 2021) — primary quality metric q(·), chosen for being cheap and shown (Section 4.6, and Appendix C.2 case study) to correlate reasonably with human/GPT-4 judgment.
- Cost advantage — % of queries routed to the small model (equivalent to RouteLLM's c(·), but inverted framing — "advantage" rather than "% strong calls").
- Performance drop (%) relative to all-at-large baseline — direct quality cost of routing, reported at fixed cost-advantage checkpoints (10%, 20%, 40%) in Table 1, rather than an integral metric like RouteLLM's APGR.
- Router validation metric (Section 4.3, Figure 6): average quality-gap difference between queries routed to S vs L, compared against random routing — a sanity check that the router is actually separating easy/hard queries rather than routing well by coincidence.
- Latency (Table 2): explicit wall-clock router latency vs. LLM latency (router: 0.036s vs FLAN-T5-800M: 0.46s vs Llama-2-13B: 14.61s) — a much more concrete overhead argument than RouteLLM's cost-per-request framing.
- Cross-metric validation (Section 4.6): Pearson/Spearman correlation between BART score and GPT-4 score, used to explain when/why routing trained on BART generalizes to GPT-4-scored evaluation.
- Cross-pair generalization (Section 4.7): Pearson/Spearman correlation between quality gaps of training vs. test model pairs, used as a predictive indicator of transferability.

Baselines: all-at-large, all-at-small, random — same minimal baseline set as RouteLLM; notably **no comparison to RouteLLM itself** since these are contemporaneous works (this paper predates RouteLLM's ICLR 2025 camera-ready, though RouteLLM cites and differentiates from this one directly, as you saw in the previous paper's related work section).

**8. Production/deployment use**

Two explicit deployment scenarios (Figure 2):
- Consumer/edge: user runs S locally (laptop/phone), calls cloud API (L) only for hard queries — direct cost savings to the end user, motivated by real deployment paths (Llama-2-Onnx runs on laptops, PaLM-2 runs on mobile).
- Platform/provider: MLaaS backend automatically routes across cost tiers without the user noticing, controlling the quality/cost knob via the threshold τ.

Deployment format: router runs as a lightweight pre-processing service (single DeBERTa forward pass, 0.036s), threshold set per deployment scenario using a small calibration set (500 validation samples, Section 4.5) to hit a target quality-drop ceiling (e.g., ≤1%) — this calibration procedure is the paper's answer to "how do you actually set this in production" and is more concretely worked out than RouteLLM's single global α.

They don't report a live production case study or real serving-system integration — like RouteLLM, this is an offline benchmark-based validation, with the open-sourced code (m365-core/hybrid_llm_routing) as the production-readiness artifact rather than a deployed system trace.

---

One thing worth flagging directly for your ACL paper's related-work framing: this paper and RouteLLM are genuinely complementary precedents for your "pre-inference vs. requires-generation" tension. RouteLLM never generates anything — pure query-text-based preference-label prediction. This paper generates 10 responses per model per query to build its training *labels*, but the trained router itself is still query-text-only at inference time (single DeBERTa pass, no generation needed at serving time). Your SAAR/paraphrase-uncertainty signals are different again — they need generation at **inference time**, not just training time — so you have three distinct positions on the "how much does routing cost" axis: (1) RouteLLM — no generation ever, (2) Hybrid LLM — generation only at training time to build soft labels, (3) yours — generation required per-query at inference time to compute entropy/uncertainty. That's a clean three-way related-work table you could build directly for your methodology section's honest framing of the pre-inference tension.



## IRT ROUTER
**Paper: IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory (Song et al., USTC/NetEase, ACL 2025)**

**1. Core problem — technical and application-wise**

Technically: N-way (not binary) routing across 20 candidate LLMs, framed as a psychometric measurement problem borrowed wholesale from Item Response Theory. Each LLM is treated as a "test-taker" with a latent multidimensional ability θ_M ∈ ℝ^N; each query is treated as a "test item" with latent difficulty b_i and discrimination a_i. The router predicts P̂(q_i, M_j) — the probability model M_j answers query q_i correctly — via an IRT-style logistic interaction function, then combines this with a fixed per-model cost C(M_j) into a linear scoring function S = α·P̂ − β·C, and routes to argmax_j S(q_i, M_j).

Application-wise: this directly targets the "many candidate LLMs" version of the routing problem that RouteLLM and Hybrid-LLM both flagged as future work (both were binary strong/weak). It's explicitly framed as a paid-tiered-API cost optimization problem across a real, heterogeneous 20-model pool spanning $0.0137/M to $10/M in pricing (Table 5) — this is the most realistic multi-vendor deployment scenario of the three papers you've now reviewed.

**2. Training dataset format — is it open source?**

Format: D_train = {(q_i, M_j, y_ij)} where y_ij ∈ [0,1] is the *empirical performance score* of model M_j on query q_i, computed by actually running all 20 LLMs on every query and scoring against ground truth with task-appropriate metrics (accuracy, EM, F1, pass@1 — Table 7). This is a dense full-matrix design: every query is paired with every model (24,430 queries × 20 models = 488,600 training tuples), unlike RouteLLM's sparse pairwise preference data or Hybrid-LLM's 10-sample generation approach.

Source datasets: 8 in-distribution benchmarks (MMLU, CMMLU, ACLUE, ARC_C, HotpotQA, SQuAD, MATH, MBPP) + 4 out-of-distribution benchmarks (CEVAL, CommonsenseQA, GSM8K, HumanEval) — all open, standard academic benchmarks, no proprietary data. The 20 candidate LLMs span both closed APIs (GPT-4o, Gemini, GLM) and open-weight models (Llama, Qwen, Mistral, DeepSeek), with pricing sourced from official vendor rates or Together AI for open-source hosting (Table 5).

Openness: code released at github.com/Mercidaiha/IRT-Router. This is the most reproducible of the three papers in terms of dataset construction methodology (no human preference votes, no GPT-4-judge cost) — but note it requires actually querying all 20 LLMs on ~24K queries, which is a non-trivial one-time API cost to reproduce (488,600 total generations).

**3. Central innovative idea**

Borrow the IRT framework wholesale from psychometrics: instead of treating routing as classification (RouteLLM) or quality-gap regression (Hybrid-LLM), model it as a **latent trait measurement problem** — LLMs have latent, multidimensional "abilities," queries have latent "difficulty" and "discrimination" parameters, and the interaction between them (via a logistic function, Eq. 5) determines predicted performance. This borrows IRT's **Monotonicity assumption** directly: as an LLM's ability increases, predicted performance on a query should only increase — which is what gives the framework its interpretability guarantee (ability rankings should behave sensibly, e.g., a 70B model should dominate its own 8B counterpart in nearly every ability dimension, which they verify empirically in Figure 4).

Two concrete implementations:
- **MIRT-Router**: lightweight, linear/bilinear logistic interaction (θ_M, a_i, b_i all derived via learned linear projections of BERT embeddings).
- **NIRT-Router**: adds a **predefined, human-interpretable 25-dimensional ability taxonomy** (Reasoning, Mathematical calculation, Ethical consideration, etc. — Appendix C.1) plus a query-specific relevance vector r_qi (which dimensions matter for this query, determined via UMAP+HDBSCAN clustering + LLM-labeled cluster abilities), and a small neural interaction layer instead of a pure logistic form.

**4. What's novel and why it works**

- The interpretability claim is genuinely substantiated, not just asserted — they show concrete qualitative validations: (a) Llama-3.1-70B's learned ability vector dominates Llama-3.1-8B's in nearly every dimension (matches known scaling expectations), (b) GPT-4o-Mini+CoT's ability exceeds GPT-4o-Mini's (matches known CoT benefit), (c) MATH dataset queries' learned difficulty correlates with their human-annotated "Level" labels (Figure 5), (d) routing behavior itself is interpretable — top-30%-difficulty queries get routed to DeepSeek-Chat 80% of the time; bottom-30% get routed to the cheaper Qwen2.5-32B-GPTQ 99% of the time (Section 6.2, "Routing Analysis"). This is a much stronger interpretability validation than either prior paper attempted.
- N-way routing genuinely outperforms binary routing baselines run on the same data — Tables 1-2 show IRT-Router beating RouterBench, RouteLLM, and HybridLLM (the latter two re-implemented as binary routers between their fixed small/large pair) across nearly all reward settings, both ID and OOD. This directly validates the multi-model routing thesis that both RouteLLM and Hybrid-LLM flagged as unaddressed limitations.
- The **warm-up mechanism** (Eq. 10) for query cold-start — blending a new query's embedding with a weighted average of its k-nearest-neighbor training queries — is a lightweight, cheap fix for OOD generalization, and their ablation (Figure 6, 7) confirms it helps more in OOD (where cold-start is more severe) and more for NIRT-Router (the more complex, presumably more overfitting-prone model).
- Generalization to genuinely new LLMs (Claude 3.5 Haiku, held out from training) is tested directly (Table 4) — RouterBench collapses to near-random (AUC 0.50) on the new model, while both IRT variants retain meaningfully better-than-random accuracy (AUC 0.62, ACC ~0.67-0.68), though the paper is honest that this is still far from strong generalization.

**5. Limitations and how you might address them**

Authors' own stated limitations:
- Benchmark queries are short and don't reflect real-world query diversity — same limitation flagged by RouteLLM/Hybrid-LLM, seemingly endemic to the whole line of work; a natural fix is incorporating human-preference data (Chatbot Arena style) as RouteLLM does, blended with the IRT scoring framework.
- Router is "insufficiently sensitive to α" — the performance/cost trade-off doesn't respond as sharply as expected to the α/β weighting, which they attribute to the crude fixed-cost representation C(M_j) (just normalized pricing, not adjusted per query or per token count dynamically).
- No ordering constraint between LLM sizes and learned ability — they note you could enforce "larger models should have higher average ability" as an inductive bias/regularizer during training, which currently isn't enforced (it emerges empirically but isn't guaranteed).
- Weak generalization to genuinely new LLMs (ACC 0.67 on Claude 3.5 Haiku) — they suggest few-shot or similarity-based warm-up for the *model* side (parallel to their existing query-side warm-up), which currently doesn't exist.
- Top-1 routing accuracy is surprisingly low (2.72% ID, 2.15% OOD — Table 3), though they argue this is partly an artifact of many near-tied models rather than router failure; still, this is worth scrutinizing carefully if you cite their routing-accuracy numbers, since Top-1 in isolation looks weak even though Reward-based evaluation looks strong.

Additional gaps not explicitly named:
- The full-matrix training data requirement (every query × every model) is expensive to construct and doesn't scale gracefully as the candidate model pool grows — N models means N generations per query at training time. This is a real practical constraint your SAAR framework's tiered/gated design likely handles more gracefully (you don't need every agent to attempt every query).
- The 25-dimensional NIRT ability taxonomy is hand-designed via LLM-assisted clustering + human correction — a somewhat ad hoc, domain-specific ontology that would need to be redesigned for different task domains; not obviously a plug-and-play general framework.
- No entropy, uncertainty, or self-consistency signal anywhere in this framework — like RouteLLM, prediction is entirely query-embedding-driven, with zero use of generation-time signal from any candidate model. This is again a clean contrast point for your paper's entropy/paraphrase-uncertainty framing.

**6. Training procedure — inputs and outputs, step by step**

Step 1 — Collect 24,430 training queries across 8 ID datasets (70/30 train/test split per dataset, combined).

Step 2 — Generate a response from **all 20 candidate LLMs** for every training query, score each against ground truth with the task's native metric (accuracy/EM/F1/pass@1) → y_ij ∈ [0,1]. This produces the full 488,600-row training tensor.

Step 3 — Embed each query with BERT (bert-base-uncased, 768-dim) → e_qi. Embed each LLM using a text profile (release date, developer, description — Table 10) also through BERT → e_Mj.

Step 4 (MIRT-Router) — Learn linear projections: θ_Mj = W_θ·e_Mj (multidimensional ability, N=25 dims), a_i = W_a·e_qi (discrimination), b_i = W_b·e_qi (difficulty). Predicted performance via logistic function: P̂ = 1/(1+exp(−a_i^T θ_Mj + b_i)). Train via binary cross-entropy against y_ij, Adam optimizer, lr=0.002, batch 512, 1 A100 GPU.

Step 4 (NIRT-Router, alternative) — Additionally requires a **relevance vector** r_qi ∈ ℝ^25 per query: cluster training queries via UMAP+HDBSCAN, label each cluster's relevant ability dimensions using GPT-4o-Mini prompted on 5 sample questions per cluster (Appendix C.2 prompt shown), assign binary relevance per dimension. For test queries (no ground-truth relevance), approximate via 5-NN average in embedding space. Interaction: x_ij = r_qi ⊙ (θ_Mj − b_i) × a_i, then P̂ = σ(ϕ(W1·x_ij + b1)) — a small neural layer rather than pure logistic. Same BCE loss and training setup.

Step 5 — At inference: compute P̂(q_i, M_j) for the incoming query against all 20 models, combine with fixed cost C(M_j) via S(q_i, M_j) = α·P̂ − β·C, route to argmax.

Step 6 (cold-start) — Blend new query embedding with k=5-NN average from training set embeddings (λ weighting, Eq. 10) before feeding to the router, to combat train/test distribution mismatch.

**7. Metrics and evaluation**

Three primary metrics (following GraphRouter convention):
- **Performance** — average task accuracy/metric across test queries, using the routed model's response scored against ground truth.
- **Total Cost** — actual dollar cost summed over test queries based on real per-model API pricing and token counts (Eq. 11) — genuinely realistic cost accounting, more granular than RouteLLM's binary call-count metric.
- **Reward** — α·Performance − β·(normalized Total Cost) (Eq. 12), evaluated at three α/β settings (0.8/0.2, 0.5/0.5, 0.2/0.8) spanning performance-priority to cost-priority regimes.

Secondary diagnostic metrics for the new-LLM generalization test (Table 4): RMSE, MAE (regression framing of performance prediction), AUC, ACC (classification framing — presumably thresholding y_ij).

Router-quality diagnostics beyond the main metrics: Top-k routing accuracy (Table 3) — how often the router's chosen model is actually within the top-k best-performing models for that query, evaluated separately from the cost-weighted Reward metric.

Ablations: embedding model choice (BERT vs bge-m3 vs OpenAI vs Zhipu — Table 8, Appendix B), ability dimension N sensitivity (Figure 8), cold-start λ sensitivity (Table 9).

**8. Production/deployment use — dataset format, evaluation**

Deployment framing: router sits between the user and a live pool of 20 heterogeneous LLMs spanning real vendor pricing tiers (from $0.0137/M for GLM-4-Flash to $10/M output for GPT-4o) — this is the most realistic multi-vendor production simulation among the three papers, explicitly modeling actual API economics rather than a synthetic strong/weak pair.

Cost model: C(M_j) is a fixed, pre-computed linear mapping of published pricing into [0,1] (footnote example: GPT-4o's $10/M → C=1.0). They acknowledge this is a simplification — in practice cost varies by input/output token count per query (which their Total Cost *evaluation* metric does account for, Eq. 11, even though the *routing decision* itself uses the simpler fixed C(M_j)). This mismatch between the routing objective's cost term and the evaluation's cost accounting is worth noting critically — it may partly explain their own observed "insufficient sensitivity to α" limitation.

They test explicit OOD generalization to 4 held-out benchmark types and to one entirely unseen LLM (Claude 3.5 Haiku) not part of the original 20 — this is the closest thing among the three papers to a genuine "new model onboarding" production scenario (a new LLM release hitting the market, can the router route to it without retraining from scratch). Results are honestly reported as still weak (ACC 0.67), which is a useful, non-oversold data point.

No live-traffic or A/B deployment; still an offline benchmark evaluation, same limitation as RouteLLM and Hybrid-LLM.

---

For your related-work table, this is your cleanest N-way comparator — worth explicitly noting that IRT-Router's "interpretability" claim rests on a fundamentally different mechanism than your entropy/uncertainty signals: they get interpretability from a *psychometric latent-trait structure* (ability/difficulty decomposition with a monotonicity guarantee), while you'd be getting interpretability (if you claim it) from *direct, named routing signals* (query complexity, model entropy, paraphrase uncertainty) that don't require a learned latent space to inspect. That's a meaningfully different notion of "interpretable routing," and reviewers might ask you to distinguish it explicitly — IRT-Router's ability/difficulty parameters are learned and post-hoc interpreted, whereas your signals are defined interpretably from the start. Also worth flagging: all three papers (RouteLLM, Hybrid-LLM, IRT-Router) require either human preference data, repeated generation sampling, or full N-model generation matrices at training time — none of them do the "generate once at inference, extract entropy/uncertainty directly" thing your gates do, so your training-data story is genuinely the cheapest of the four once your pre-inference framing tension is resolved.


#MIXLLM

**Paper: MixLLM: Dynamic Routing in Mixed Large Language Models (Wang et al., ASU/NEC Labs, arXiv 2502.18482)**

**1. Core problem — technical and application-wise**

Technically: N-way routing over a *streaming* query sequence Q = {q_n}, with the routing decision jointly optimizing three objectives simultaneously — response quality, financial cost, and **latency/waiting time** — rather than the two-objective (quality vs. cost) framing every prior paper you've reviewed uses. This is the first paper in your set to treat latency as a first-class routing signal rather than an afterthought, and to frame routing as an online contextual-bandit problem rather than a static offline classification/regression problem.

Application-wise: real-world deployed serving systems where (a) queries arrive continuously as a stream (not a static batch), (b) hardware/throughput constraints mean routing too many queries to the same LLM creates a bottleneck (queueing delay), (c) the LLM candidate pool changes over time (new models added, old ones retired) without full system retraining, and (d) the system needs to keep improving from live user feedback post-deployment. This is explicitly the most "production systems" framed paper of the four — it's modeling a live routing service, not just an offline benchmark.

**2. Training dataset format — open source?**

Base dataset: RouterBench (Hu et al. 2024a), an existing open benchmark — 36,497 queries from 8 NLP datasets (Chinese + English), each already answered by 11 different LLMs with recorded quality and cost metrics. MixLLM's own contribution here is an **extension**: they add Llama 3.1 8B and 70B as two more candidate models (running these themselves), and add prompt/response length fields to every record — this extended dataset is a genuine, reusable artifact.

Format per training tuple, conceptually: (query, tags, embedding) → per-LLM (quality label, cost/length label). This is structurally similar to IRT-Router's full-matrix design (every query × every candidate LLM has a label) but adds a length/latency dimension IRT-Router doesn't have.

Split: 80% train / 20% test, standard.

Additional data source: **InsTag** (Lu et al. 2023), a Llama-2-13B-based instruction-tagging model, used to generate fine-grained tags for every query, which are then manually clustered into 20 coarse domains (e.g., "Computer Science," "Legal"). This tag/domain layer is the paper's distinguishing preprocessing step and is derived, not part of RouterBench itself.

Openness: RouterBench base data is open; their Llama 3.1 extension and tag/domain labels appear to be a contribution of this paper (not explicitly stated as released, unlike IRT-Router's GitHub repo) — worth verifying if you want to reuse their exact extended dataset.

**3. Central innovative idea**

Three ideas bundled together, each targeting one of the three gaps the authors identify in prior work (RouteLLM, HybridLLM, RouterBench, FORC, MetaLLM):

(a) **Tag-enhanced query embeddings via unsupervised contrastive-style fine-tuning.** Rather than using raw BERT embeddings, they fine-tune BERT using an unsupervised loss that pulls a query's embedding toward its InsTag-derived domain cluster centroid (L_intra, Eq. 2) while pushing different domain centroids apart (L_inter, Eq. 3). This is motivated by an empirical observation (Figure 3) that GPT-4's error rate correlates strongly with query domain (worse on Legal/Math) — domain signal genuinely predicts routing-relevant quality variance, so baking it into the embedding should help downstream quality/cost regressors.

(b) **A LinUCB-style contextual bandit meta-decision-maker** (Eq. 7-11) that combines: a quality-cost tradeoff term (s_trade, parameterized by a willingness-to-pay λ, structurally identical to RouteLLM's/IRT-Router's α-weighted score), an **uncertainty bonus** (s_unc = e^T·A⁻¹·e, the classic LinUCB confidence-bound term — this is genuinely borrowed from the contextual bandit literature, not ad hoc), and a **latency penalty** (s_pen, an exponential penalty on a model's current waiting time relative to a tolerance threshold τ). This is the paper's most technically distinct contribution: no other paper in your set incorporates a live, dynamically-updating waiting-time term into the routing score itself.

(c) **Per-LLM independent prediction models + continual (online) learning.** Each candidate LLM gets its own quality regressor f_rq_l and length regressor f_rl_l (lightweight — Random Forest, MLP, KNN, not deep networks). Because these are independent per-model, adding a new LLM only requires training one new small regressor, not retraining the whole system — this directly targets the "varying candidate set" limitation that HybridLLM explicitly flagged as future work. Post-deployment, the system also updates from live (binary or refined) user feedback via a policy-gradient-trained "dynamic feedback score" layer (Eq. 15-22), which is the paper's answer to "continual learning in deployed systems."

**4. What's novel and why it works**

- The uncertainty bonus (s_unc, Eq. 9) is a real methodological import from the bandit literature (Li et al. 2010, LinUCB) — it explicitly rewards exploring under-sampled (query-embedding, LLM) regions, which none of RouteLLM/HybridLLM/IRT-Router do (they're all static supervised-learning routers, trained once offline, with no explicit exploration mechanism). This matters because a purely greedy predictive router can get stuck exploiting a biased early estimate and never correct itself — the bandit framing is a principled fix.
- The latency penalty is empirically validated to matter: Figure 4 vs Figure 5 (with vs. without latency constraint) shows several baselines (especially AutoMix, which cascades through multiple LLMs per query) degrade sharply once a hard waiting-time cap is imposed, while MixLLM stays stable because it explicitly penalizes routing into a congested model. This is a genuinely different failure mode than anything RouteLLM/HybridLLM/IRT-Router are evaluated against — those papers never model concurrent query load at all.
- Continual/online training with binary user feedback (satisfied/not satisfied) is handled via policy gradient over a small "appropriateness" MLP (Eq. 19-22), with an inverse-variance confidence weighting κ (Eq. 17) so noisy/unstable feedback signals get down-weighted automatically. Table 1 shows a modest but consistent improvement from online feedback across three offline:online data-split ratios, and — usefully — shows refined feedback (actual quality/cost) beats binary feedback (satisfied/not), but binary still helps, which is a realistic finding since real deployments often only have access to coarse feedback.
- Tag-enhanced embedding's benefit is shown to be real but modest and diminishing with budget (Table 2: 5.72% improvement at low cost, shrinking to 0.79% at high cost) — an honest empirical characterization rather than an overclaimed universal win.

**5. Limitations and how you might address them architecturally**

Authors' own stated limitations (very candid list):
- Requires refined feedback (quality + cost) for training, which may not be available in real deployments — they suggest training-free scaling-law-based approaches (Ruan et al. 2024) as an alternative, unexplored in this paper.
- OOD generalization is a real, measured weakness — Table 3 shows a genuine 5.44% quality drop when test domains are entirely unseen during training (mitigated to 3.35% with online training, but not eliminated). Notably, they frame this honestly as "a novel routing task" for the community rather than claiming to have solved it.
- No mechanism for selecting a single definitive answer when multiple LLMs are queried (Section 4.8) — their "Top-k" selection-policy study shows real quality gains from querying multiple LLMs and taking the best, but they explicitly did not build a reviewer/aggregator to pick among them, leaving this as future work. This is essentially LLM-Blender/ensemble territory that they deliberately didn't take on.
- Latency model is simplified — assumes ideal hardware conditions for open-source models (sufficient memory, stable network), doesn't model realistic queueing/contention dynamics beyond the exponential penalty heuristic. A more realistic queueing-theoretic model (e.g., M/M/1-style wait time estimation per backend) would be a natural extension.
- No hierarchical routing explored — they explicitly suggest domain-first-then-model routing (route to domain, then select LLM within domain) as unexplored future work. This is interesting for you specifically: your SAAR/PF-AAR framework's tiered gate structure is essentially a version of exactly this hierarchical routing idea they flag as an open problem.
- Real-world deployment validation is entirely absent — like RouteLLM, HybridLLM, and IRT-Router, this is all offline-simulated (query streams synthetically paced at "100 queries per 10 seconds," latency simulated from public API/hardware statistics rather than measured live).

Concrete architectural extension worth flagging: the uncertainty bonus (s_unc) uses a simple LinUCB-style linear confidence bound over BERT embeddings — this is a fairly crude uncertainty estimate (doesn't use the LLM's own generation-time signal at all, e.g., token-level entropy or self-consistency across samples). This is again the same gap as RouteLLM/HybridLLM/IRT-Router: all uncertainty here is about the *router's own prediction confidence*, never about the *candidate LLM's own uncertainty in generating* — which is precisely the paraphrase-based and entropy-based uncertainty signal your SAAR framework introduces. Worth explicitly naming in your related work: MixLLM's "uncertainty" is bandit-arm exploration uncertainty (how much data have we gathered about this query-region/LLM-pair), not generation uncertainty (how confident is the LLM about its own answer) — these are different types of uncertainty entirely, and conflating them in your citation would be a mistake.

**6. Training procedure — inputs and outputs, step by step**

Step 1 — Tag generation: run InsTag (Llama-2-13B backbone) over every training query to produce fine-grained tags; manually cluster tags into 20 coarse domains D.

Step 2 — Encoder fine-tuning (unsupervised): fine-tune BERT with the combined intra-domain/inter-domain contrastive loss (Eq. 1-3) so that query embeddings cluster by domain. Output: a domain-aware query embedding e_n for each query.

Step 3 — Per-LLM quality/length regressor training: for each candidate LLM l, train independent lightweight models — Random Forest for quality prediction (f_rq_l: e_n → p̂_n,l), and a mix of MLP/RF/KNN for response-length prediction (f_rl_l: e_n → len̂_res_n,l), depending on which performed best per LLM. These are trained on the offline portion of the data using actual observed (quality, length) pairs from running that LLM on training queries.

Step 4 — Cost computation: combine known prompt length (deterministic, from the query itself) with predicted response length and each LLM's published per-token pricing (Eq. 5) to get ĉ_n,l.

Step 5 — Meta decision score assembly (inference-time, per incoming query): compute s_trade (quality-cost tradeoff via λ, Eq. 8), s_unc (bandit uncertainty bonus via inverse covariance matrix A_l⁻¹, Eq. 9, updated incrementally as more queries are observed per LLM, Eq. 14), and s_pen (latency penalty based on current queue/waiting-time state per LLM, Eq. 10). Combine via weighted sum (Eq. 7, α=0.01, β=0.1 in their config) and route to argmax.

Step 6 — Offline training (pre-deployment): update θ_rq_l and θ_rl_l via gradient descent against real observed labels (η1=η2=1, "reflecting simple ML algorithms" — i.e., these are closed-form/simple-fit models, not deep nets), and incrementally accumulate the uncertainty matrices A_l (Eq. 14).

Step 7 — Online training (post-deployment): after each live query, update the same predictive models with the newly observed single data point (refined feedback), plus train a shared 3-layer MLP "dynamic feedback score" network f_df via policy gradient (Eq. 19-22, η3=0.001) using binary user satisfaction signal r_n as reward — this is the RL-flavored component layered on top of the supervised regressors.

**7. Metrics and evaluation per stage**

- **Total Quality** — sum of per-query response quality scores (0-1 scale) across the test stream; any query exceeding the max tolerable waiting time τ is scored 0 (a hard latency-violation penalty baked directly into the eval, not just the training objective).
- **Total Cost** — sum of actual dollar costs across the test stream.
- Evaluated across a **sweep of λ** (willingness-to-pay, 10⁻⁶ to 10⁶) to trace out the full quality-cost frontier (Figure 4), analogous to RouteLLM's APGR-style curve but plotted as raw quality-vs-cost rather than a normalized gap-recovery metric.
- **Oracle** baseline — theoretical best-possible assignment (requires running every LLM on every query and picking the cheapest one meeting a quality threshold) — used as an upper-bound reference point on the frontier plot, a nice honest ceiling that RouteLLM/HybridLLM/IRT-Router don't explicitly compute.
- Ablation-specific metrics: Table 2 (tag-enhanced vs. general embedding, quality at fixed cost-level bands), Table 1 (continual training quality-percentage under different offline:online split ratios), Table 3 (OOD quality drop with/without online training), Figure 7 (Top-k selection policy quality-cost curves).
- Baselines span both non-predictive (AutoMix/cascading) and predictive (RouteLLM, Zooter, RouterBench, FORC, OptLLM, MetaLLM) — the broadest baseline comparison set of the four papers you've reviewed, and notably includes a bandit-based prior work (MetaLLM) as a direct methodological comparator.

**8. Production/deployment use — dataset format, evaluation**

This is the paper most explicitly designed around deployment realism:
- Simulated query stream paced at 100 queries per 10 seconds, with waiting time updated every 10 seconds — an actual discrete-event simulation of concurrent load, not a static offline batch evaluation. This is meaningfully different from RouteLLM/HybridLLM/IRT-Router, which all evaluate on a fixed test set with no notion of concurrent queries competing for the same backend.
- Latency parameters (initial startup time, token generation speed per LLM) are sourced from a public real-world pricing/performance tracking site (artificialanalysis.ai) for both API-based and self-hosted open-source models — a genuine attempt to ground the simulation in real infra numbers rather than synthetic assumptions.
- Explicit "adding a new LLM" deployment test (Section 4.6): they literally add Llama 3.1 8B/70B post-hoc to the candidate pool and show the system incorporates them without full retraining (only the new models' own regressors need training) — this is the most concrete "hot-swap a model into a live router" demonstration among all four papers, directly answering HybridLLM's and RouteLLM's stated "N-model / evolving candidate set" limitation.
- Continual/online learning loop (Section 3.6, 4.3) is explicitly designed as the production feedback mechanism — offline training happens pre-deployment, online training runs continuously afterward using whatever feedback the live system can access (refined or just binary satisfaction).
- Deployment cost/format for the router itself: quality/length regressors are explicitly kept "lightweight" (MLP under 2MB) for fast inference/update — an intentional design choice mirroring RouteLLM's and Hybrid-LLM's emphasis on router-overhead being negligible relative to LLM generation cost, though this paper doesn't report a direct latency-overhead number for the router itself (a gap relative to Hybrid-LLM's Table 2/RouteLLM's Table 7, which both explicitly benchmark router latency).

---

For your ACL related-work section, MixLLM is your best precedent for the **latency/throughput dimension** and for **contextual-bandit-style exploration**, both of which are absent from RouteLLM, Hybrid-LLM, and IRT-Router. Two framing points worth making explicit in your paper:

First, on the "pre-inference" tension you're navigating: MixLLM's uncertainty term (s_unc) is computed purely from the query embedding and an accumulated per-LLM covariance matrix — it never touches the LLM's actual output distribution, so it's fully pre-inference, same as RouteLLM/HybridLLM/IRT-Router. This makes your paraphrase-based/entropy-based uncertainty signal (which does require generation) the clear odd-one-out across every paper in this literature — worth stating plainly that no prior routing paper actually measures the candidate model's own generation-time uncertainty; they all either skip uncertainty entirely, or model *router* uncertainty (bandit-style, over query-embedding space) rather than *generator* uncertainty (over the model's own output distribution). That's a genuinely clean, citable gap for your introduction/related-work framing.

Second, MixLLM's explicit call-out for "hierarchical routing" as unexplored future work (domain-first, then model-within-domain) is close to what your SAAR gates already do — worth citing this paper specifically when motivating your tiered/gated architecture as addressing a gap the routing literature itself has identified as open.



# ICL ROUTER
**Paper: ICL-Router: In-Context Learned Model Representations for LLM Routing (Wang et al., Shanghai AI Lab, arXiv 2510.09719)**

**1. Core problem — technical and application-wise**

Technically: N-way routing (8+ candidate LLMs) with a specific focus on the **scalability-of-model-addition problem** that every prior paper you've reviewed (RouteLLM, HybridLLM, IRT-Router, MixLLM) has flagged but not fully solved. The core technical claim is: represent each candidate LLM's capability not as a learned parameter vector baked into the router's weights (which requires retraining when the model pool changes), but as a set of **in-context vectors** — compact vector encodings of (query, correct/incorrect) pairs — that get fed to the router at inference time as context, the same way few-shot exemplars would be fed to an LLM doing in-context learning.

Application-wise: this directly targets a real operational pain point — LLM vendors release new models "almost daily" (their framing), and every prior router (RouteLLM, HybridLLM, IRT-Router's own ability vectors, MixLLM's per-LLM regressors) either can't add a new model without retraining, or degrades badly when tested on an unseen model (recall IRT-Router's own Table 4 showing weak ACC~0.67 generalization to Claude 3.5 Haiku). ICL-Router's answer: profiling a new model needs only running it on a small existing query set (500 queries) — no router retraining at all.

**2. Training dataset format — open source?**

Two distinct training stages with different data needs:

Stage 1 (Query Reconstruction Training) data: just a set of raw queries Q = {q_n} — no labels needed at all, purely self-supervised (query → embed → project → reconstruct the same query autoregressively). Any query corpus works; they use their training benchmark queries directly for this. Fully open — no external annotation needed.

Stage 2 (ICL Model Routing Training) data: a **capability profile** per model, P_t = {(v_k, c_k)}, where v_k is a query's vector representation and c_k ∈ {Yes, No} is whether model M_t answered query q_k correctly. This requires actually running all 8 candidate LLMs on a curated 500-query "challenging" subset (deliberately chosen so between 1-4 of 8 models get each query right — Appendix A.3, explicitly excluding all-correct/all-wrong queries as uninformative). This is a full-matrix generation requirement like IRT-Router's, but on a much smaller curated slice (500 queries × 8 models = 4,000 generations) rather than the full training set.

Source benchmarks: 10 open, standard academic datasets (OlympiadBench, BBH, LogicBench, MMLUPro, MBPP as in-distribution; AIME, KORBench, MMLU-CF, AGIEval, HumanEval as held-out OOD) — all public, no proprietary data. Code released at github.com/lalalamdbf/ICL-Router.

**3. Central innovative idea**

Decouple **model capability representation** from **router training** by borrowing the in-context vector paradigm (Zhuang et al. 2024b Vector-ICL; Liu et al. 2024a) — instead of literally putting hundreds of (query, correct/incorrect) exemplars as text tokens in the router's context window (too long, expensive), compress each exemplar into a dense vector via the same embedding model + learned projector used for the query itself, and feed these compact vectors as the router's "in-context" evidence about a model's capability. The router (an actual LLM, Qwen2.5-7B-Instruct) then reasons over: [new query vector] + [candidate model's set of (query vector, correct/incorrect) pairs] → probability the candidate model answers correctly.

This is architecturally distinct from every other paper you've reviewed: RouteLLM/HybridLLM/IRT-Router/MixLLM all bake model identity into the *router's trained weights* (a learned embedding, an ability vector, a per-model regressor head) — meaning the router's parameters are tied to the specific model pool it was trained on. ICL-Router instead treats model capability as **external, swappable context** the router conditions on at inference time, analogous to how a person could evaluate a new employee's résumé (capability profile) without needing to "retrain" their own judgment process.

**4. What's novel and why it works**

- The two-stage training separates concerns cleanly: Stage 1 (query reconstruction) is purely about *aligning the projector's output space with the router LLM's semantic space* — forcing the projected vector v_n to contain enough information that the router can reconstruct the original query token-by-token. This is a clever self-supervised proxy objective: if the router can regenerate the query from the vector alone, the vector has captured the query's semantics faithfully, which is a prerequisite for the router being able to reason over *other* models' performance vectors later. Table 4's ablation (removing QRT costs 2.29% ID / 2.41% OOD) confirms this stage matters substantively, not just as a formality.
- The empirical scalability results (Figures 2-3) are the paper's strongest evidence: as 5 new, unseen-at-training-time LLMs are incrementally added to the candidate pool (Falcon-H1, Gemma3-12B, DeepSeek-R1-Llama-8B, OpenThinker3-7B, AceReason-Nemotron), ICL-Router's accuracy climbs monotonically (76.3%→79.9% ID, 66.4%→69.9% OOD) with **no retraining** — while MODEL-SAT (the next-best scalable baseline) shows inconsistent, sometimes-declining gains. This is a genuinely different scaling behavior than IRT-Router's own new-model test, which showed only marginal, honestly-reported generalization (ACC 0.67).
- Beats "Max Expert" (the strongest single model per dataset) by 3.11% ID average and clearly on several OOD tasks — meaning the router isn't just learning "which dataset am I in, pick the best model for that dataset" (task-level routing) but genuinely differentiating at the individual-query level, since Max Expert already captures the dataset-level optimum.
- The "challenging query" curation strategy (Appendix A.3: only keep queries where 1-4 of 8 models got it right, discard unanimous cases) is a sensible, cheap way to maximize training signal density without needing more raw data — directly analogous to Hybrid-LLM's t*-relaxation trick for fixing label imbalance, but solved architecturally at the data-curation stage rather than via a loss-relaxation parameter.

**5. Limitations and how you might address them**

Authors' own stated limitations (Appendix A.5, refreshingly explicit):
- Model pool restricted to small-parameter LLMs (7-12B range) due to compute/data-collection cost — they haven't validated whether the in-context-vector representation scales to routing among much larger models (70B+, frontier closed models), which is a real open question for your related-work framing, since RouteLLM/MixLLM do test with GPT-4-class models in the pool.
- Benchmarks are general-knowledge/reasoning evaluation sets, not chat/instruction-following-quality benchmarks — so this router hasn't been validated on the kind of open-ended conversational quality judgments that RouteLLM's Chatbot-Arena-based approach specifically targets. This is a genuine domain-coverage gap worth flagging if you're comparing routing methods across "reasoning-task routing" vs. "chat-quality routing" — they're not the same problem and this paper only addresses the former.

Additional limitations not explicitly named:
- The router itself is a 7B LLM performing an actual forward pass with in-context vectors per candidate model — this is a heavier inference-time cost than RouteLLM's lightweight BERT/matrix-factorization routers, IRT-Router's small MIRT projection, or MixLLM's Random-Forest regressors. Their own cost analysis (Appendix A.4) argues this is fine because the router only generates ~8 output tokens (one Yes/No decision token per candidate model, presumably), but the *input* context includes potentially hundreds of in-context vectors per candidate — the paper doesn't report router latency numbers the way Hybrid-LLM (Table 2) or RouteLLM (Table 7) do, so the actual wall-clock routing overhead relative to those lighter architectures isn't directly comparable from what's given.
- In-context exemplar quantity shows a clear diminishing-returns / degradation pattern (Figures 4-5: peaks at 500 exemplars, drops at 1000) — this "too much context hurts" phenomenon (consistent with many-shot ICL literature they cite) means the capability-profile size itself is a hyperparameter requiring tuning per deployment, not a free scaling knob.
- Like every other paper in this line, all uncertainty/capability signal is derived from **binary correct/incorrect outcomes on a static profiling set** — again no generation-time entropy or self-consistency signal from the candidate model itself is used. The "capability profile" is behavioral/outcome-based (did it get this right), not distributional (how confident was it) — same fundamental gap as RouteLLM, HybridLLM, IRT-Router, and MixLLM relative to your entropy/paraphrase-uncertainty approach.

A concrete architectural extension: the capability profile P_t is currently a static, fixed-size set of (query, correct/incorrect) pairs computed once when a model is added. An adaptive/prioritized profile — e.g., preferentially including profiling queries most similar to the live query distribution, or updating the profile online as MixLLM does with its continual learning loop — would combine ICL-Router's scalability story with MixLLM's continual-adaptation story; currently these are two orthogonal contributions across two different papers.

**6. Training procedure — inputs and outputs, step by step**

Stage 1 — Query Reconstruction Training:
- Step 1: encode every training query q_n via a frozen embedding model (Qwen3-Embedding-8B, itself never trained/updated) → e_n = f_emb(q_n).
- Step 2: project e_n through a trainable 2-layer MLP → v_n = f_proj(e_n), mapping embedding-space dimension to router-space dimension.
- Step 3: feed v_n to the router (Qwen2.5-7B-Instruct) and train it to autoregressively reconstruct the original query token sequence, minimizing negative log-likelihood of each token conditioned on v_n and prior tokens (Eq. 3).
- Step 4: train projector alone for 1 epoch (lr 2e-5), then jointly train projector + router for 2 epochs (lr 5e-6) — two-phase warmup within Stage 1 itself.

Stage 2 — ICL Model Routing Training:
- Step 1: construct the 500-query challenging subset (125 queries each at difficulty levels 1/2/3/4-of-8-models-correct).
- Step 2: for each of the 8 candidate LLMs, run it on all 500 queries, record correct/incorrect labels c_k, and encode each query via the (now-trained) embedding model + projector to get its vector v_k → assemble capability profile P_t = {(v_k, c_k)} per model.
- Step 3: train the router (jointly with the projector, continuing to fine-tune both) to predict P(y_t,n | P_t, q_n) — whether model t will answer a new training query q_n correctly, conditioned on that model's full capability profile — via cross-entropy loss (Eq. 5), 5 epochs, projector lr 1e-5, router lr 2e-6, batch size 32, repeated across 3 random seeds.

Step 4 (Inference) — for a new query q′: compute its vector, feed it alongside each candidate model's stored capability profile to the router, get P(Yes | M_t, q′) per candidate, route to argmax_t (Eq. 6). Evaluation samples each routed LLM's response 10 times (temperature 0.3, top-p 1.0) and reports average accuracy — a variance-reduction measure not present in the other three papers' evaluation protocols.

Step 5 (New model onboarding) — evaluate the new model M_{T+1} on the same fixed 500-query profiling set, encode via the frozen embedding model + trained projector, assemble its capability profile, and immediately route to it — no router weight updates at all.

**7. Metrics and evaluation**

Primary metric: task-appropriate accuracy or Pass@1 per benchmark (Table 1 in Appendix: accuracy for MC/QA tasks, Pass@1 for code tasks MBPP/HumanEval), averaged across benchmarks within each ID/OOD split.

Baselines spanning training-free and trained: Random Router, LLM Router (prompt-based, training-free — an LLM literally reads natural-language model profiles and picks), Max Expert (oracle-ish upper reference using best single model per dataset, not a true oracle since it's dataset-level not query-level), RouterDC, EmbedLLM, MODEL-SAT — this is a well-chosen baseline set since RouterDC/EmbedLLM represent the "fixed pool, must retrain" camp and MODEL-SAT represents the prior best attempt at scalable model-addition (via hand-crafted MMLU-based capability instructions).

Scalability-specific evaluation (Figures 2-3): accuracy trajectory as 5 new models are incrementally added, compared only against MODEL-SAT (the only other baseline claiming scalability without retraining) — RouterDC/EmbedLLM are excluded from this comparison since they structurally can't do this without retraining, which itself is a meaningful evaluation design choice worth noting (they're not just omitted, they're *inapplicable* to this test).

Ablations: embedding model choice (Table 3, showing a clean monotonic trend of stronger embedding models → better routing, similar in spirit to MixLLM's finding that embedding quality matters), in-context exemplar quantity (Figures 4-5, inverted-U pattern), query reconstruction training presence/absence (Table 4).

**8. Production/deployment use — dataset format, evaluation**

Deployment framing centers specifically on the **new-model-onboarding workflow**: profile a new LLM on a small (500-query) fixed evaluation set, encode via existing frozen embedding model + trained projector, and it's immediately routable — the router LLM's own weights never change. This is presented as the paper's core practical contribution, addressing what they characterize as a "prohibitive" cost in prior routing systems (repeating large-scale evaluation and retraining every time a new model is released).

Cost accounting (Appendix A.4): they note the router itself, despite being a 7B-parameter LLM, only needs to generate a small number of output tokens per routing decision (their example: 8 tokens for 8 candidate models — presumably one Yes/No-flavored token per candidate), which is trivial relative to the hundreds/thousands of tokens the actually-routed LLM will generate. This is a reasonable but less rigorously benchmarked cost argument than Hybrid-LLM's or RouteLLM's explicit latency tables — no wall-clock numbers are given here.

They do not report a live-traffic deployment or dollar-cost accounting the way MixLLM does (no pricing tables, no cost-vs-quality frontier plot) — this paper's evaluation is entirely accuracy-focused, notably omitting the cost dimension that RouteLLM, HybridLLM, IRT-Router, and MixLLM all treat as a co-equal objective. This is a real difference in scope worth flagging: ICL-Router optimizes purely for **routing accuracy / correctness-maximization**, not cost-quality tradeoff — closer to a "Max Expert but query-level" framing than a "quality-cost Pareto frontier" framing like the other three papers.

---

For your ACL related-work section, ICL-Router is your sharpest precedent for the **scalable model-addition problem** specifically — more so than IRT-Router (which tests generalization to a new model but still requires the router itself to have been trained with a fixed 20-model pool structure) and more explicitly than MixLLM (which handles new-model addition via independent per-model regressors, a different and arguably simpler mechanism worth contrasting directly). Two points worth making explicit in your framing:

First, ICL-Router's capability profiles are still purely outcome-based (correct/incorrect on a held-out query set), computed once, offline, before deployment — this is the same "profile via generation, then route without generation" pattern as every other paper in this literature. Your entropy/paraphrase-uncertainty signals are fundamentally different in requiring a forward pass at *query time*, not just at *profiling time* — worth stating precisely, since "requires generation" can otherwise get conflated across these very different points in the pipeline (profiling-time generation vs. inference-time generation).

Second, note that this paper drops the cost/latency dimension entirely, while your SAAR framework (like RouteLLM/HybridLLM/IRT-Router/MixLLM) is explicitly cost-aware — this is a genuine axis of comparison in your related-work table: routing methods split into cost-aware (most of the literature) vs. accuracy-maximizing-only (ICL-Router, LLM-Blender-style ensembling), and you should probably state where SAAR sits on that axis explicitly, since reviewers familiar with this ICL-Router paper will likely ask.


### Leveraging Uncertainty Estimation for Efficient LLM Routing

 (Zhang, Mehradfar et al., "Leveraging Uncertainty Estimation for Efficient LLM Routing," USC/Amazon, Feb 2025).

**1. Core problem — technical and application**

Technical problem: existing predictive LLM routers (RouteLLM, TO-Router) train on either (a) noisy, subjective human-preference data (Chatbot Arena) or (b) binary accuracy labels from benchmarks. Both are flawed — preference data doesn't scale (Fig 1 shows CPT performance is non-monotonic and even degrades as Arena training size grows from 10K→50K), and binary accuracy throws away information about *how confident* a model was, so two models that both get an answer "correct" are treated identically even if one was barely right and the other was decisively right.

Application problem: edge-cloud LLM deployment. A small model runs on-device (cheap, low latency), and a large cloud model (GPT-4-class) is called only when needed. The router's job is a binary/preemptive decision per query — before generation is fully useful — of whether to send the query to the weak local model or pay for the strong cloud model. Goal: minimize cloud calls (cost) while maintaining response quality.

**2. Training dataset format — open source or not**

Format per training row: `{id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie}` — the last three are binary one-hot columns indicating which model won (or tie).

Source datasets used to build this: Natural QA, TriviaQA, PopQA (factual QA), and MAWPS (math word problems). They sampled 3,610 each from the three QA sets + 1,418 from MAWPS = 12,247 samples total, deliberately matched to the size of the Chatbot Arena sample RouteLLM uses, for a fair comparison.

Open-source status: yes — the constructed preference dataset (with SE-derived winner labels) is published on HuggingFace at `AsalMehradfar/uncertainty_0.1`. All the underlying QA/math datasets are also public benchmarks used "under research license."

**3. Central innovative idea**

Replace human-preference labels (RouteLLM) or benchmark-accuracy labels (TO-Router) with **Semantic Entropy (SE)** — an uncertainty score — as the signal used to construct preference/winner labels for router training.

Mechanism:
- For a given prompt, sample multiple generations from a model, cluster them by semantic equivalence (bidirectional entailment via a fine-tuned DeBERTa-large classifier — two generations cluster together only if each entails the other).
- Compute SE(x) = −(1/|C|) Σ log p(Cᵢ|x) over clusters — this is entropy over meaning-clusters rather than raw token sequences, so paraphrases don't inflate uncertainty.
- Compare SEstrong (GPT-4) vs SEweak (Mixtral-8x7B) per prompt via a normalized difference δSE(x) = |SEstrong − SEweak| / SEstrong.
- If δSE(x) > τ, the model with lower SE wins (i.e., the more *confident* model is the "preferred" model for that query) — else it's a tie.
- This SE-derived winner label replaces the RouteLLM human-preference winner label / TO-Router accuracy label in the exact same downstream training pipeline (same four lightweight router architectures: SW-ranking, MF, MLP, kNN).

So the innovation is entirely in **label construction**, not in the router architecture itself — they deliberately reuse RouteLLM's/TO-Router's exact predictive models so the only variable is the training signal.

**4. What's novel, and why it works**

Novel elements:
- Using semantic entropy (originally a hallucination-detection technique from Kuhn et al. 2023) repurposed as a *router training signal*, not just an inference-time uncertainty flag.
- Using LLM-as-a-Judge (GPT-o1) to evaluate response *quality* of the routing outcome, not just binary correctness — this is presented as "the first systematic assessment of response quality across routing strategies."
- Empirically demonstrating that preference-data scale doesn't help (Fig 1) — used as motivation, not as the contribution itself, but it's a notable negative result.

Why it plausibly works: SE is a continuous, per-query signal computed directly from the model's own generation distribution — no external human labeling, no reliance on Arena-style crowd data with its known sparsity/imbalance across model pairs. It also encodes *gradation* — the normalized δSE with threshold τ lets near-tie cases be explicitly modeled as ties rather than forced into a binary win/loss, which is something benchmark-accuracy labels (TO-Router) can't represent at all (accuracy is 0/1, there's no "tie" or degree-of-confidence concept in a per-sample accuracy label).

**5. Limitations and possible architectural/training fixes**

Author-stated limitations (Section 5):
- Text-only; no multimodal (image-text) query routing evaluated.
- No analysis of computational overhead of different router architectures (SW vs MF vs MLP vs kNN) — no latency/scalability comparison, which matters a lot for the "edge-cloud" framing they lead with.

Limitations I'd flag beyond what they state, relevant to your SAAR/PF-AAR work:

- **Pre-inference framing tension** (same issue you're wrestling with in your own ACL paper): SE requires multiple full generations per model per query to build the entailment clusters. That is *not* a preemptive/pre-inference signal — it's post-hoc, computed only after paying the generation cost on both models for the training set. The paper is honest that this is a training-time construction pipeline, but it doesn't clearly separate "how the router decides at inference-time" (presumably cheap — a classifier over query embeddings) from "how labels were derived" (expensive — full multi-sample generation + entailment clustering + clustering-based entropy). This is worth flagging explicitly in your related-work discussion since it's exactly the tension you're resolving in your own routing signals (entropy/uncertainty needing full generation passes vs a router that's supposed to act before generation).
- **τ is a fixed global hyperparameter** with no reported sensitivity analysis — how δSE ties are decided isn't validated empirically (no ablation over τ shown in the tables).
- **Two-model routing only** (GPT-4 vs Mixtral-8x7B) — doesn't extend to multi-agent/multi-tier routing (which is your setting). Extending SE-based labeling to N models raises combinatorial cost (need pairwise δSE, or some N-way generalization of Eq. 3–4) — an obvious extension you could position your work against.
- **DeBERTa entailment classifier is a fixed off-the-shelf-ish component** with no discussion of its own error propagation into cluster assignment, which directly affects SE — no error analysis of clustering quality.
- **Cost of SE computation itself isn't counted** in their reported "system cost" (Section 3.2) — they report inference-time API cost but not the offline SE-computation cost (multiple generations × two models × entailment classifier calls) needed to build the training set. This is a good comparison point for cost accounting rigor if you cite this paper.

Possible fixes/extensions (architecture/training side): distill the multi-sample SE computation into a single-pass proxy (e.g., predictive/token entropy or a lightweight uncertainty head trained to approximate SE) so that at inference time you don't need multiple generations — this is essentially the direction of your own PF-AAR / SAAR pre-inference framing, and a natural citation-and-contrast point: "unlike Zhang et al. 2025, whose uncertainty signal requires N full generations per model at label-construction time, our approach uses [your pre-inference proxy]..."

**6. Training pipeline, step by step — inputs/outputs at each stage**

Phase 1 — Data prep + clustering:
- Input: prompt x from Natural QA / TriviaQA / PopQA / MAWPS.
- For each model (GPT-4, Mixtral), generate multiple responses to x.
- Cluster responses via bidirectional DeBERTa-large entailment (pairwise forward+backward entailment against a cluster representative).
- Output: cluster assignments per model per prompt, and cluster probabilities p(c|x) (Eq. 1).

Phase 2 — Preference/label construction:
- Input: SE(x) computed per model per prompt (Eq. 2) from Phase 1 clusters.
- Compute δSE(x) (Eq. 3) between GPT-4 and Mixtral.
- Apply threshold τ (Eq. 4) → output: winner label (model_a wins / model_b wins / tie) per prompt, formatted as the one-hot triple.
- Encode prompt+responses into embeddings via a pretrained embedding model.
- Output: the full dataset row `{id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie}`, published to HuggingFace.

Phase 3 — Router training:
- Input: vectorized (embedded) prompts + winner labels from Phase 2.
- Train four separate lightweight router models: SW-ranking (Bradley-Terry, no gradient training, inference-time optimization), MF (bilinear score function s(M,q), BT-based win probability), MLP (2-layer, softmax output), kNN (no training — nearest-neighbor lookup by embedding cosine similarity, inherits label from nearest training point).
- Output: P(win_Mstrong | q) — a probability that the strong model should be used for query q, used at deployment to threshold the routing decision.

**7. Metrics and evaluation per stage**

- **System cost**: Call-Performance Threshold, CPT(x%) — minimum fraction of queries that must go to the strong model to achieve x% accuracy improvement over the weak model's baseline. Reported at CPT(50%) and CPT(80%) on MT-Bench, GSM8K, MMLU (Table 1). Lower is better. Also reported: actual OpenAI API dollar cost on MT-Bench at CPT(80%) — Random $4.06, TO-Router $3.88, RouteLLM $4.04, Confidence-Driven $3.74.
- **Response quality**: LLM-as-a-Judge (GPT-o1) picks the best of three routed responses vs. ground truth; Score(i) = (Sᵢ/T) × 100 where Sᵢ = number of times router i's response was judged best, T = total queries. Reported only on GSM8K (Table 2), compared across TO-Router / RouteLLM / Confidence Router at matched CPT(50%) and CPT(80%) operating points.
- No separate metric is reported for the clustering/entailment step itself (Phase 1) — no precision/recall on entailment classification, which is a gap.

**8. Deployment / production use — dataset format and evaluation**

At deployment, the router takes a live query, embeds it with the same pretrained embedding model used in training, and the trained classifier (SW/MF/MLP/kNN) outputs P(win_Mstrong | q). This probability (or the kNN nearest-neighbor winner) is thresholded to decide: route to on-device weak model or cloud strong model. No SE computation happens at inference time — SE is only used offline to build training labels. Deployment evaluation datasets are MT-Bench (160 open-ended questions, judged via LLM-as-a-judge), GSM8K (~1,000+ grade-school math), and MMLU (14,042 questions, 57 subjects) — chosen to be broader/different from the QA+MAWPS training distribution, testing generalization. Evaluation is purely at the CPT/cost level plus the GSM8K quality-judging described above; no production-latency or router-inference-cost numbers are given (this ties back to their own stated Limitation #2).

**9. On your idea generation** — noted, no editorializing on idea quality here, just laying out the paper as requested.

One thing worth deciding before you write the related-work paragraph: do you want to position your SAAR/PF-AAR entropy+paraphrase-uncertainty signals as *philosophically aligned* with this paper's SE approach (both use uncertainty over accuracy/preference), or as a *direct improvement* on its main weakness (the full-generation-pass cost problem)? The second framing gives you a much sharper contribution story and directly supports your "pre-inference framing tension" discussion, since this paper is a clean example of a method that claims routing-relevant uncertainty but actually pays the full generation cost to get it.