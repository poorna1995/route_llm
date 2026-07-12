# Related Work

We situate our work within **supervised LLM routing**: methods that learn a mapping from queries to models using labeled outcomes—preferences, quality gaps, correctness scores, or uncertainty-derived winners—and then apply the trained router at inference time. We organize prior work by how those labels are obtained and by the size of the candidate pool, then contrast this paradigm with our unsupervised, signal-based formulation.

## Binary Supervised Routing from Preferences and Quality Gaps

A large line of work frames routing as a binary decision between a weak/cheap and a strong/expensive LLM.

**RouteLLM** (Ong et al., ICLR 2025) treats routing as Bradley–Terry win-probability estimation: given a query, predict *P*(strong wins | *q*) and threshold against a cost parameter. Training relies on Chatbot Arena preference battles, optionally augmented with gold-labeled MMLU comparisons and GPT-4–judged pairs. The router is parameterized in several ways (similarity-weighted ranking, matrix factorization, BERT, and causal LLM classifiers), but in all cases the learning target is a supervised preference label; at inference the router sees only the query text and never inspects a candidate model's generation behavior.

**Hybrid LLM** (Ding et al., 2024) also trains a binary router, but constructs labels from an estimated *quality gap* between a small and a large model. For each training query it samples multiple responses from both models, scores them with an automatic metric (BARTScore), and forms hard or soft labels for whether the small model matches or exceeds the large one; a threshold relaxation is introduced when the large model dominates almost everywhere and labels become imbalanced. A DeBERTa classifier is then trained to predict this label from the query alone. Thus generations are used to *build training labels*, while serving-time routing remains a single query-encoder forward pass.

Both systems target cost-efficient quality recovery and provide tunable thresholds at test time. They differ in label source—human (and judge) preferences versus metric-based quality gaps—but share the same supervised structure: labeled query-to-model outcomes first, predictive router second.

## Multi-LLM Supervised Routing

RouteLLM and Hybrid LLM leave open the problem of routing over many heterogeneous models rather than a fixed pair.

**IRT-Router** (Song et al., ACL 2025) addresses *N*-way routing by casting each LLM as a latent ability vector and each query as difficulty and discrimination parameters in an Item Response Theory model. Predicted correctness is combined with a fixed per-model cost into a linear reward, and the router selects argmax over candidates. Training requires a dense matrix of empirical performance scores obtained by running every candidate LLM on every training query and scoring against ground truth. Interpretability comes from the learned ability–difficulty structure, not from named generation-time signals such as entropy.

**MixLLM** (Wang et al., 2025) further expands the supervised objective to quality, financial cost, and latency under a streaming query load. It fine-tunes domain-aware query embeddings, trains independent per-LLM quality and length regressors on observed outcomes, and combines a quality–cost tradeoff with a LinUCB-style uncertainty bonus and a waiting-time penalty. The uncertainty term measures confidence in the *router's* prediction over embedding space (bandit exploration), not uncertainty in a candidate LLM's own output distribution.

**ICL-Router** (Wang et al., 2025) targets adding a new model without retraining router weights. It represents each LLM by an in-context capability profile of (query vector, correct/incorrect) pairs collected on a curated profiling set, and trains a router LLM to predict success conditioned on that profile. The profiles are still outcome-based correctness labels; the paper optimizes routing accuracy rather than a joint cost–quality objective.

Across these multi-LLM systems, supervision takes the form of correctness or quality/length labels over query–model pairs (or compact profiles derived from them). None of them use model-conditional entropy or paraphrase stability of a live query as the primary routing feature.

## Uncertainty as a Source of Supervised Labels

**Zhang et al.** (2025) replace preference or accuracy winners with labels derived from **semantic entropy (SE)**: multiple generations per model are clustered by semantic equivalence, SE is computed over meaning clusters, and the more confident (lower-SE) model is marked the winner when the SE gap exceeds a threshold. These SE-derived winners train the same family of lightweight predictive routers used in preference-based routing. SE is computed *offline for label construction*; at deployment the router again maps an embedded query to a weak/strong decision without recomputing SE. Thus uncertainty improves the *supervised training signal*, but the deployed system remains a supervised query-to-label predictor rather than an unsupervised signal-driven router.

## Positioning: From Supervised Outcomes to Unsupervised Routing Signals

| Method | Pool | Primary label / signal | Inference input |
|---|---|---|---|
| RouteLLM | binary | preference winners | query only |
| Hybrid LLM | binary | quality-gap labels | query only |
| IRT-Router | *N*-way | correctness matrix | query (+ model embeds) |
| MixLLM | *N*-way | quality/length labels | query embeds + load state |
| ICL-Router | *N*-way | correct/incorrect profiles | query + capability profile |
| Zhang et al. | binary | SE-derived winners | query only |
| **This work** | LLM pool | unsupervised routing signals | signal vector (rules / weighted) |

Prior work learns routers from prior labeled outcomes and at inference typically conditions only on the query (or on a static model profile), not on live model-dependent behavioral signals for that query. Where uncertainty appears, it is either (i) a training-label proxy (Zhang et al.) or (ii) router/bandit prediction uncertainty over embeddings (MixLLM), not generator uncertainty used directly to choose a model.

Our work studies **unsupervised LLM routing**: estimate **routing signals** (model-independent: query complexity; model-dependent: model-conditional entropy, paraphrase-based uncertainty) and decide via rules or parameter weighting. Labeled data, when used, learn combination weights only—not the primary mapping from preference-/outcome-scale supervision. We emphasize a **pre-inference** framing. Agent extension is out of scope for this comparison.
