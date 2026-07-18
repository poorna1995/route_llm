# Large Language Models — Concepts From Scratch
## A detailed, phase-by-phase paper for reading and interview prep

**Purpose.** Walk the full stack in order: what an LLM is → mental models → architecture → pretraining → post-training → RAG vs fine-tuning → inference systems. End with interview questions that reuse the same stories.

**How every concept is told (do not skip this).**
The teaching style follows the public Vizuara interview series (especially [PagedAttention #5](https://www.youtube.com/watch?v=-AB6m0Spo6c)): never open with the acronym. Open with the world you are in, the naive approach, why it breaks, the root design cause, then the method, then the payoff and the new tradeoff.

**Honesty.** Numbers that come from a paper or blog are labeled as *that source’s setting*. Teaching videos give pedagogic examples (nutrition chatbot, open-book exam); those are analogies, not measurements. Do not invent benchmark scores or universal speedups.

**Deeper companion notes (same content, lecture layout):**
- `research/llm/vizuara_inference_phase1_notes.md`
- `research/llm/llm_end_to_end_survey.md`

---

# Phase 0 — The map before the details

Read this once so later sections attach to a spine.

```text
World text
  → tokenize into subwords
  → pretrain a decoder-only Transformer (base model = next-token predictor)
  → post-train (SFT → preferences / reasoning)
  → product choice: RAG and/or fine-tuning
  → serve autoregressive decode:
        prefill vs decode
        → KV cache (good compute / evil memory)
        → shrink KV (MQA / GQA / …)
        → FlashAttention (IO schedule)
        → continuous batching + PagedAttention (multi-tenant memory)
        → quantization → speculative decoding
  → evaluate capability + preference + safety + cost
```

| Symptom | Likely stage |
|---|---|
| Cannot do basics even with good prompts | Pretraining data / capacity |
| Knows facts, ignores instructions | SFT |
| Fluent but rude / unsafe / sycophantic | Preference post-training |
| Chat OK, hard math fails | Reasoning post-training or test-time compute |
| Correct but too slow / expensive | Inference systems / routing |
| Leaderboard up, users unhappy | Evaluation mismatch |

---

# Phase 1 — What an LLM actually is

## Step 1.1 — Stop saying “understands language”

### Setup
People say ChatGPT “understands” Spanish travel plans. Under the hood, the deployed object is a parameterized function: given tokens so far, it outputs a probability distribution over the next token in a fixed vocabulary.

### Problem
Language is open-ended. Hand-written grammar rules and templates do not cover web text, code, dialogue, and instructions at scale.

### Root cause
We need one training objective that (i) uses abundant unlabeled text and (ii) produces a generative system.

### Solution — Autoregressive language modeling
Factorize a sequence with the chain rule:

\[
P(x_1,\ldots,x_T)=\prod_{t=1}^{T} P(x_t \mid x_{<t}).
\]

Train by minimizing next-token **cross-entropy** on large corpora. At inference, choose the next token by sampling from that distribution (or taking \(\arg\max\)), append it, and repeat.

### Payoff
One scalable objective can absorb many downstream behaviors as “continue this prompt usefully.” This is the standard framing used across GPT-style models.

**Anchors:** Brown et al., GPT-3 (2020); standard LM factorization. We are not claiming a specific chatbot’s private training recipe.

## Step 1.2 — Tokenization (why text is not raw characters)

### Setup
Neural nets need integer IDs. Text must become a sequence of token IDs.

### Problem
- **Character-level:** sequences become very long → attention cost explodes.  
- **Word-level:** vocabulary explodes; rare words, typos, and code break.

### Root cause
We need open vocabulary without pathological sequence length.

### Solution
**Subword** tokenization. Common in GPT-style models: Byte-Pair Encoding (BPE) / byte-level BPE — merge frequent symbol pairs into subwords. Other families use WordPiece or Unigram/SentencePiece.

### Payoff
Any Unicode string can be encoded; frequent words stay short; rare pieces split.

---

# Phase 2 — Mental models you will reuse later

These are not serving kernels. They are shared intuitions under loss, architecture choices, and sampling. Learn them before optimizing decode.

## Step 2.1 — Entropy (surprise)

**Video:** [Entropy: AI Mental Model #9](https://www.youtube.com/watch?v=q3agYqgqklU)  
**Visual notes:** [lecture-09-entropy.html](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-09-entropy.html)

### How to explain it (story first — do not open with \(-\sum p\log p\))

1. Everyday disorder (messy room, cream in coffee).  
2. Formal meaning: **count of arrangements** that look the same from outside.  
3. Physics: heat = molecular jiggling → more microstates → higher entropy; Second Law.  
4. Boltzmann: \(S = k \log W\).  
5. Shannon: information = **surprise** \(=-\log p\); entropy = **expected surprise**.  
6. Only then: where LLMs use the same idea.

**Core reframing:** Entropy is not a vibe of “messiness.” Low entropy ≈ few arrangements (tidy / peaked / certain). High entropy ≈ many arrangements (messy / flat / uncertain).

### Shannon side

### Setup
Shannon (1948) needed to measure how much information a message carries.

### Problem
“The sun rose” and “it snowed in the desert” are not equally informative.

### Root cause
Common events are unsurprising; rare events are surprising.

### Solution
\[
\text{surprise}(x) = -\log p(x)
\]
Entropy of a distribution = average surprise. Fair coin → high uncertainty. Biased coin → lower average surprise.

### Where it shows up in LLM land

**(1) Cross-entropy loss = training surprise.**  
If the true next token has model probability \(p_{\text{true}}\):
\[
\mathcal{L} = -\log p_{\text{true}}.
\]
Pretraining is largely minimizing this next-token cross-entropy.

**(2) Perplexity.**  
Perplexity ≈ effective branching factor of the model’s uncertainty (teaching English: perplexity 2 ≈ coin-flip unsure; perplexity 50 ≈ ~50 doors). Lower perplexity ⇒ more confident next-token predictions *on that data*. It is an intrinsic LM metric, not the same as chat quality.

**(3) Sampling temperature.**  
Softmax with temperature \(T\): low \(T\) → peaked (low entropy, deterministic); high \(T\) → flatter (high entropy, more random). The knob is called temperature because it spreads probability mass the way physical temperature spreads molecular arrangements.

**(4) KL divergence = relative entropy.**  
Extra surprise from using the wrong distribution. Shows up as the **KL penalty** in RLHF/PPO that keeps the policy near a reference model.

**Interview one-liner:** Whenever the problem is uncertainty, surprise, confidence, randomness, or distance between beliefs — reach for entropy. In LLM land: CE trains, perplexity scores uncertainty, temperature spends entropy at decode, KL limits drift in RLHF.

## Step 2.2 — Expressivity (lift when stuck)

**Video:** [Expressivity: AI Mental Model #6](https://www.youtube.com/watch?v=YE0udJCgDqc)

### Story
If patterns cannot be separated in the current representation (classic XOR picture), **lift** to a richer / higher-dimensional space where they become separable (kernel trick, wider FFN, multi-head perspectives).

### Warning
More expressivity can also fit noise → overfitting. In Transformers, the **FFN** is the per-token “wide lift”; attention is the inter-token mixer. Do not confuse those jobs.

## Step 2.3 — Compression (narrow waist)

**Video:** [Compression: AI Mental Model #5](https://www.youtube.com/watch?v=hGS6RbAYLl0)

### Story
When something is too big (memory, \(\Delta W\), pixels), find a **narrow latent**, do the work there, project back — because the giant object was never as high-dimensional as it looked.

### Same gesture in different places
| Place | Too big | Waist |
|---|---|---|
| Autoencoder | Input dim | Latent code |
| LoRA | Full \(\Delta W\) | \(A\,(D\times r)\), \(B\,(r\times D)\), \(r\ll D\) |
| GQA / MLA-style | Full MHA KV | Shared / latent KV |
| Embedding stores | Float embeddings | Quantized / compressed vectors |

**Do not confuse:** GQA changes attention KV layout at architecture/serve time; LoRA compresses fine-tuning updates.

## Step 2.4 — Diffusion (contrast with autoregressive LLMs)

**Video:** [Diffusion: AI Mental Model #7](https://www.youtube.com/watch?v=bTEfdB2D1Ek)

### Story (not DDPM jargon first)
1. Creating a masterpiece in one shot is hard.  
2. Trick: **learn to destroy gently**, then run destruction backward.  
3. Forward: add a little noise repeatedly until nearly pure noise (and know what was added).  
4. Train a network to undo **one tiny step**.  
5. Generate: start from pure noise; repeatedly remove predicted noise.

### Contrast with ChatGPT-class inference

| | Autoregressive LLM | Diffusion-style generator |
|---|---|---|
| Order | Left-to-right, one token at a time | Iterative denoise / unmask; can update many positions |
| Starts from | Prompt prefix | Noise / mask (formulation-dependent) |
| Phase-1 tools | KV cache, GQA, PagedAttention, speculation | Different bottleneck story |

**Do not invent product claims** (“Model X ships diffusion language”). Stick to: AR is dominant for chat APIs; diffusion-for-language is an active alternative paradigm.

---

# Phase 3 — Transformer architecture (interview depth)

**Teaching video:** [LLM Interview Series #3](https://www.youtube.com/watch?v=c533te7NSpI)

Do **not** recite “attention + FFN, done.” Lead big picture → stack → one block → zoom attention.

## Step 3.1 — Scope: decoder-only for chat

Original “Attention Is All You Need” (Vaswani et al., 2017) had **encoder + decoder**. Modern chat models are **decoder-only**: generate the next token left-to-right. Unless asked for machine translation / seq2seq, focus the interview answer on the decoder stack.

## Step 3.2 — Depth 1: whole LLM = three blocks

| Block | Role |
|---|---|
| **Input** | Tokens → token embeddings + positional info → input embeddings |
| **Processor** | Stacked Transformer blocks (this *is* the architecture) |
| **Output** | Hidden state → logits over vocabulary → next-token choice |

Place the Transformer in the **processor**. Ground it with a real prompt path: “Give me a travel plan for Spain” → embeddings → stack → logits → next token.

**Same diagram, two regimes:**
- Pretrain: embeddings / \(W_Q,W_K,W_V\) / FFN weights are **learned**.  
- Inference: those weights are **fixed**; forward path shape is the same.

## Step 3.3 — Depth 2: stack many identical blocks

One block’s output feeds the next: \(T_1 \to T_2 \to \cdots \to T_L\) (dozens to 100+ layers in large models). Composable stacking is a reason the architecture scaled.

Innovations cluster on two modules inside each block:
- **Attention** variants: MQA, GQA, MLA, …  
- **FFN** variants: MoE, SwiGLU, …

## Step 3.4 — Depth 3: modules inside one block

Typical decoder block as taught in the lecture (exact norm placement varies — pre-norm vs post-norm):

1. LayerNorm (or RMSNorm in many modern models)  
2. Multi-head attention  
3. Dropout  
4. Residual / skip  
5. LayerNorm / RMSNorm  
6. Feed-forward network (MLP)  
7. Dropout  
8. Residual / skip  

| Module | First-principles job |
|---|---|
| Norm (LN / RMSNorm) | Stabilize activations / training |
| Multi-head attention | **Only place tokens talk to each other** — mix into context vectors. Causal mask: no future tokens |
| Dropout | Training regularization (classic DL) |
| Residual | Alternate path for gradients → fight vanishing gradients in deep stacks |
| FFN / MLP | **Per-token** expand (often ~\(4\times d\)) → nonlinearity → contract. Tokens do **not** mix here. Reason: **expressivity** |

**Critical contrast:** Attention = inter-token communication. FFN = intra-token wide transform. Without attention, “Harry … he” never binds.

## Step 3.5 — Depth 4: enough attention for this question

1. \(X \rightarrow Q,K,V\) via \(W_Q,W_K,W_V\).  
2. Scores \(\propto QK^\top\); scale by \(\sqrt{d_k}\); causal mask; softmax → weights.  
3. Context \(\propto\) weights \(\times V\).  
4. Multi-head: split into heads; concat.

Full KV-cache story is Interview #1; GQA is Interview #6.

## Step 3.6 — Modern knobs (flag, don’t invent)

| Older textbook | Common modern twist |
|---|---|
| Absolute positional embeddings | **RoPE** inside attention |
| LayerNorm | **RMSNorm** |
| GELU MLP | **SwiGLU** (gated) |
| Dense FFN | **MoE** routing (family-specific) |

Mention as family-specific, not universal law.

### 60-second script
“Decoder-only LLM: embeddings in, stacked Transformer blocks, logits out. Each block is norm → causal MHA (tokens mix) → residual → norm → wide FFN (per-token expressivity) → residual. Attention is the only inter-token path; FFN lifts width then contracts. Variants like GQA/MoE change attention or FFN; serving then caches \(K,V\) at decode.”

---

# Phase 4 — Attention in full (conceptual then matrix)

**Teaching video:** [LLM Interview Series #4](https://www.youtube.com/watch?v=cquX2tOODUI)

## Step 4.1 — Place it in the bird’s-eye view first

Input → Processor (attention sits inside each block after the first norm) → Output. Only then zoom.

## Step 4.2 — Conceptual problem attention solves

### Setup
Language is full of long-range links:
- `Harry boarded the train. **He** …` — who is *he*?  
- `The dog chased the ball. **It** could not catch **it**.` — first *it*→dog, second *it*→ball.  
- Code: “change the first 20 lines” needs earlier context.

### Problem
If every token were processed in isolation, the model would have **no clue about its neighbors**. Context would not exist.

### Root cause
Next-token modeling needs **links** between a token and earlier tokens, plus **how much** to weight each earlier token.

### Solution in words
Attention captures links: for a given query token, which past keys matter, and with what strength.

**One-liner:** Without attention, tokens are islands; with attention, each token can read a weighted view of its past.

## Step 4.3 — Matrix mechanism (write on the board)

Input embeddings \(X\) (e.g. 3 tokens × \(d\)). Trained in pretraining; frozen at inference.

1. \(Q = X W_Q,\quad K = X W_K,\quad V = X W_V\).  
2. Scores \(S = Q K^\top\). Row \(i\), column \(j\) = how much token \(i\) attends to token \(j\).  
3. **Causal mask:** upper triangle blocked — no future tokens.  
4. Scale (typically \(/\sqrt{d_k}\)), **softmax** → attention weights.  
5. Context \(=\) weights \(\times V\) — each position becomes a **context vector** mixed from values.

Scores decide **where** to look; values carry **what** content is mixed.

## Step 4.4 — Multi-head

Single-head = one score matrix = one perspective.  
Multi-head: multiple projection sets → multiple score matrices → concat.

**Why:** one sentence can need several bindings at once (*artist painted … woman with a brush* — brush with artist vs brush with woman).

### 60-second script
“Attention is how tokens get context: without it they don’t know their neighbors. For each position we form Q, K, V; scores are \(QK^\top\); causal mask blocks the future; softmax turns scores into weights; weighted values become the context vector. Multi-head repeats that with different projections.”

---

# Phase 5 — Pretraining

## Step 5.1 — Data

### Setup
Pretraining is next-token prediction on massive text (web, books, code, …).

### Problem
Raw crawls contain spam, boilerplate, duplicates, PII. Duplicates inflate memorization and can contaminate evaluation later.

### Root cause
Gradient updates follow the empirical data distribution. Garbage in → capability and safety issues out.

### Solution
Filter, deduplicate, and **domain-mix** (web + code + books + …), as described in modern open recipes (e.g. Llama-family reports). Synthetic rewritten data is sometimes used carefully; overuse risks distribution collapse.

## Step 5.2 — Scaling laws

### Setup
Pretraining costs money. Choose model size and number of training tokens under a FLOP budget.

### Problem
“Just make the model bigger” can waste compute if the model is undertrained on too few tokens (or the reverse).

### Root cause
Loss depends jointly on parameters, data, and compute — not parameters alone.

### Solution
Empirical scaling laws: Kaplan et al. (2020); Hoffmann et al. / Chinchilla (2022). Chinchilla emphasizes compute-optimal **tokens-per-parameter**. Fits are empirical; data quality shifts constants. This guides budgets; it does **not** guarantee every downstream skill scales smoothly.

## Step 5.3 — What pretraining produces — and what it does not

After pretraining you have a **base model**: a strong next-token predictor.

It does **not** reliably:
- follow user instructions,  
- refuse unsafe asks,  
- use a chat format,  
- prefer helpful answers among many fluent ones.

**Root cause:** the objective never saw an explicit “be a helpful assistant” reward — only “predict the next token on internet-like text.”

Hand the baton to **post-training**. Do not expect prompting alone to fully replace missing base competence.

## Step 5.4 — Training systems (names after the problem)

### Setup
A frontier run does not fit on one GPU: weights, optimizer states, activations, long sequences.

### Problem
Naive data-parallel replication of full Adam states OOMs; single-device batch is too small.

### Solution toolkit (standard, not a new paper claim)
AdamW + schedules; mixed precision (BF16/FP16/FP8); gradient checkpointing; data / tensor / pipeline parallelism; ZeRO / FSDP; sequence/context parallelism for long context.

---

# Phase 6 — Post-training

## Step 6.1 — Supervised Fine-Tuning (SFT)

### Setup
You have a base LM. Users want answers to instructions, not random web continuations.

### Problem
Ask a base model a question; it may continue as if writing a webpage, ignore the ask, or produce the wrong format.

### Root cause
Pretraining distribution ≠ assistant distribution.

### Solution
Collect \((\mathrm{prompt}, \mathrm{desired\ response})\) demonstrations and fine-tune with ordinary likelihood. This is Step 1 of InstructGPT (Ouyang et al., 2022).

### Payoff
Teaches format, tool schemas, basic instruction following. Still insufficient when many answers are fluent but differently preferred.

## Step 6.2 — Classic RLHF (do not start with “PPO”)

### Setup
After SFT, the model can produce multiple fluent answers. Humans still have preferences: more helpful, less toxic, more truthful.

### Problem
SFT clones demonstrations. It does not directly optimize “prefer A over B.” Writing one gold answer for every open-ended prompt is incomplete.

### Root cause
We need a ranking signal, then a way to optimize against it without reward hacking / drift.

### Solution — InstructGPT three steps (Ouyang et al., 2022)
1. **SFT** on demonstrations.  
2. **Reward model (RM):** humans compare outputs; train an RM to predict the preferred one.  
3. **PPO:** optimize the policy to increase RM reward, with a **KL penalty** toward a reference policy (limits over-optimization). The KL is the relative-entropy idea from Phase 2.

### Tradeoffs
Preference data cost, RM misspecification, reward hacking, multi-model training complexity (policy + value + RM + reference).

**Sources:** Ouyang et al., 2022; OpenAI instruction-following writeup.

## Step 6.3 — DPO (problem first)

### Setup
RLHF works, but the stack is heavy: sample → RM → PPO with value model and KL control.

### Problem
Engineering instability and cost make iteration slow for many labs.

### Solution
**DPO** (Rafailov et al., 2023): train the policy directly on preference pairs with a classification-style objective that implicitly encodes the reward (under Bradley–Terry + KL-constrained optimal-policy assumptions). Related objectives (IPO, KTO, …) exist for different data/failure modes — name them only after stating which pain point they target.

## Step 6.4 — Reasoning post-training (high level, no invented scores)

Chat alignment ≠ contest math / coding reliability.

Families (keep claims checkable):
- **ORM** — score final answer when a verifier exists.  
- **PRM** — score steps (denser credit).  
- **RL with verifiable rewards** — unit tests / exact checkers.  
- **GRPO** (DeepSeekMath; used in DeepSeek-R1 line): group of outputs per prompt; group-normalized rewards as advantages; no separate value network.

Do **not** invent score tables.

## Step 6.5 — Parameter-efficient fine-tuning

Full copies of weights per task are impractical. Task updates are often low-rank relative to full \(W\). **LoRA** / **QLoRA** / adapters store small \(A,B\) instead. Fine-tune on smaller GPUs; serve multiple adapters on one base.

---

# Phase 7 — RAG vs fine-tuning (application design)

**Teaching video:** [LLM Interview Series #8](https://www.youtube.com/watch?v=cCXjumE70-g)

This is a **system-design** question. Interviewers test whether you can choose an approach for a vague industry request — not recite two definitions.

## Step 7.1 — The interview trap

You may know RAG and fine-tuning in isolation. The hard part is connecting them: *for this product, which one (or both), and why?*

Vague client: “Build a chatbot for our domain.”  
Strong answer: clarify **grounded lookup** of owned documents vs **new behavior / style / pattern** that is not literally a passage to retrieve.

## Step 7.2 — What problem does RAG solve? (before the acronym)

### Setup
A nutrition / fitness company has ~2,000 curated resources (their IP). Customers ask “what should I eat for breakfast?” Answers must be **grounded** in those resources — not a generic web opinion.

### Naive solution
Stuff **all** resources into the prompt every turn.

### Problem
Corpus can be huge (teaching order-of-magnitude in the video: tens–hundreds of millions of tokens). Context windows are large but still limited (example contrast used in teaching: ~1M ≪ 100M). You cannot fit the whole library.

### Root cause
Context is a scarce resource (“like land”). Only a **slice** of the corpus is relevant per question.

### Solution — Retrieval-Augmented Generation (Lewis et al., 2020)
1. Chunk the corpus (paragraphs / sections — a design choice).  
2. Embed chunks → vectors; embed the query.  
3. Retrieve top-\(k\) chunks by similarity (e.g. nearest neighbors / dot product).  
4. Put **query + retrieved chunks** into the LLM context; generate.

Weights stay **fixed**. Name: **retrieval** augments **generation**.

**Production caveat from the teaching video:** embedding and storing vectors is expensive; stores often use quantization/compression. “Vectorless RAG” (tree-style) is mentioned as a fork — do not invent details unless asked.

**Analogy:** open-book exam — flip to relevant pages; book stays beside you.

**When you do not need RAG:** if owned knowledge fits in context with margin, pass it directly.

## Step 7.3 — What is fine-tuning? (start from pretraining)

### Setup
A pretrained LM is billions of “knobs” fitted on internet-scale data. A raw pretrained checkpoint is often weak at following instructions (“convert 35 km to meters”).

### Problem
You need narrower behavior (instructions, domain style, persona) without full pretraining again.

### Solution
Curate a much smaller supervised set of (question, answer) pairs (teaching ballpark: order of ~10k, not pretraining scale). Train again so weights **nudge**. That is fine-tuning.

**Key contrast:** fine-tuning **changes weights**; RAG does **not**.

**Pointer only:** the video links “why small data works” to intrinsic dimensionality of a pretrained model — say “small supervised nudge on a strong pretrained base,” not a fake theorem.

## Step 7.4 — Decision examples

### Example A — Digital clone → fine-tuning
User dumps LinkedIn / YouTube / Instagram. You want a chatbot that **sounds like them** on *new* questions not literally in the posts.

| Approach | What happens |
|---|---|
| RAG | Retrieves relevant posts; does **not** teach speaking patterns for novel asks |
| Fine-tuning | Adjusts weights so patterns of speech are internalized |

### Example B — Pharma / nutrition grounded bot → start with RAG
Answers must stay inside owned documents. Start with RAG: faster, cheaper, simpler. Caveat: RAG surfaces what’s (near) in the store; it does not discover deep new patterns.

### Escalation
```text
easiest → harder
RAG  →  fine-tune pretrained model  →  train SLM from scratch (pretrain)
```

## Step 7.5 — Comparison table (draw this)

| Axis | RAG | Fine-tuning |
|---|---|---|
| Finds patterns in data? | No — retrieves existing text | Yes — weights encode patterns |
| Cost | Lower | Higher |
| Simplicity | Higher | Lower |
| Personalization | Weak | Strong when trained for it |
| Analogy | Open-book exam | Study the night before (no book in the room) |
| Weights at serve | Unchanged | Changed (or LoRA/adapters) |

Freshness: RAG wins when knowledge updates often (re-index). Many real systems do **both** — say that *after* the primary axes.

### 60-second script
“RAG and fine-tuning solve different failures. If a company’s library won’t fit in context, RAG retrieves relevant chunks and grounds the answer — open-book; weights stay fixed. Fine-tuning nudges weights on a small labeled set so behavior or patterns change — studying the night before. Grounded nutrition/pharma bot: start RAG. Digital clone on novel questions: fine-tune. Compare patterns, cost, simplicity, personalization; escalate RAG → FT → pretrain only as needed.”

---

# Phase 8 — Inference engineering begins

**Course framing:** [inference.vizuara.ai](https://inference.vizuara.ai/)

### Setup
Training builds weights. Users experience **serving**: tokens out per second, dollars per million tokens, GPU OOMs at 2 a.m.

### Problem
A notebook `model.generate()` that feels fine for one prompt collapses under concurrent users, long contexts, and cost caps.

### Root cause
Inference is autoregressive, often memory-bandwidth heavy, and multi-tenant. Training tricks do not automatically solve decode.

### Discipline
Treat inference as its own stack: hardware → kernels → attention/KV design → scheduler → quantization → speculation.

### Hardware lab mindset
The same 7B weights bottleneck differently on a laptop, Raspberry Pi, phone, or datacenter GPU. Measure tok/s, VRAM/RSS, quality per device. Do not quote an A100 blog post as universal truth.

---

# Phase 9 — Prefill vs decode, latency, memory, roofline

## Step 9.1 — Prefill vs decode (must name this split)

Example: prompt `A sunset is` → model continues `extremely beautiful…`

1. **Prefill:** consume **all** prompt tokens (largely in parallel), run attention, emit the **first** output token. Often more compute-bound.  
2. **Decode:** emit **one new token at a time** until stop. This is where naive recompute hurts.

Until first token → prefill; after that → decode. Every later optimization attaches to this split.

**Root cause of the split:** autoregressive factorization \(P(x_t\mid x_{<t})\) forces sequential new tokens even when the prompt can be processed in parallel.

## Step 9.2 — Latency vs throughput

- **Latency:** TTFT (time to first token), TPOT (time per output token) for *one* user.  
- **Throughput:** tokens/sec or requests/sec across *many* users.

GPUs want large batches; users want snappy single-stream decode. Continuous batching (Phase 13) trades them consciously.

## Step 9.3 — Memory bill of materials

During inference, GPU/host memory typically holds:
1. **Weights**  
2. **Activations** (short-lived)  
3. **KV cache** (grows with sequence × layers × KV layout)

At long context / high concurrency, **KV often dominates** and OOMs first.

## Step 9.4 — Roofline intuition

### Setup
Every kernel needs FLOPs and bytes from HBM.

### Problem
You “optimize FLOPs” but wall-clock does not move.

### Root cause
Decode is often **memory-bandwidth bound** (reload weights + KV each step), not FLOP-bound.

### Solution direction
Quantization, GQA/MQA, FlashAttention IO patterns, speculation — attack **bytes moved** or **steps taken**.

---

# Phase 10 — KV cache (full interview walkthrough)

**Teaching video:** [LLM Interview Series #1](https://www.youtube.com/watch?v=CxRGWfcGVbs)

Do **not** open with “KV cache makes generation faster.” Construct from first principles.

## Step 10.1 — It is an inference concept
Shows up at serve / generate time. Weights \(W_Q,W_K,W_V\) are already fixed.

## Step 10.2 — Prefill at matrix level
Input embeddings \(X\) multiply fixed \(W_Q,W_K,W_V\) → \(Q,K,V\).  
Scores \(\propto QK^\top\) (scale / softmax / causal) → weights.  
Context \(\propto\) weights \(\times V\). Last position → logits → first generated token.

**At end of prefill:** you already computed \(K\) and \(V\) rows for every prompt token. **Cache them.**

## Step 10.3 — Naive decode (the wrong picture)
To generate the next token, a naive approach recomputes \(Q,K,V\) for the **entire** sequence every step, full score matrix, full context matrix — then keeps only the last row.

That recomputes past tokens’ \(K,V\) that **cannot change** (same tokens, frozen weights).

## Step 10.4 — Two realizations that define KV cache

**Realization A.** Only the latest context vector is needed for the next token — not a full context matrix of all positions for the logits step.

**Realization B.** That latest context still needs **full** past \(K\) and **full** past \(V\). The new query attends over all past keys; the weighted sum uses all past values.

So you need the whole \(K,V\) history — but you must not **recompute** it.

## Step 10.5 — Actual decode with cache
1. Embed **only the new token** (\(1 \times d\)).  
2. Compute only that token’s \(q_{\text{new}}, k_{\text{new}}, v_{\text{new}}\).  
3. **Append** \(k_{\text{new}}, v_{\text{new}}\) to cached \(K,V\).  
4. Use \(q_{\text{new}}\) against full \(K^\top\) → scores for this step only (\(1 \times\) seq).  
5. Mix with full \(V\) → one context vector → logits → next token.  
6. Repeat; cache grows by one \(K,V\) row per new token (per layer / head layout).

## Step 10.6 — Why not a Q cache?
You need full \(K\), full \(V\), but **only** \(q\) for the **new** token. Past queries are not reused for the next step’s scores. Hence **KV** cache, not QKV cache.

## Step 10.7 — Payoff vs evil

| Good | Evil |
|---|---|
| Avoids redundant matmuls; compute closer to **linear** in new tokens vs **quadratic** recompute of the past | Memory grows with sequence × layers × KV heads × dim × 2 |
| Better interactive latency | Moving KV from HBM each step → bandwidth bottleneck → GQA, PagedAttention, quant |

KV size scales roughly with:
`layers × kv_heads × head_dim × sequence_length × bytes_per_element × 2`  
(Exact formula depends on GQA/MQA and layout.)

## Step 10.8 — Unknown output length (bridge)
You do not know final decode length when the request arrives. Contiguous over-reservation → fragmentation → **PagedAttention** (Phase 12).

### Related names (state the problem first)
| Idea | Problem | One line |
|---|---|---|
| Prefix / prompt caching | Repeated system prompts | Reuse KV for shared prefixes |
| Chunked prefill | Huge prefills block the GPU | Split prefill; interleave with decode |
| Eviction / compression (H2O, StreamingLLM, …) | Infinite context, finite VRAM | Drop/compress less useful KV (method-specific) |

### 60-second script
1. Inference-only; prefill vs decode.  
2. Prefill builds \(Q,K,V\); **store \(K,V\)**.  
3. Next token needs latest context ⇒ full \(K,V\) + new \(q\) only.  
4. Don’t recompute past \(K,V\); append new \(k,v\).  
5. No Q-cache.  
6. Speeds decode; costs memory/bandwidth.

---

# Phase 11 — Shrink the KV: MHA → MQA → GQA

**Teaching video:** [LLM Interview Series #6](https://www.youtube.com/watch?v=mtsY7JsGQjw)

Shared setup: at decode, loading \(K,V\) from the cache is a major **HBM → compute** bandwidth cost.

## Step 11.1 — Place attention, then multi-head
LM = input → stacked Transformer blocks → logits. Inside a block: norm → **MHA** → residual → norm → FFN → residual.

Multi-head exists for **perspectives** (ambiguous attachment: brush with artist vs woman). Paper for MHA: Vaswani et al., 2017.

## Step 11.2 — The need for GQA starts in the KV cache
In MHA with \(h\) heads, decode stores per-head keys and values. **KV cache size scales with the number of KV heads.** Larger KV ⇒ more bytes moved each decode step.

Innovation need: **shrink KV without throwing away too much quality.**

## Step 11.3 — MQA (Shazeer, 2019) — extreme fix
Keep different queries per head, but force **one shared K and one shared V** across all heads.

**Store:** only one \(K\) and one \(V\) (not \(h\) of each) → large memory cut in the counting argument (up to ~\(h\times\)).

**Why quality suffers:** heads diversify attention scores. If all heads share the same \(K\) (and \(V\)), diversity comes only from different \(Q\)s → weaker multi-perspective modeling. Teaching frame: MQA = low KV (good) + weaker language pattern capture (bad).

## Step 11.4 — GQA (Ainslie et al., EMNLP 2023) — middle
Partition query heads into \(g\) groups. **Within a group**, share K and V. **Across groups**, K/V differ.

Toy with 4 query heads and 2 groups:
- Group 1: \(K_1=K_2\), \(V_1=V_2\)  
- Group 2: \(K_3=K_4\), \(V_3=V_4\)  
- But \(K_1 \neq K_3\)

Endpoints: \(g=h\) → MHA; \(g=1\) → MQA.

| | KV memory | Multi-perspective quality |
|---|---|---|
| **MHA** | Highest | Highest |
| **GQA** | Middle | Middle |
| **MQA** | Lowest | Lowest (often too weak) |

Teaching practical claim: production open models rarely ship pure MQA; **GQA is the common compromise** (e.g. Llama family uses GQA; number of groups is a hyperparameter — verify per model card, don’t invent).

Honest drawback: GQA is middle for everything. MLA-style latent attention is a **different** mechanism — do not equate it with GQA.

### 60-second script
Locate MHA → multiple heads for perspectives → MHA KV ∝ heads hurts decode → MQA shares one KV (extreme) → GQA shares KV per group (middle) → draw the spectrum.

## Step 11.5 — Other long-context families (brief)
| Family | Problem | Idea |
|---|---|---|
| Sliding window | Full \(n^2\) too costly | Attend inside a window |
| Sparse attention | Dense all-pairs wasteful | Structured sparsity |
| SSM / Mamba (Gu & Dao et al., 2023) | Want better asymptotic cost | Selective state that summarizes the past |

---

# Phase 12 — FlashAttention (IO schedule, not new math)

### Shared problem
Standard attention materializes large \(S=QK^\top\) (and softmax) in HBM. Memory traffic dominates even when FLOPs look fine. Root cause: GPU **HBM ↔ SRAM** IO.

### FlashAttention-1 (Dao et al., NeurIPS 2022)
- Tiling: bring blocks of \(Q,K,V\) into SRAM.  
- **Online softmax** so you never need the full score matrix in HBM.  
- Recomputation in backward (training) to save memory.  
- **Exact** attention (same math result, different IO schedule).

### FlashAttention-2
Better work partitioning / parallelism; higher utilization; MQA/GQA support called out in FA2 materials. Reported ~2× over FA1 in *their* settings (hardware/workload dependent).

### FlashAttention-3 (Dao et al., NeurIPS 2024)
Targets Hopper (H100) asynchrony (Tensor Cores + TMA): warp specialization, interleave matmul and softmax, FP8 path. Paper reports ~1.5–2.0× vs FA2 on H100 in BF16/FP16 regimes — **on that hardware/paper setting**.

**Interview line:** FlashAttention does not change mathematical attention; it changes the **IO schedule** so GPUs stop drowning in HBM traffic.

---

# Phase 13 — Serving: continuous batching and PagedAttention

**Teaching video:** [LLM Interview Series #5](https://www.youtube.com/watch?v=-AB6m0Spo6c)  
**Paper/blog:** Kwon et al., SOSP 2023; [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm)

## Step 13.1 — Why a serving engine ≠ `transformers.generate`
HF generate is fine for demos. Multi-tenant production needs a scheduler, memory manager, and batching policy.

## Step 13.2 — Continuous batching

### Setup
Requests arrive and finish at different times.

### Problem
Static batching leaves GPU lanes idle when short sequences finish.

### Solution
Dynamically add/remove sequences each iteration (iteration-level scheduling; Orca/vLLM lineage).

## Step 13.3 — PagedAttention — full layers (interview gold path)

### Layer 1 — Basics: GPU memory during inference
1. Model weights (fixed after training)  
2. Activations (ephemeral)  
3. KV cache (grows; major during serving)

### Layer 2 — Problem: traditional KV allocation wastes memory
You do **not** know final decode length in advance. Naive approach: reserve a **contiguous** KV slab per request up to `max_tokens`.

Failures:
- **Reserved but unused** memory while a request is still generating (cannot give that RAM to another user).  
- After a short request finishes early, free holes may be **too small / wrongly placed** for a waiting large request (**fragmentation**).

Result: low effective batch size → poor GPU utilization → low throughput.

vLLM authors report existing systems can waste a large fraction of KV memory via fragmentation and over-reservation (blog: on the order of **60–80%** waste in the regimes they studied). Treat as **their characterization**, not a universal constant.

### Layer 3 — Root cause
KV for a request is treated like one contiguous physical allocation. Variable lengths + contiguous reservation ⇒ fragmentation.

### Layer 4 — Solution: PagedAttention (+ vLLM)
Inspired by OS **virtual memory / paging**:

1. Split KV memory into fixed-size **blocks** (pages). Block size is a system parameter (often discussed as **16 tokens** per block in vLLM materials).  
2. Each block stores K/V for that many tokens.  
3. Maintain a **free block list** and a **block table** mapping each request’s logical token spans → physical blocks.  
4. Allocate blocks **on demand** as decode proceeds; a request’s blocks need **not** be contiguous.  
5. Attention kernel gathers K/V by following the block table (**PagedAttention**).

Additional capability from the same abstraction ([vLLM blog](https://vllm.ai/blog/2023-06-20-vllm)): share blocks across sequences (e.g. parallel samples from one prompt) with reference counting / copy-on-write.

### Layer 5 — Payoff
Near-zero waste except partial last block (blog: under ~**4%** waste in their characterization) → larger batches → higher throughput. Paper/blog report large gains vs prior systems **in their benchmarks**. **Do not treat a single speedup number as universal.**

### One engine step (mental model)
Schedule requests → allocate/free KV blocks → run forward (prefill and/or decode) → sample → update caches / finish requests.

---

# Phase 14 — Quantization

### Setup
Decode is often memory-bandwidth bound; weights + KV compete for VRAM.

### Problem
FP16/BF16 models may not fit, or bandwidth dominates latency.

### Solution family (separate the names)
- Precision ladder: FP16/BF16 → INT8 → INT4 (and related formats).  
- Post-training methods: GPTQ, AWQ, GGUF ecosystems — different recipes for *which* tensors and *how* calibrated.  
- Can target weights, and sometimes KV/activations depending on the stack.

### Interview discipline
Always say **which tensors**, **which bitwidth**, and **what quality/latency you measured**. Never claim “INT4 is always better.”

---

# Phase 15 — Speculative decoding

### Setup
Target-model decode is serial: one (or few) tokens per expensive forward.

### Problem
Latency wall from sequential target steps.

### Solution — draft–verify (Leviathan et al., 2023)
A cheaper **draft** proposes several tokens; the **target** verifies them in parallel and accepts a prefix consistent with the target distribution (exact algorithms preserve the target’s distribution when done correctly).

### Payoff / tradeoff
Speedup depends on **acceptance rate**. Variants: small draft model, Medusa-style heads, EAGLE / EAGLE-3, MTP objectives. Name one method and its tradeoff; do not invent acceptance numbers.

---

# Phase 16 — Sampling and frozen-weight procedures

**Sampling:** temperature, top-\(k\), nucleus / top-\(p\) (Holtzman et al.), beam search. Changes behavior without retraining.

**With frozen weights:** few-shot / ICL (Brown et al., 2020); Chain-of-Thought (Wei et al., 2022); self-consistency (Wang et al., 2022); ReAct / tools; RAG (Phase 7); test-time compute (more samples / search / “thinking” tokens on hard items).

---

# Phase 17 — Evaluation (brief but honest)

### Problem
A single leaderboard number can rise while users get worse or unsafe answers.

### Root cause
Benchmarks measure narrow operationalized skills. Proxies can be gamed; contamination inflates scores; LLM-as-judge has biases.

### Practice
Portfolio: capability suites, instruction rubrics, human preference (arena-style Elo), safety, **plus cost/latency**. Report threats honestly.

Routing systems (optional): report quality **and** cost (strong-call fraction, PGR) — RouteLLM (Ong et al., ICLR 2025), Hybrid LLM (Ding et al., 2024).

---

# Phase 18 — End-to-end in one breath

```text
Text in the world
  → filter / dedup / mix + tokenize
  → pretrain next-token Transformer (+ distributed training)
  → base model
  → SFT → preferences (RLHF or DPO/…) / reasoning RL
  → product: RAG and/or fine-tune
  → serve: prefill/decode, KV cache, GQA, FlashAttention,
           continuous batching + PagedAttention, quant, speculation
  → optional tools / test-time compute / routing
  → evaluate → iterate
```

---

# Phase 19 — Explicit non-claims

- No claim that one alignment method is universally best.  
- No fabricated benchmarks or universal speedups.  
- Preference alignment does not “solve” safety.  
- Agents = orchestration + tools + memory **on top of** this stack.  
- Hardware and traffic change which optimization wins — measure.

---

# Phase 20 — Landmark sources (only these; do not invent)

| Topic | Anchor |
|---|---|
| Transformer | Vaswani et al., 2017 |
| GPT-3 / few-shot | Brown et al., 2020 |
| Scaling | Kaplan et al., 2020; Hoffmann et al., 2022 |
| InstructGPT / RLHF | Ouyang et al., 2022 |
| DPO | Rafailov et al., 2023 |
| RAG | Lewis et al., 2020 |
| CoT / self-consistency | Wei et al., 2022; Wang et al., 2022 |
| Nucleus sampling | Holtzman et al. |
| MQA | Shazeer, 2019 |
| GQA | Ainslie et al., EMNLP 2023 |
| FlashAttention | Dao et al., 2022; FA3 NeurIPS 2024 |
| PagedAttention / vLLM | Kwon et al., SOSP 2023; vLLM blog 2023 |
| Speculative decoding | Leviathan et al., 2023 |
| Mamba | Gu & Dao et al., 2023 |
| GRPO / R1 line | DeepSeekMath 2024; DeepSeek-R1 2025 |
| RouteLLM | Ong et al., ICLR 2025 |
| Teaching | Vizuara Interview #1 KV, #3 Transformer, #4 Attention, #5 PagedAttention, #6 GQA, #8 RAG vs FT; Mental Models #5–#7, #9 |

---

# Phase 21 — Interview questions (with full answer skeletons)

Speak **setup → problem → root cause → solution → payoff**. Prefer an example you thought through.

## Foundations

**Q1. What is an LLM, really?**  
Parameterized next-token model via \(P(x_t\mid x_{<t})\); train with CE; generate by sampling. Not a separate “understanding” module. (Brown et al. / standard LM.)

**Q2. Why Transformers instead of RNNs?**  
Need parallel training and long-range mixing at scale; self-attention + FFN + causal mask. (Vaswani 2017.)

**Q3. Explain the Transformer for a chat model.**  
Use Phase 3 script: three blocks → eight modules with *why* → QKV zoom → variants live in attention/FFN. ([Video #3](https://www.youtube.com/watch?v=c533te7NSpI))

**Q4. What is attention?**  
Use Phase 4 script: islands → Harry/he → QKV → causal → multi-head. ([Video #4](https://www.youtube.com/watch?v=cquX2tOODUI))

**Q5. What is entropy in LLMs?**  
Expected surprise; CE trains; perplexity scores; temperature spends; KL limits RLHF drift. ([Video Mental Model #9](https://www.youtube.com/watch?v=q3agYqgqklU))

**Q6. Expressivity vs compression?**  
Expressivity: lift when stuck (watch overfit). Compression: narrow waist, work small, project up.

## Pretraining / post-training

**Q7. What does pretraining give / miss?**  
Strong base next-token predictor; misses reliable instructions/preferences → post-training.

**Q8. What is SFT?**  
Supervised \((\mathrm{prompt},\mathrm{response})\) fine-tune; InstructGPT step 1. (Ouyang 2022.)

**Q9. Explain RLHF without starting at PPO.**  
Many fluent answers → preference signal → RM → optimize policy with KL to reference. Tradeoffs: cost, hacking, complexity.

**Q10. What problem does DPO solve?**  
Heavy RM+PPO loop; train policy on preference pairs directly. (Rafailov 2023.)

**Q11. What is LoRA?**  
Task updates often low-rank; store \(A,B\) with \(r\ll d\); cheaper multi-task fine-tunes.

## RAG vs fine-tuning

**Q12. RAG vs fine-tuning — when which?**  
Use Phase 7 full story + table + escalation. ([Video #8](https://www.youtube.com/watch?v=cCXjumE70-g))

**Q13. Can you use both?**  
Yes: FT for behavior/format; RAG for grounded facts — after primary axes.

## Inference core

**Q14. Prefill vs decode?**  
Prefill = parallel over prompt, first token; decode = sequential; different bottlenecks.

**Q15. What is the KV cache? Why not cache Q?**  
Use Phase 10 full walkthrough. ([Video #1](https://www.youtube.com/watch?v=CxRGWfcGVbs))

**Q16. Latency vs throughput?**  
Name the product goal; continuous batching trades them.

**Q17. Why is decode often memory-bound?**  
Each step reloads weights (+KV) from HBM for little new compute.

## Attention variants / kernels

**Q18. MHA vs MQA vs GQA?**  
Use Phase 11 spectrum story. (Ainslie 2023; [Video #6](https://www.youtube.com/watch?v=mtsY7JsGQjw))

**Q19. What is FlashAttention?**  
Same math; IO-aware tiling + online softmax. (Dao et al.)

## Serving

**Q20. What is PagedAttention?**  
Use Phase 13 five layers exactly. ([Video #5](https://www.youtube.com/watch?v=-AB6m0Spo6c); Kwon/vLLM)

**Q21. Continuous batching?**  
Sequences finish at different times; dynamically admit/finish.

**Q22. Walk one vLLM-style engine step.**  
Schedule → allocate/free KV blocks → forward → sample → update.

## Quant / speculation / design

**Q23. Why quantize? Which tensors?**  
Fit + bandwidth; specify tensors, bitwidth, method, measured quality.

**Q24. Speculative decoding?**  
Draft proposes; target verifies; preserve target distribution; acceptance rate is the dial. (Leviathan 2023.)

**Q25. Design a low-latency, high-throughput serving system.**  
Prefill/decode → KV good/evil → GQA → FlashAttention → continuous batching + PagedAttention → quant → optional speculation → measure TTFT, TPOT, throughput, VRAM; state tradeoffs.

**Q26. Domain chatbot tomorrow — what first?**  
Clarify grounded docs vs persona. Docs → RAG MVP. Style → FT/LoRA. Escalate with evidence.

**Q27. OOM at long context with many chats?**  
KV size → GQA? PagedAttention/fragmentation? max-token reservation? quant? batch vs latency SLO?

**Q28. Accuracy fine, p95 latency bad?**  
Prefill vs decode; TTFT vs TPOT; batching; KV bandwidth; speculation acceptance; FLOP vs bandwidth bound.

## Quick-fire openers

| If they ask… | Open with… |
|---|---|
| Entropy | Disorder/surprise → CE, temp, KL |
| Transformer | Input / processor / output, then modules |
| Attention | Tokens are islands → QKV → causal |
| KV cache | Prefill/decode recompute waste |
| GQA | MHA KV bandwidth → share KV in groups |
| FlashAttention | HBM traffic, not new math |
| PagedAttention | Contiguous KV fragmentation |
| Quantization | Memory-bound decode + which tensors |
| Speculative decoding | Serial target steps → draft–verify |
| RAG vs FT | Grounded retrieve vs weight patterns |

## Universal answer contract
1. Do **not** start with the definition of X.  
2. System context.  
3. Naive approach + break.  
4. Root design mistake.  
5. X + mechanism.  
6. Payoff + tradeoff.  
7. One credible source.

---

*Files: `research/llm_concepts_from_scratch.md` (this) · `research/llm_concepts_from_scratch.tex` (LaTeX twin)*
