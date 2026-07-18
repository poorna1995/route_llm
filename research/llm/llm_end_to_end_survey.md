# Large Language Models End to End
## Explained the Interview Way: Basics → Problem → Root Cause → Solution

**Style source (example, not the only topic):**
[Vizuara — LLM Interview Series #5: What Is PagedAttention?](https://www.youtube.com/watch?v=-AB6m0Spo6c)

In that video, the narrator explicitly says: do **not** jump to the definition.
Answer in layers:

1. **Basics / setup** — what must the listener already understand?
2. **Problem** — what breaks, with a concrete example?
3. **Root cause** — which design choice creates the failure?
4. **Solution** — only then name the method and walk the mechanism.
5. **Why it matters** — what improves in a real system?

This document applies that same narration to the full LLM lifecycle.
Claims stay with landmark, checkable sources. Where something is approximate or model-specific, it is marked.

---

## 0. The Explanation Contract (How Every Concept Below Is Told)

| Layer | Question answered |
|---|---|
| Setup | What world are we in? |
| Problem | What fails if we do the naive thing? |
| Root cause | Which assumption or design choice causes that failure? |
| Solution | What method addresses that root cause? |
| Payoff | Why practitioners use it |

If you only remember the name of a method, you have not explained it.

---

# Part A — Foundations

## A1. What is an LLM, really?

### Setup
People say “ChatGPT understands language.”
Under the hood, the deployed object is a parameterized function that maps a prefix of tokens to a distribution over the next token.

### Problem
Language is open-ended. Hand-written rules (grammar parsers + templates) do not scale to the variety of web text, code, dialogue, and instructions.

### Root cause
We need one objective that (i) uses abundant unlabeled text and (ii) produces a generative system.

### Solution — Autoregressive language modeling
Model a sequence by the chain rule:

\[
P(x_1,\ldots,x_T)=\prod_{t=1}^{T} P(x_t \mid x_{<t}).
\]

Train by minimizing next-token cross-entropy on large corpora.
At inference, sample or argmax next tokens one by one.

### Payoff
One scalable objective can absorb many downstream behaviors as “continue this prompt usefully.”
This is the standard framing used across GPT-style models and teaching lectures on building LLMs.

**Source anchors:** Brown et al., GPT-3 (2020); standard LM factorization.

---

## A2. Why Transformers?

### Setup
We need a neural architecture that estimates \(P(x_t\mid x_{<t})\) for long contexts and trains efficiently on accelerators.

### Problem
Older sequence models (RNNs) process tokens mostly sequentially; long-range dependencies and GPU parallelism are awkward at web scale.

### Root cause
The architecture must allow every position to gather information from other positions in parallel, with enough capacity to store patterns.

### Solution — Transformer (Vaswani et al., 2017)
Self-attention: each position builds Query/Key/Value projections; attention weights decide what to read; a feed-forward network transforms each position.
Decoder-only LLMs add **causal masking** so position \(t\) cannot see future tokens \(>t\) (required for next-token training and left-to-right generation).

### Payoff
Expressivity + hardware fit made Transformers the default LLM backbone.

**Source:** Vaswani et al., “Attention Is All You Need,” 2017.

---

## A3. Tokenization (why text is not fed as raw characters or whole words)

### Setup
Neural nets need discrete integer IDs. Text must become a sequence of token IDs.

### Problem
- Character-level: sequences become very long → expensive attention.
- Word-level: vocabulary explodes; rare words / typos / code break.

### Root cause
We need a middle ground: open vocabulary without pathological sequence length.

### Solution — Subword tokenization
Common practice in GPT-style models: **Byte-Pair Encoding (BPE)** / byte-level BPE — merge frequent symbol pairs into subwords.
Other families use WordPiece or Unigram/SentencePiece.

### Payoff
Any Unicode string can be encoded; frequent words stay short; rare pieces split.

---

## A4. Scaling laws (capacity × data × compute)

### Setup
Pretraining costs money. You must choose model size and number of training tokens under a FLOP budget.

### Problem
“Just make the model bigger” can waste compute if the model is undertrained on too few tokens (or the reverse).

### Root cause
Loss depends jointly on parameters, data, and compute—not on parameters alone.

### Solution — Empirical scaling laws
Kaplan et al. (2020) and Hoffmann et al. / Chinchilla (2022) fit how loss scales with size and tokens.
Chinchilla emphasizes compute-optimal **tokens-per-parameter** allocations: many early large models were undertrained relative to their size.

### Payoff
Guides pretraining budget decisions. Does **not** guarantee every downstream skill scales smoothly.

**Caution:** fits are empirical; data quality and mixture shift constants.

---

# Part B — Pretraining

## B1. Pretraining data

### Setup
Pretraining is next-token prediction on massive text (web, books, code, etc.).

### Problem
Raw crawls contain spam, boilerplate, duplicates, PII, and low-signal pages.
Duplicates also inflate memorization and can contaminate evaluation later.

### Root cause
Gradient updates follow the empirical data distribution. Garbage in → capability and safety issues out.

### Solution — Filter, deduplicate, mix
Modern open recipes (e.g., Llama-family reports) describe quality filtering, deduplication, and **domain mixtures** (web + code + books + …).
Synthetic rewritten data is sometimes used carefully; overuse risks distribution collapse.

### Payoff
Higher effective quality per token; better code/math if those domains are present.

---

## B2. What pretraining produces — and what it does not

### Setup
After pretraining you have a **base model**: a strong next-token predictor.

### Problem
Base models complete text. They do not reliably:
- follow user instructions,
- refuse unsafe asks,
- use a chat format,
- prefer helpful answers among many fluent ones.

### Root cause
The pretraining objective never saw an explicit “be a helpful assistant” reward—only “predict the next token on internet-like text.”

### Solution
Hand the baton to **post-training** (next part). Do not expect prompting alone to fully replace missing base competence, and do not expect pretraining alone to create chat alignment.

---

## B3. Modern architecture knobs (only after the problem is clear)

These are refinements, not a new definition of intelligence.

| Method | Problem it targets | Idea (compressed) |
|---|---|---|
| Pre-norm / RMSNorm | Deep-net training instability | Normalize inside blocks |
| SwiGLU / gated FFNs | Need more MLP quality per param | Gated activations |
| RoPE | Absolute position encodings transfer poorly to longer context | Rotary relative positions |
| GQA / MQA | KV cache bandwidth/memory at decode | Share KV heads |
| MoE (e.g., Switch, Mixtral) | Dense compute too expensive at scale | Activate sparse experts per token |

Explain each in interviews the same way: setup → failure without it → mechanism → payoff.

---

## B4. Training systems (why pretraining needs distributed tricks)

### Setup
A frontier run does not fit on one GPU: weights, optimizer states, activations, and long sequences all need memory.

### Problem
Naive data-parallel replication of full Adam states OOMs; single-device batch is too small; long context blows activation memory.

### Root cause
Memory and compute must be sharded across devices.

### Solution (standard toolkit)
- **AdamW** + warmup/decay schedules
- Mixed precision (BF16/FP16/FP8)
- Gradient checkpointing
- Data / tensor / pipeline parallelism
- ZeRO / FSDP (shard optimizer/grad/param state)
- Sequence/context parallelism for long context

### Payoff
Makes billion–trillion parameter training feasible.

---

# Part C — Post-Training

## C1. Supervised Fine-Tuning (SFT) / instruction tuning

### Setup
You have a base LM. Users want answers to instructions, not random web continuations.

### Problem
Ask a base model a question; it may continue as if writing a webpage, ignore the ask, or produce the wrong format.

### Root cause
Pretraining distribution ≠ assistant distribution.

### Solution — SFT
Collect \((\mathrm{prompt}, \mathrm{desired\ response})\) demonstrations and fine-tune with ordinary likelihood.
This is Step 1 of InstructGPT (Ouyang et al., 2022): supervised policy from human demonstrations.

Variants: chat multi-turn SFT, domain SFT, rejection-sampling then SFT (keep best of many samples).

### Payoff
Teaches format, tools schema, and basic instruction following.
Still insufficient when many answers are fluent but differently preferred.

**Source:** Ouyang et al., InstructGPT / NeurIPS 2022.

---

## C2. Classic RLHF (do not start with “PPO”)

### Setup
After SFT, the model often can produce multiple fluent answers. Humans still have preferences: more helpful, less toxic, more truthful.

### Problem
SFT clones demonstrations. It does not directly optimize “prefer A over B” when both are grammatical.
Also, writing one gold answer for every open-ended prompt is expensive and incomplete.

### Root cause
We need a **ranking signal**, then a way to optimize the policy against that signal without drifting into reward hacks.

### Solution — InstructGPT three steps (validated)
From Ouyang et al. (2022) / OpenAI instruction-following writeup:

1. **SFT** on demonstrations.
2. **Reward model (RM):** humans compare model outputs; train an RM to predict the preferred one.
3. **PPO:** optimize the policy to increase RM reward, with a **KL penalty** toward a reference policy to limit over-optimization.

### Payoff
Produced the first widely deployed instruction-aligned GPT-style assistants.
Costs: preference data, RM misspecification, reward hacking, multi-model training complexity (policy + value + RM + reference).

**Sources:** [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155); [OpenAI: Aligning language models to follow instructions](https://openai.com/index/instruction-following/).

---

## C3. RLAIF / Constitutional AI

### Setup
Human preference labeling does not scale forever.

### Problem
Safety/helpfulness labels are expensive; coverage of edge cases is thin.

### Root cause
Human feedback is the bottleneck, not only GPU time.

### Solution
Use AI feedback (RLAIF) and/or written principles (Constitutional AI: critique/revise under a constitution, then preference training) — Bai et al. line of work.

### Payoff
Scales labeling. **Does not** automatically remove teacher-model bias.

---

## C4. Direct Preference Optimization (DPO) — problem first

### Setup
RLHF works, but the stack is heavy: sample → RM → PPO with value model and KL control.

### Problem
Engineering instability and cost of full RLHF make iteration slow for many labs.

### Root cause
If preferences already define a Bradley–Terry ranking, the optimal KL-constrained policy can be expressed so that you may not need an explicit RM + PPO loop.

### Solution — DPO (Rafailov et al., 2023)
Train the policy directly on preference pairs with a classification-style objective that implicitly encodes the reward.

Related objectives exist when data or failure modes differ: IPO, KTO (unpaired thumbs-up/down style feedback), ORPO, SimPO, etc.
Explain those only after stating which RLHF pain point they target—do not dump acronyms.

### Payoff
Simpler alignment pipeline; widely adopted in open post-training.
Still depends on preference data quality.

**Source:** Rafailov et al., NeurIPS 2023.

---

## C5. Reasoning post-training (ORM / PRM / RLVR / GRPO)

### Setup
Chat alignment ≠ contest math / coding reliability.
Some answers need multi-step latent work.

### Problem
Outcome-only supervision is sparse: one bit at the end of a long solution.
Learned soft rewards can be hacked.
Full PPO needs a critic (value network), which is heavy.

### Root cause
Credit assignment and reward design for long reasoning traces are hard.

### Solution family (keep claims checkable)
- **ORM:** score final answer (good when verifiers exist).
- **PRM:** score steps (process supervision line; denser credit).
- **RL with verifiable rewards:** unit tests / exact checkers instead of a soft RM when possible.
- **GRPO** (introduced in DeepSeekMath; used in DeepSeek-R1): sample a **group** of outputs per prompt; use group-normalized rewards as advantages; **no separate value network**.
- DeepSeek-R1 recipe (high-level, from the paper): R1-Zero does RL with rule-based rewards on a base model; full R1 adds cold-start SFT and multi-stage training for readability/helpfulness.

### Payoff
Raises reasoning benchmarks when rewards are grounded.
Do **not** invent exact score tables—copy numbers from a specific paper revision if needed.

**Sources:** DeepSeekMath (Shao et al., 2024); DeepSeek-R1 (2025).

---

## C6. Parameter-efficient fine-tuning

### Setup
Full fine-tunes of large models are memory-heavy; you may need many task adapters.

### Problem
Storing a full copy of weights per task is impractical.

### Root cause
Task updates are often low-dimensional relative to the full weight matrix.

### Solution
LoRA (low-rank adapters), QLoRA (LoRA on quantized bases), classical adapters, prefix/prompt tuning, model merging.

### Payoff
Fine-tune on smaller GPUs; serve multiple adapters.

---

# Part D — Inference (where the Vizuara style shines)

## D1. Prefill vs decode (basics before KV cache)

### Setup
User sends a prompt; the model returns tokens.

### Problem people skip
They say “inference” as one blob. Serving engineers split it:

1. **Prefill:** process all prompt tokens (largely parallel) and produce the first output token.
2. **Decode:** generate one new token at a time, each step depending on the past.

### Root cause
Autoregressive factorization forces sequential decode even if prefill is parallel.

### Payoff of naming this split
Every later optimization (KV cache, continuous batching, PagedAttention) is about these two stages.

*(Same setup used in the [PagedAttention interview video](https://www.youtube.com/watch?v=-AB6m0Spo6c).)*

---

## D2. KV cache — problem before the name

### Setup
At each decode step, attention needs keys/values for all previous tokens.

### Problem (naive)
Recompute K/V for the entire prefix every new token → wasteful repeated matmuls; latency explodes with length.

### Root cause
Past tokens’ K/V do not change; only the new token’s K/V is new.

### Solution — KV cache
Store past keys and values in GPU memory; reuse them; append the new token’s K/V each step.
Trade: less compute, **more memory**.

### Payoff
Makes long decode practical.
New problem created: KV memory dominates serving (bridge to PagedAttention).

---

## D3. PagedAttention — full interview-style walkthrough

*Narration pattern mirrors [Vizuara #5](https://www.youtube.com/watch?v=-AB6m0Spo6c); mechanisms validated against the [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm) and Kwon et al., SOSP 2023.*

### Layer 1 — Basics: GPU memory during inference
During inference, GPU memory holds roughly:
1. **Model weights** (fixed after training),
2. **Activations** (ephemeral tensors: Q, K, V, etc.),
3. **KV cache** (grows with generated tokens; major during serving).

Inference runs on GPUs whether you call OpenAI-like APIs or self-host.

### Layer 2 — Problem: traditional KV allocation wastes memory
vLLM authors report existing systems can waste a large fraction of KV memory via fragmentation and over-reservation (blog: on the order of 60–80% waste in the regimes they studied).

Concrete multi-user story (as in the teaching video):
- Several users send requests.
- You do **not** know final decode length in advance.
- Naive approach: reserve a **contiguous** KV slab per request up to `max_tokens`.
- Failures:
  - **Reserved but unused** memory while a request is still generating (cannot give that RAM to another user).
  - After a short request finishes early, free holes may be **too small / wrongly placed** for a waiting large request.
- Result: low effective batch size → poor GPU utilization → low throughput.

### Layer 3 — Root cause
KV for a request is treated like one contiguous physical allocation (like requiring contiguous RAM for a process’s entire address space).
Variable lengths + contiguous reservation ⇒ fragmentation.

### Layer 4 — Solution: PagedAttention (+ vLLM)
Inspired by OS **virtual memory / paging**:

1. Split KV memory into fixed-size **blocks** (pages). Default block size in vLLM discussions is often **16 tokens** per block (block size is a system parameter).
2. Each block stores K/V for that many tokens.
3. Maintain:
   - a **free block list**,
   - a **block table** mapping each request’s logical token spans → physical blocks.
4. Allocate blocks **on demand** as decode proceeds; a request’s blocks need **not** be contiguous.
5. Attention kernel gathers K/V by following the block table (PagedAttention).

Additional capability from the same abstraction ([vLLM blog](https://vllm.ai/blog/2023-06-20-vllm)):
- Share blocks across sequences (e.g., parallel samples from one prompt) with reference counting / copy-on-write.

### Layer 5 — Payoff
Near-zero waste except partial last block (blog: under ~4% waste in their characterization) → larger batches → higher throughput.
Paper reports large throughput gains vs prior serving systems at similar latency; blog reports large gains vs HF Transformers in their benchmarks.
**Do not treat a single speedup number as universal**—it depends on model, GPU, and traffic.

**Sources:** Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023; [vLLM announcement blog](https://vllm.ai/blog/2023-06-20-vllm).

---

## D4. Decoding / sampling

### Setup
The model outputs logits → probabilities over the vocabulary.

### Problem
Always taking \(\arg\max\) can be dull/repetitive; sampling the full tail can be degenerate nonsense.

### Root cause
You need a **decision policy** on top of \(P(x_t\mid x_{<t})\).

### Solution
Temperature, top-\(k\), **nucleus / top-\(p\)** (Holtzman et al.), beam search (more classic MT than chat), etc.

### Payoff
Behavior changes without retraining.

---

## D5. Speculative decoding, quantization, continuous batching

Explain each with the same layers:

| Method | Problem | Root idea |
|---|---|---|
| Continuous batching | Static batches leave GPU idle as sequences finish at different times | Dynamically admit/finish requests |
| Speculative decoding | Large-model decode is serial and latency-bound | Small draft proposes; large model verifies in parallel |
| Quantization (INT8/INT4, GPTQ, AWQ, …) | Weights/KV blow VRAM; decode is memory-bandwidth heavy | Lower-bit tensors |
| Cascade / routing | Always calling the largest model is too costly | Send easy queries to smaller models |

---

## D6. Prompting, RAG, tools, test-time compute

### Setup
Weights are frozen at serve time (unless you fine-tune). Users still need better answers.

### Problem
Parametric memory is stale; multi-step tasks fail in one shot; exact arithmetic/API work is unreliable in pure text.

### Root cause
Next-token prediction alone is not a database, calculator, or search engine.

### Solutions (procedure changes, not weight changes)
- Zero-shot / few-shot (in-context learning; Brown et al., 2020)
- Chain-of-Thought (Wei et al., 2022)
- Self-consistency (Wang et al., 2022)
- Tree-of-Thoughts / search (Yao et al.)
- ReAct (reason + act with tools)
- RAG (Lewis et al., 2020): retrieve documents, then generate
- Tool / function calling + constrained decoding
- **Test-time compute:** spend more samples/search/“thinking” tokens on hard items (third pillar next to SFT and RLxF in post-training scaling surveys)

### Payoff
Higher quality per query at extra latency/cost → motivates adaptive compute and routing.

---

# Part E — Evaluation

## E1. Why evaluation is its own stage

### Setup
You changed data, loss, or decoding. Something on a dashboard moved.

### Problem
A single leaderboard number can rise while users get worse answers (or unsafe ones).

### Root cause
Benchmarks measure **narrow operationalized skills**, not “intelligence.”
Proxies can be gamed; contamination inflates scores; LLM-as-judge has biases.

### Solution — Portfolio evaluation
Capability suites (e.g., MMLU, GSM8K, HumanEval), instruction-following rubrics, human preference (Chatbot Arena Elo), safety suites, plus **cost/latency**.
Report threats: contamination, prompt sensitivity, judge bias, distribution shift.

### Payoff
Know whether the change was real.

---

## E2. Routing metrics (multi-model systems)

### Setup
You can call a weak/cheap or strong/expensive model per query (RouteLLM, Hybrid LLM, …).

### Problem
Reporting only accuracy hides the point of routing.

### Root cause
The objective is Pareto: quality vs cost.

### Solution metrics (as used in routing papers)
- Quality \(r(R)\)
- Cost \(c(R)\) (e.g., fraction of strong calls)
- \(\mathrm{PGR}=(r(R)-r(M_\mathrm{weak}))/(r(M_\mathrm{strong})-r(M_\mathrm{weak}))\)
- CPT(\(x\%\)): min strong-call fraction to hit target PGR

**Sources:** Ong et al., RouteLLM (ICLR 2025); Ding et al., Hybrid LLM (2024).

---

# Part F — End-to-End Story (one breath)

```text
Text in the world
  → filter/dedup/mix + tokenize
  → pretrain next-token LM (Transformer + distributed systems)
  → base model
  → SFT (teach the assistant protocol)
  → preferences (RLHF or DPO/…) / reasoning RL
  → aligned / reasoning model
  → serve: prefill/decode, KV cache, batching, PagedAttention, sampling
  → optionally: RAG, tools, more test-time compute, routing
  → evaluate capability + preference + safety + cost (watch contamination)
  → iterate
```

### Symptom → stage map

| Symptom | Likely stage |
|---|---|
| Cannot do basics even with good prompts | Pretraining data/capacity |
| Knows facts, ignores instructions | SFT |
| Fluent but rude/unsafe/sycophantic | Preference / safety post-training |
| Chat OK, hard math fails | Reasoning post-training or test-time compute |
| Correct but too slow/expensive | Inference systems / routing |
| Leaderboard up, users unhappy | Evaluation mismatch |

---

# Part G — Interview Cheat Sheet (same narration for any method)

When asked “What is X?”:

1. **I will not start with the definition of X.**
2. I explain the **system context** (training or serving).
3. I show the **naive approach** and where it breaks (numbers/examples if honest).
4. I name the **root design mistake**.
5. I introduce **X** as the fix and walk the mechanism.
6. I state **what improves** and **what tradeoff** appears.
7. I mention **one credible source**.

That is exactly how the PagedAttention video recommends answering—and how every concept in this document is organized.

---

# Part H — Landmark Sources (do not invent)

| Topic | Anchor |
|---|---|
| Transformer | Vaswani et al., 2017 |
| Few-shot / GPT-3 | Brown et al., 2020 |
| Scaling | Kaplan et al., 2020; Hoffmann et al., 2022 |
| InstructGPT / RLHF | Ouyang et al., 2022 |
| DPO | Rafailov et al., 2023 |
| CoT | Wei et al., 2022 |
| Self-consistency | Wang et al., 2022 |
| RAG | Lewis et al., 2020 |
| Nucleus sampling | Holtzman et al. |
| PagedAttention / vLLM | Kwon et al., SOSP 2023; [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm) |
| GRPO / R1 | DeepSeekMath 2024; DeepSeek-R1 2025 |
| RouteLLM | Ong et al., ICLR 2025 |
| Teaching style example | [Vizuara PagedAttention interview](https://www.youtube.com/watch?v=-AB6m0Spo6c) |

---

# Explicit Non-Claims

- No claim that one alignment method is universally best.
- No fabricated benchmarks or speedups beyond what cited sources state in their settings.
- Preference alignment does not “solve” safety.
- Agents = orchestration + tools + memory **on top of** this stack; not the same as pretraining.

---

*Files: `research/llm_end_to_end_survey.md` (narrative) and `research/llm_end_to_end_survey.tex` (LaTeX twin).*
