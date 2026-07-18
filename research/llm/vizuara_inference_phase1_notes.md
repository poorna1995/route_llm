# Vizuara Inference Book — Phase 1 Lecture Notes
## Study companion (problem-first narration)

**Aligned to curriculum:** Vizuara / Maven *LLM Inference Engineering* — Phase 1: Foundations & Optimization  
**Public course pages:** [inference.vizuara.ai](https://inference.vizuara.ai/) · [Maven workshop](https://maven.com/vizuara/inference-workshop)  
**Narration style:** same as [PagedAttention interview](https://www.youtube.com/watch?v=-AB6m0Spo6c) — Basics → Problem → Root cause → Solution → Payoff  

**Important honesty note.** These are **study notes** mapped to the Phase 1 outline you listed. They are **not** a transcript of paid Vizuara videos, Colabs, or private slides. Mechanisms below are validated against public papers/blogs (vLLM, FlashAttention, GQA, Mamba, GPTQ/AWQ, speculative decoding). Where the course goes deeper with proprietary visuals/labs, treat this as the conceptual scaffold—not a substitute for the recordings.

---

## Curriculum map (your Phase 1 outline)

| Block | Your label | What these notes cover |
|---|---|---|
| Book / intro | Inference Book Phase 1 (2) | Why inference is its own discipline |
| Mental model | Entropy (AI Mental Model #9) | Disorder → Shannon surprise → CE / perplexity / temperature / KL |
| Mental model | Diffusion (AI Mental Model #7) | Destroy gently → learn undo one step → generate from noise |
| Mental model | Expressivity (AI Mental Model #6) | If stuck, lift to a richer / higher-dimensional space |
| Mental model | Compression (AI Mental Model #5) | Too big → find the narrow waist / latent; work small; project up |
| Interview foundation | Transformer architecture (#3) | Input / processor / output + 8 modules in one block |
| Interview foundation | Attention (#4) | Why tokens need links → QKV → scores → causal → multi-head |
| Interview application | RAG vs fine-tuning (#8) | Grounded knowledge vs pattern/behavior change; when to choose |
| Hardware Labs | 6 lessons | Laptop, Pi, Android, edge bottlenecks |
| Visual Walkthroughs | 1 | How to use visuals while studying |
| L1 | Intro to Inference Engineering (4) | Prefill/decode, latency, throughput, memory |
| L2 | Good and Evil of KV Cache (5) | Cache math, wins, memory hell, paging bridge |
| Bonus | Inference and YCombinator (3) | Product/startup framing of inference |
| L3 | Attention Variants P1 (6) | MHA, MQA, GQA, latent, sparse |
| L4 | Attention Variants P2 (3) | Sliding window, SSM, Mamba |
| L5 | FlashAttention 1/2/3 (4) | IO-aware kernels |
| L6 | Anatomy of a vLLM Step (6) | Scheduler, PagedAttention, batching |
| L7 | Quantization (4) | FP16/BF16/INT8/INT4, GPTQ, AWQ, GGUF |
| L8 | Speculative Decoding | Draft–verify, EAGLE/Medusa family |

---

# Block 0 — Why “Inference Engineering” exists

### Setup
Training builds weights. Users experience **serving**: tokens out per second, dollars per million tokens, GPU OOMs at 2 a.m.

### Problem
A notebook `model.generate()` that feels fine for one prompt collapses under concurrent users, long contexts, and cost caps.

### Root cause
Inference is autoregressive, memory-bandwidth heavy, and multi-tenant. Training tricks (huge activation checkpointing, different batching) do not automatically solve decode.

### Solution discipline
Treat inference as its own stack: hardware → kernels → attention/KV design → scheduler → quantization → speculation → routing.

### Payoff
You can answer the interview prompt Vizuara highlights: *design a low-latency, high-throughput LLM serving system and walk the trade-offs* ([course framing](https://inference.vizuara.ai/)).

---

# Hardware Labs (6 lessons) — bottlenecks change with the device

Do **not** memorize “INT4 is always better.” Lab goal: **measure** where you are stuck.

| Lab (typical Phase 1) | Device | What usually bottlenecks | What you practice |
|---|---|---|---|
| Lab 1 | Laptop / PC | CPU mem bandwidth or Apple GPU/Neural Engine vs discrete GPU | `llama.cpp` / MLX; tok/s vs model size |
| Lab 2 | Raspberry Pi 4 (ARM) | Weak CPU + RAM; no big GPU | INT4 vs INT8 latency/quality; power |
| Lab 3 | Android | On-device RAM + thermal throttling | SmolChat-style phone deploy |
| Optional | Jetson Orin Nano | Small CUDA GPU + power envelope | TensorRT-LLM demos when available |
| + extras in “6 lessons” | Setup, benchmarking harness, compare tables | Methodology | Always report: model, quant, context, batch, tok/s, RSS/VRAM |

### Setup → Problem → Root cause (hardware)
- **Setup:** Same 7B weights, different silicon.  
- **Problem:** Numbers from an A100 blog post do not transfer to a Pi.  
- **Root cause:** Roofline changes — compute-bound vs memory-bound depends on device + batch + sequence length.  
- **Solution:** Benchmark matrix per device; choose quant/serving stack for *that* roofline.  
- **Payoff:** Real deployment intuition, not cloud-only folklore.

*(Hardware list from public workshop pages: laptop, Pi 4, Android, optional Jetson.)*

---

# All Visual Walkthroughs (1) — how to study them

When watching a Vizuara visual:

1. Pause before the “solution” slide — state the problem in one sentence.  
2. Name the **naive approach** and where it breaks.  
3. Only then watch the mechanism.  
4. Redraw the diagram from memory (KV growth, block table, FlashAttention tiles).  
5. Write one interview answer aloud (60–90 seconds).

---

# Mental Model Bridge — Entropy (before L1)

**Video:** [Entropy: AI Mental Model #9](https://www.youtube.com/watch?v=q3agYqgqklU) (Vizuara)  
**Free visual notes:** [lecture-09-entropy.html](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-09-entropy.html)

This is not an “inference kernel” lecture. It is the **shared mathematics** under pretraining loss, sampling temperature, and RLHF KL penalties. Learn it before optimizing decode.

## E.0 How to explain entropy (same story discipline)

Do **not** open with “entropy is \(-\sum p\log p\).”
Tell the story the video tells:

1. Everyday disorder (messy room, cream in coffee).  
2. **Count of arrangements** (formal meaning of “messiness”).  
3. Physics: heat = molecular jiggling → more microstates → higher entropy; Second Law (arrow of time).  
4. Boltzmann: \(S = k \log W\).  
5. Shannon: information = **surprise** \(=-\log p\); entropy = **expected surprise**.  
6. Only then: where AI uses the same idea.

### Core reframing (from the lecture)
Entropy is not a vibe of “messiness.” It **counts how many ways** a situation can look the same from the outside.
Low entropy ≈ few arrangements (tidy / peaked / certain).  
High entropy ≈ many arrangements (messy / flat / uncertain).

---

## E.1 Physics side (setup for the bridge)

| Idea | One-line meaning |
|---|---|
| Temperature | Average molecular speed / jiggling |
| High heat | More ways molecules can arrange → higher entropy |
| Ink in water | One “tight drop” vs countless “spread out” states |
| Second Law | Closed systems tend toward higher entropy |
| Boltzmann | \(S = k\log W\) — log of number of microstates |

Dice check: sum=7 has more microstates than sum=2 → more likely. “Most likely” often = “most ways.”

---

## E.2 Shannon side — information as surprise

### Setup
Shannon (1948) needed to measure how much information a message carries on a noisy channel.

### Problem
“The sun rose” vs “it snowed in the desert” are not equally informative.

### Root cause
Common events are unsurprising; rare events are surprising.

### Solution
\[
\text{surprise}(x) = -\log p(x)
\]
Entropy of a distribution = average surprise across outcomes.
Fair coin → high uncertainty (1 bit). Biased coin → lower average surprise.

---

## E.3 Where entropy shows up in LLM / ML (the payoff for this course)

### (1) Cross-entropy loss = training surprise
Classifier / next-token LM loss: punish the model when it assigns **low probability to the true label/token**.
If the true next token has model probability \(p_{\text{true}}\):
\[
\mathcal{L} = -\log p_{\text{true}}
\]
(Same surprise formula.)  
Pretraining an LLM is largely minimizing this next-token cross-entropy.

### (2) Perplexity = “how many doors am I choosing among?”
Perplexity ≈ effective branching factor of the model’s uncertainty
(video’s English: perplexity 2 ≈ coin-flip unsure; perplexity 50 ≈ ~50 doors).
**Lower perplexity ⇒ more confident next-token predictions** (on that data).
Use it as an intrinsic LM metric; it is not the same as chat quality.

### (3) Sampling temperature = physics temperature analogy
Softmax with temperature \(T\):
- **Low \(T\)** → peaked distribution → low entropy → precise / deterministic.  
- **High \(T\)** → flatter distribution → high entropy → more random / “creative” / riskier.

Video’s bridge: physical temperature spreads molecules across arrangements;
sampling temperature spreads probability across vocabulary tokens.
That is why the knob is literally called **temperature**.

### (4) KL divergence = relative entropy
Extra surprise from using the wrong distribution instead of the true/reference one.
Shows up in:
- RLHF / PPO (**KL penalty** keeps the policy near a reference model),
- VAEs, diffusion objectives,
- knowledge distillation.

English: if your belief matches reality, extra surprise is ~0; the more wrong, the more bits you “waste.”

### (5) Decision trees (side note)
Splits maximize information gain = entropy reduction (chaos → order in the labels).

---

## E.4 Interview / study one-liner

> Whenever the problem is uncertainty, surprise, confidence, randomness, or distance between beliefs — reach for entropy.
> In LLM land that means: cross-entropy trains the model, perplexity scores uncertainty, temperature spends entropy at decode, KL limits drift in RLHF.

### Link forward into Phase 1
- L1 decode/sampling → temperature is an entropy dial.  
- Pretraining discussion → CE / perplexity.  
- Later alignment (outside Phase 1 kernels, but same stack) → KL to reference.

---

# Mental Model Bridge — Diffusion / “Reverse the Corruption”

**Video:** [Diffusion: AI Mental Model #7](https://www.youtube.com/watch?v=bTEfdB2D1Ek) (Vizuara)  
**Free visual notes:** [lecture-07-reverse-the-corruption.html](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-07-reverse-the-corruption.html)

Not a vLLM lecture. It is the **generation mental model** behind Stable Diffusion / Midjourney-class image systems—and a contrasting paradigm to autoregressive LLM decode (L1).

## D.0 How to explain diffusion (story first)

Do **not** open with “DDPM / score matching.”
Tell the video’s story:

1. Naive alien idea: teach AI to *draw* strokes from scratch.  
2. Actual trick: **to create, first learn to destroy** — then run destruction backward.  
3. Forward process: gently corrupt data into noise (and remember what noise was added — “breadcrumbs”).  
4. Train a network to undo **one tiny step** (often: predict the noise that was added).  
5. At generation: start from **pure noise**, repeatedly ask “what noise should I remove?”, get a sample.  
6. Why it works: one hard leap → many easy reverse steps chained.  
7. Where it shows up: images, video, audio, molecules — and increasingly **language**.

**One-line mental model:**  
When one-shot creation is too hard, define a *gradual reversible corruption*, learn to undo a single step, run that undo from pure noise.

---

## D.1 Forward process (noising)

### Setup
You have clean data \(x_0\) (image, later: text structure, etc.).

### Problem
Teaching “paint a masterpiece in one forward pass from nothing” is brutally hard.

### Root cause
The clean ↔ noise map is complex; jumping the whole gap at once is unstable.

### Solution
**Forward diffusion:** add a little noise again and again until the sample is (nearly) pure noise.
Critical teaching point from the video: at each step you **know** what noise was added (traceable path back).

---

## D.2 Reverse process (denoising / learning)

### Setup
You have noisy samples at many noise levels.

### Problem
You need a machine that can walk back toward clean data.

### Solution
Train a model (classically U-Net-style for images) that, given a noisy input at some step, predicts the noise (or the clean signal / velocity — formulation variants exist).
Learning target = **undo one small corruption**, not invent a full image in one shot.

---

## D.3 Generation = start from chaos

1. Sample pure noise (never was a real image).  
2. Ask the model what to remove at this step; subtract / update.  
3. Repeat for many steps → structure emerges.

Video’s fluid analogy: slow stirring (gentle corruption) can be reversed; destroy **slowly** or reverse fails. That is why the name ties to physical **diffusion**.

---

## D.4 Contrast with autoregressive LLM inference (ties to L1)

| | Autoregressive LLM (ChatGPT-class) | Diffusion LM (emerging) |
|---|---|---|
| Generation order | Left-to-right, **one token at a time** | Iterative **denoise / unmask**; tokens can appear **in parallel** |
| Starts from | Prompt prefix | Noise / full mask (formulation-dependent) |
| Latency shape | Serial decode steps | Fewer coarse iterations possible (workload-dependent) |
| Dominant today | Almost all chat APIs | Research + early products (video cites examples; treat as evolving) |

**Phase 1 implication:** when someone says “inference,” ask *which generator family*.
KV cache, GQA, PagedAttention, speculative decoding are primarily about **autoregressive** decode.
Diffusion LMs change the bottleneck story (parallel refinement vs serial tokens).

Do **not** invent product claims (e.g. “Gemini ships X”). Stick to: AR is dominant; diffusion-for-language is an active alternative paradigm (Karpathy-style contrast: not left-to-right, all-at-once refinement).

---

## D.5 Where the same trick spreads

| Domain | Corruption → reverse idea |
|---|---|
| Images | Pixel/latent noise → denoise (Stable Diffusion family, etc.) |
| Video | Space–time noise → denoise (Sora-class systems use diffusion ideas) |
| Audio / speech | Corrupt waveform/latents → denoise to speech |
| Molecules / drugs | Scatter atoms → denoise to valid structure |
| Language | Mask/noise tokens → iterative demask/denoise |

---

## D.6 Interview / study one-liner

> Creation too hard in one shot? Define a gentle, reversible way to destroy the finished thing; train a model to undo one step; run that undo from pure noise.
> That is diffusion — destruction played backward.

### Link to Phase 1
- Clarifies why L1–L8 focus on AR decode machinery.  
- Prepares you for exam/interview questions that contrast AR vs diffusion generation.  
- Entropy lecture (temperature / distributions) still applies inside each denoising step’s probabilistic choices.

---

# Mental Model Bridge — Expressivity (“If you cannot solve it, lift it”)

**Video:** [Expressivity: AI Mental Model #6](https://www.youtube.com/watch?v=YE0udJCgDqc) (Vizuara)  
**Free visual notes:** [lecture-06-expressivity.html](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-06-expressivity.html)

Companion opposite of **compression / latent squeeze**: compression finds essence by going *down*; expressivity finds patterns by going *up* into a bigger space.

## X.0 How to explain expressivity (story first)

Do **not** open with “universal approximation.”
Tell the video’s story:

1. Stuck in a low-dimensional representation → patterns invisible.  
2. Classic picture: **XOR** not linearly separable in 2D; lift red points in height → plane separates in 3D.  
3. Same idea as the **kernel trick**: lift features so a simple separator works.  
4. Regression: higher-degree polynomials = more expressivity (and overfitting risk).  
5. LLMs: scale / width creates room for capabilities; inside the block, **FFN expands** (~4×) then projects back.  
6. Also: CNNs (stack of feature maps), NeRF (Fourier feature lift of coordinates).  
7. Warning: too much expressivity → fit noise → poor generalization.

**One-line mental model:**  
If the problem won’t yield, you may be in too few dimensions — **lift it** so the pattern has room to appear. Then find the sweet spot (not underfit, not overfit).

---

## X.1 XOR / kernel picture (the visual to keep)

### Setup
Two classes that no straight line can separate in 2D.

### Problem
Linear model in that space is *structurally* insufficient — not just undertrained.

### Root cause
The representation is not expressive enough for the decision boundary you need.

### Solution
Add dimensions (raise some points on a new axis / kernel map). Same data, richer coordinates → linear separator exists.

### Payoff
“Impossible” becomes “easy” after a lift — this is the intuition behind many nonlinear methods.

---

## X.2 Regression and the overfitting warning

Degree-0 (constant) → too little expressivity.  
Higher degree → can follow richer curves.  
**Too high** → wiggles through noise (high variance).

Interview question when a model fails:  
*Is it expressive enough to represent what I’m asking it to learn?*  
If yes and it still fails: data, optimization, or regularization — not only “add layers.”

---

## X.3 Where this shows up in Transformers / LLMs

| Place | Expressivity move |
|---|---|
| Model scale | More params / width / depth → capacity for harder patterns (empirical scaling / emergents — don’t overclaim magic) |
| **FFN / MLP block** | Per-token expand to a **wider** hidden size (often ~4× `d_model`), nonlinearity, project back |
| Why FFN is huge | Video framing: attention moves information across tokens; FFN is where the model is **wide enough to store/transform patterns** (related research view: FFN as key–value memories — cite Geva et al. if you name it) |
| MoE (later / L3) | Conditional extra capacity — more expressivity without always paying dense FLOPs |

### Setup → problem for FFN specifically
Attention mixes context across positions.  
If you only had narrow per-token maps, capacity to *hold* associations is limited.  
Expand–contract FFN buys a wide temporary workspace per token.

---

## X.4 Other domains (same lift)

| Domain | Lift |
|---|---|
| CNNs | Image → stack of many feature-map channels |
| Classical NNs | Low-dim inputs → wide hidden layers for hard functions |
| NeRF | XYZ → Fourier (sin/cos at many frequencies) so high-frequency detail is representable |

---

## X.5 Pair with compression (and with Phase 1)

See the next section (**Compression**, Mental Model #5) for the full squeeze story.
Short pairing table:

| Mental model | Direction | When you reach for it |
|---|---|---|
| Compression | Down through a narrow waist | Too much memory / compute / pixels / ΔW |
| Expressivity | Up to richer space | Structure invisible; under-capacity model |
| Phase 1 inference | Often *reduce* bytes (GQA, MLA, quant, paging) | Serving cost — often *is* compression in disguise |

Serving optimizations assume a capable model already exists.  
Expressivity answers *why the FFN is fat*; Compression + Phase 1 answer *how to not pay full price for bigness*.

---

## X.6 Interview / study one-liner

> If you can’t solve it, lift it: richer features / wider layers / more capacity so the pattern has room — then stop before you fit noise.

---

# Mental Model Bridge — Compression (“When it’s too big, find the waist”)

**Video:** [Compression: AI Mental Model #5](https://www.youtube.com/watch?v=hGS6RbAYLl0) (Vizuara)  
**Free visual notes:** [lecture-05-compression.html](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-05-compression.html)

Natural pair with Expressivity (#6): compress = go **down**; express = go **up**.

## C.0 How to explain compression (story first)

Do **not** open with “autoencoder loss.”
Tell the video’s story:

1. Something is **enormously big** (pixels, KV, ΔW, game frames).  
2. Force it through a **narrow waist** into a small space.  
3. Do the hard work **there**.  
4. **Project back up** when you need the big object.  
5. Shock: crushing a million numbers to hundreds is often *not* catastrophic — the big thing was never full-rank / full of independent facts.  
6. Same gesture in BPE, latent diffusion, MLA, LoRA, world models.

**One-line mental model:**  
When it’s too big, don’t accept the bigness — ask where the **narrow waist** is and how few numbers really carry the information.

---

## C.1 Why little is lost (manifold intuition)

### Setup
A face photo has ~millions of pixels; a weight matrix is \(D\times D\); a squiggle is 80 samples.

### Problem
Naively, compressing should destroy meaning.

### Root cause
Real signals live on a **lower-dimensional manifold** inside the ambient space (neighbors predictable; rows of weights nearly low-rank). A blurred road photo still reads as “cars on a road.”

### Solution intuition
Keep the few latent degrees of freedom that rebuild the structure; discard axes that were never independent.

Video’s reconstruction demo: storing a small fraction of coefficients can rebuild a curve with low error — the pattern Phase 1 methods exploit.

---

## C.2 Five places the same gesture appears

| Domain | Big thing | Narrow waist | Project back |
|---|---|---|---|
| **Tokenization (BPE)** | Raw character sequences | Merged subword codes (shorter codes for common pieces) | Detokenize to text |
| **Stable Diffusion** | High-res pixel space (expensive to denoise) | Latent image from VAE encoder | Image decoder |
| **MLA (DeepSeek-style)** | Full K and V per head/token in KV cache | One compressed **latent** vector stored | Expand latent → K,V when attending |
| **LoRA** | Full fine-tune ΔW (\(D\times D\)) | \(A\,(D\times r)\), \(B\,(r\times D)\), \(r\ll D\) | \(ΔW \approx AB\) at apply time |
| **World models** | Raw game/screen pixels every frame | Tiny latent per frame (autoencoder) | Agent “dreams” / controls in latent space |

### MLA vs GQA (don’t confuse — L3)
- **GQA:** share K/V across *groups of query heads* (still store K,V — fewer heads).  
- **MLA:** compress what is stored into a *latent* then expand (compression mental model).  
Different mechanisms; both shrink KV pain.

### LoRA vs full FT
Freeze giant \(W\); train small \(A,B\) per task; store many adapters cheaply; compose at serve time.

### Latent diffusion vs pixel diffusion
Denoise in latent (cheap); decode once to pixels — compression makes the Diffusion mental model affordable.

### World models
Agent acts on a **compressed dream** of the world, not raw frames — same “few numbers suffice” bet.

---

## C.3 Interview answer skeleton

1. Name the bigness (memory / compute / pixels / parameters).  
2. Ask: what is the thin manifold / low-rank structure?  
3. Pinch through a waist; work small; lift back.  
4. Give **one** concrete example matching the interview domain (KV→MLA/GQA, FT→LoRA, images→latent diffusion).  
5. Warning: waist too narrow → irreversible information loss / quality cliff (like over-quantizing).

---

## C.4 Pairing table (Compression ↔ Expressivity ↔ Phase 1)

| If the failure is… | Reach for… |
|---|---|
| Can’t represent the pattern | **Expressivity** (lift / widen) |
| Can represent it but too expensive | **Compression** (waist / latent / low-rank / quant / GQA) |
| Multi-tenant AR serving | Phase 1 stack (often compression + systems: paging, batching, speculation) |

---

## C.5 Interview / study one-liner

> When it’s too big, find the waist: squeeze into a small latent, do the work there, project back — because the giant object was never as high-dimensional as it looked.

---

# Foundation Interview — What Is the Transformer Architecture?

**Video:** [LLM Interview Series #3: What Is the Transformer Architecture?](https://www.youtube.com/watch?v=c533te7NSpI) (Vizuara)

Do **not** recite “attention + FFN, done.”
Lead with the big picture, then go layer by layer ([video](https://www.youtube.com/watch?v=c533te7NSpI)).

## T.0 Scope for chat / decoder-only LLMs

Original “Attention Is All You Need” had **encoder + decoder**.
Modern chat models (ChatGPT / Claude-class) are **decoder-only**: generate the next token left-to-right, so the interview answer should focus on the **decoder stack**, not a full seq2seq diagram unless asked.

---

## T.1 Depth 1 — Whole LLM = three blocks

| Block | Role |
|---|---|
| **Input** | Tokens → token embeddings + positional info → **input embeddings** |
| **Processor** | Stacked **Transformer blocks** (this *is* the architecture) |
| **Output** | Still in embedding dim → **logits** over vocabulary → next-token choice |

Place the Transformer in the **processor**. Ground it with input and output so the interviewer sees the full path from “Give me a travel plan for Spain” to the next token.

**Pretrain vs inference (same diagram):**
- Pretrain: embeddings / \(W_Q,W_K,W_V\) / FFN weights are **learned**.
- Inference: those weights are **fixed**; forward path is the same shape.

---

## T.2 Depth 2 — Stack many identical blocks

One block’s output feeds the next: \(T_1 \to T_2 \to \cdots \to T_L\) (dozens to 100+ layers in large models).
**Composable stacking** is a reason the architecture scaled.

Innovations cluster on two modules inside each block:
- **Attention** variants: MQA, GQA, MLA, … (L3)
- **FFN** variants: MoE, SwiGLU, … (Expressivity / Compression)

---

## T.3 Depth 3 — Eight modules inside one block (video order)

Typical decoder block as drawn in the lecture:

1. **LayerNorm** (or RMSNorm in many modern models)  
2. **Multi-head attention**  
3. **Dropout**  
4. **Residual / skip** (add input of the sublayer)  
5. **LayerNorm** (or RMSNorm)  
6. **Feed-forward network (MLP)**  
7. **Dropout**  
8. **Residual / skip**

Exact norm placement (pre-norm vs post-norm) varies by family — say what you assume if grilled.

### What each piece is *for* (interview meaning)

| Module | First-principles job |
|---|---|
| **LayerNorm / RMSNorm** | Stabilize activations / training; RMSNorm is a simpler modern variant (scale by RMS, fewer centering params) |
| **Multi-head attention** | **Only place tokens talk to each other** — mix neighbor information into **context vectors**. Causal mask: no looking at future tokens (next-token prediction must not cheat) |
| **Dropout** | Randomly zero units in training so the net doesn’t rely on a few “lazy” pathways → better generalization (classic DL; not LLM-specific) |
| **Residual / skip** | Alternate path for gradients → fight **vanishing gradients** in deep stacks (\(O+I\) instead of only deep multiplies) |
| **FFN / MLP** | **Per-token** expand (often ~\(4\times d\)) → nonlinearity (GELU / SwiGLU) → contract. Tokens do **not** mix here. Reason: **expressivity** — patterns that won’t fit in narrow \(d\) (XOR / lift story) |

**Critical contrast to memorize:**  
Attention = inter-token communication.  
FFN = intra-token wide transform.  
Without attention, “Harry … he” never binds — the model stays locally blind.

---

## T.4 Depth 4 — Inside attention (enough for this question)

When asked to go one level deeper on module 2:

1. \(X \rightarrow Q,K,V\) via \(W_Q,W_K,W_V\) (trained in pretrain; frozen at inference).  
2. Scores \(\propto QK^\top\); scale by \(\sqrt{d_k}\); causal mask; softmax → weights.  
3. Context \(\propto\) weights \(\times V\).  
4. Multi-head: split into heads, each with its own scores/perspective; concat.

Full KV-cache story is Interview [#1](https://www.youtube.com/watch?v=CxRGWfcGVbs) (L2); GQA is [#6](https://www.youtube.com/watch?v=mtsY7JsGQjw) (L3).

---

## T.5 Modern knobs the video flags (don’t invent details)

| Older textbook block | Common modern twist |
|---|---|
| Absolute positional embeddings before the stack | **RoPE** applied inside attention |
| LayerNorm | **RMSNorm** |
| GELU MLP | **SwiGLU** (gated; extra projection matrix) |
| Dense FFN | **MoE** routing (research/product variants) |

Mention as “open research / family-specific,” not as universal law.

---

## T.6 How to deliver the answer (video’s interview advice)

1. Bird’s-eye: input → processor (stack) → logits.  
2. One block: list the eight modules with **why**.  
3. Zoom attention: QKV → scores → causal → context.  
4. Name where innovation lives (attention + FFN).  
5. Show passion / first principles — don’t stop at depth 1.

### 60-second script
“Decoder-only LLM: embeddings in, stacked Transformer blocks, logits out. Each block is norm → causal MHA (tokens mix) → residual → norm → wide FFN (per-token expressivity) → residual. Attention is the only inter-token path; FFN lifts width then contracts. We stack many identical blocks. Variants like GQA/MLA/MoE change attention or FFN; serving then caches \(K,V\) at decode.”

---

# Foundation Interview — What Is Attention?

**Video:** [LLM Interview Series #4: What is Attention?](https://www.youtube.com/watch?v=cquX2tOODUI) (Vizuara)

Do **not** open with “the model focuses on important tokens.”
Answer at **two depths**, then stop so the interviewer asks follow-ups ([video](https://www.youtube.com/watch?v=cquX2tOODUI)).

## A.0 Place attention in the bird’s-eye view first

Same three blocks as Interview [#3](https://www.youtube.com/watch?v=c533te7NSpI):
**Input** (token + position → embeddings) → **Processor** (Transformer stack; attention sits inside each block after the first norm) → **Output** (logits).

Only then zoom into the attention module.

---

## A.1 Depth 1 — Conceptual: the problem attention solves

### Setup
Language is full of long-range links:  
`Harry boarded the train. **He** …` — who is *he*?  
Talk about India, later ask “capital of **this country**.”  
Code: “change the first 20 lines” — need earlier context.  
`The dog chased the ball. **It** could not catch **it**.` — first *it*→dog, second *it*→ball.

### Problem
If every token were processed in isolation, the model would have **no clue about its neighbors**. Context would not exist.

### Root cause
Next-token modeling needs **links** between a token and earlier tokens, plus **how much** to weight each earlier token.

### Solution (in words)
Attention **captures links between tokens**: for a given query token, which past keys matter, and with what strength.
Humans do something analogous: keep a working context, not a perfect transcript of every word.

**One-liner:** Without attention, tokens are islands; with attention, each token can read a weighted view of its past.

---

## A.2 Depth 2 — Matrix level: Q, K, V

### Setup
Input embeddings \(X\) (e.g. 3 tokens × \(d\)). Frozen at inference; trained in pretraining.

### Mechanism (write this on the board)
1. \(Q = X W_Q,\quad K = X W_K,\quad V = X W_V\).  
2. **Scores** \(S = Q K^\top\) (per-row view: each query vector dotted with every key).  
3. Interpret \(S\): row \(i\), column \(j\) = how much token \(i\) attends to token \(j\).  
4. **Causal mask:** upper triangle blocked — no future tokens (past causes present; future does not).  
5. Scale (typically \(/\sqrt{d_k}\)), **softmax** → **attention weights**.  
6. Context \(=\) weights \(\times V\) — each input vector becomes a **context vector** mixed from values.

Batch form \(QK^\top\) is just all per-token score rows stacked — know both the row story and the matrix story.

### What “attention weights × values” means
Scores decide **where** to look; values carry **what** content is mixed into the new representation.

---

## A.3 Multi-head (when they ask the fork)

Single-head = one score matrix = one perspective.  
Multi-head: split \(W_Q,W_K,W_V\) into heads → multiple score matrices of the **same** spatial size → concat contexts.

**Why:** one sentence can need several bindings at once  
(*artist painted … woman with a brush* — brush with artist vs brush with woman).  
Each head can specialize in a different pattern.

Full GQA/MQA tradeoffs: Interview [#6](https://www.youtube.com/watch?v=mtsY7JsGQjw) (L3).  
Serving reuse of \(K,V\): Interview [#1](https://www.youtube.com/watch?v=CxRGWfcGVbs) (L2).

---

## A.4 How to lead the interview

1. Bird’s-eye placement.  
2. Conceptual need (Harry/he, dog/ball).  
3. QKV → \(QK^\top\) → causal → softmax → \(\times V\).  
4. Stop; let them ask softmax / multi-head / KV cache / GQA.  
5. Prefer an example you thought through yourself — signals real understanding.

### 60-second script
“Attention is how tokens get context: without it they don’t know their neighbors. For each position we form Q, K, V; scores are \(QK^\top\) — how much this query matches each past key; causal mask blocks the future; softmax turns scores into weights; weighted values become the context vector. Multi-head repeats that with different projections so several linguistic perspectives can coexist.”

---

# Application Interview — RAG vs Fine-Tuning

**Video:** [LLM Interview Series #8: RAG vs Fine-Tuning](https://www.youtube.com/watch?v=cCXjumE70-g) (Vizuara)

This is a **system-design** interview question, not an inference-kernel question. Interviewers are testing whether you can choose an approach for a vague industry request (pharma / legal / nutrition chatbot) from first principles—not recite two definitions.

Do **not** open with “RAG is for knowledge, fine-tuning is for behavior” and stop. That is a start; depth is the decision axes + examples.

## R.0 Interview trap

You may know RAG and fine-tuning **in isolation**. The hard part is connecting them: *for this product, which one (or both), and why?*

Vague client: “Build a chatbot for our domain.”  
Strong answer: clarify whether they need **grounded lookup** of owned documents, or **new behavior / style / pattern** that isn’t literally written as a passage to retrieve.

---

## R.1 What problem does RAG solve? (before the acronym)

### Setup
Nutrition / fitness company has ~2,000 curated resources (their IP). Customers ask “what should I eat for breakfast?” Answers must be **grounded** in those resources—not a generic web opinion.

### Naive solution (and why it fails)
Stuff **all** resources into the prompt every turn.

### Problem
Corpus can be huge (video’s order-of-magnitude: tens–hundreds of millions of tokens). Context windows are large but still **limited** (e.g. ~1M ≪ 100M). You cannot fit the whole library.

### Root cause
Context is a scarce resource (“like land”). Only a **slice** of the corpus is relevant per question.

### Solution — Retrieval-Augmented Generation
1. Chunk the corpus (paragraphs / sections — a design choice).  
2. Embed chunks → vectors; embed the query.  
3. Retrieve top-\(k\) chunks by similarity (e.g. dot product / nearest neighbors).  
4. Put **query + retrieved chunks** into the LLM context; generate.

Weights stay **fixed**. Name: **retrieval** augments **generation**.

**Production caveat (video):** embedding + storing vectors is expensive in compute and disk; people use quantization/compression for embedding stores. “Vectorless RAG” (tree-style) exists as a fork—don’t invent details unless asked.

### Analogy
**Open-book exam:** you don’t memorize the whole book at test time; you flip to the relevant pages, then write the answer with the book beside you.

### When you do *not* need RAG
If the owned knowledge **fits** in context with margin, you can pass it directly. RAG exists because context is limited and relevance is sparse.

---

## R.2 What is fine-tuning? (start from pretraining)

### Setup
A pretrained LM is billions of “knobs” fitted on internet-scale data. It has absorbed facts/patterns, but a **raw pretrained** checkpoint is often weak at following instructions (“convert 35 km to meters”, “active → passive”).

### Problem
You need the model to behave well on a **narrower** objective (instruction following, domain QA style, persona) without redoing full pretraining.

### Solution
Curate a **much smaller** supervised set of (question, answer) pairs (video’s ballpark: order of ~10k, not pretraining scale). Train again so weights **nudge**. That is fine-tuning: take a pretrained model, adjust parameters on labeled task data.

**Key contrast with RAG:** fine-tuning **changes weights**; RAG does **not**.

### Why small data can work (pointer only)
Video links this to **intrinsic dimensionality**: pretraining already moved the model into a useful lower-dimensional region, so less data can reorient behavior. Don’t claim a specific theorem unless you know it—say “small supervised nudge on a strong pretrained base.”

---

## R.3 When to choose which (decision examples)

### Example A — Digital clone of a person (fine-tuning)
User dumps LinkedIn / YouTube / Instagram presence; you want a chatbot that **sounds like them** on *new* questions not literally in the posts.

| Approach | What happens |
|---|---|
| RAG | Retrieves relevant posts; does **not** teach speaking patterns (pauses, assertiveness, diplomacy) for novel asks |
| Fine-tuning | Adjusts weights so the model internalizes **patterns** of how they speak |

→ Prefer **fine-tuning** when the goal is pattern / persona / “intelligence style,” not document lookup.

### Example B — Pharma / nutrition grounded chatbot (start with RAG)
Answers must cite / stay inside owned documents. You don’t need to invent a new speaking style; you need **relevant passages** at query time.

→ Prefer **RAG** first: faster, cheaper, simpler than training. Caveat: RAG only surfaces what’s (near) in the store; it does not discover deep new patterns.

### Practical escalation (video’s project advice)
```text
easiest → harder
RAG  →  fine-tune pretrained model  →  train SLM from scratch (pretrain)
```
Start left; move right only when RAG fails to capture needed patterns. Don’t jump to full pretraining unless necessary.

---

## R.4 Comparison axes (draw this table)

| Axis | RAG | Fine-tuning |
|---|---|---|
| Finds patterns in data? | No — retrieves existing text | Yes — weights encode patterns |
| Cost | Lower | Higher (train + maintain adapters/checkpoints) |
| Simplicity | Higher (index + retrieve + prompt) | Lower (data, train, eval, deploy new weights) |
| Personalization / “clone” feel | Weak | Strong when trained for it |
| Analogy | Open-book exam (book at test time) | Study the night before (rewire brain; no book in the room) |
| Weights at serve | Unchanged | Changed (or LoRA/adapters applied) |

Freshness / control (interview extras you can add carefully):
- **RAG** wins when knowledge updates often (re-index docs; no retrain).  
- **Fine-tuning** wins when you need stable style, format, or domain *behavior* that prompting+retrieval doesn’t stick.

Many real systems do **both**: fine-tune (or instruct-tune) for behavior, RAG for grounded facts. Say that only after you’ve shown you understand the primary axes.

---

## R.5 How to deliver the answer (video’s interview advice)

1. Explain **RAG** with a grounded-domain example + open-book analogy.  
2. Explain **fine-tuning** from pretrain → small supervised nudge → weights change.  
3. Give a **forking example** (clone vs nutrition/pharma).  
4. Walk the **comparison table**.  
5. End with **start-with-RAG, escalate** judgment.

Lead with examples and passion; don’t blurt definitions and stop—invite the interviewer’s next question.

### 60-second script
“RAG and fine-tuning solve different failures. If a company’s library won’t fit in context, RAG retrieves the relevant chunks and grounds the answer—open-book exam; weights stay fixed. Fine-tuning starts from a pretrained model and nudges weights on a small labeled set so behavior or patterns change—studying the night before with no book in the room. For a grounded nutrition/pharma bot I’d start with RAG; for a digital clone that must sound like someone on novel questions I’d fine-tune. Compare on patterns, cost, simplicity, personalization; escalate from RAG → FT → pretrain only as needed.”

---

# Lecture 1 — Introduction to Inference Engineering (4 lessons)

## L1.1 Prefill vs decode

### Setup
User sends prompt tokens; model returns completion tokens.

### Problem
People say “inference” as one blob and then cannot optimize it.

### Root cause
Autoregressive factorization:

\[
P(x_t\mid x_{<t})
\]

forces **sequential** new tokens even when the prompt can be processed largely in parallel.

### Solution
Split the timeline:
1. **Prefill** — process prompt; produce first output token (highly parallel over prompt length).  
2. **Decode** — one (or few) new tokens per step; depends on past.

### Payoff
Every later lecture (KV, FlashAttention, vLLM, speculation) attaches to this split.

## L1.2 Latency vs throughput

### Setup
Two different goals.
- **Latency:** time to first token (TTFT) / time per output token (TPOT) for *one* user.  
- **Throughput:** tokens/sec or requests/sec across *many* users.

### Problem
Optimizing only TTFT can destroy GPU utilization; optimizing only batch throughput can destroy interactive latency.

### Root cause
GPUs want large batches; users want snappy single-stream decode.

### Solution
Measure both; use continuous batching / schedulers (Lecture 6) to trade them consciously.

## L1.3 Memory bill of materials

During inference, GPU/host memory typically holds:
1. **Weights**  
2. **Activations** (short-lived)  
3. **KV cache** (grows with sequence × layers × heads × dim)  

### Problem
At long context / high concurrency, KV dominates and OOMs first.

### Root cause
Decode reuses past K/V; you must store them (Lecture 2).

## L1.4 Compute-bound vs memory-bound (roofline intuition)

### Setup
Every kernel either needs more FLOPs or more bytes from HBM.

### Problem
You “optimize FLOPs” but wall-clock does not move.

### Root cause
Decode is often **memory-bandwidth bound** (reload weights + KV each step), not FLOP-bound.

### Solution
Quantization, GQA/MQA, FlashAttention IO patterns, speculation — all attack bytes moved or steps taken.

---

# Lecture 2 — Good and Evil of KV Cache (5 lessons)

**Primary teaching video:**
[Vizuara — LLM Interview Series #1: What exactly is the KV Cache?](https://www.youtube.com/watch?v=CxRGWfcGVbs)

---

## L2.0 Full interview walkthrough (follow this video’s story)

Do **not** open with “KV cache makes generation faster.”
Construct the answer from first principles ([video](https://www.youtube.com/watch?v=CxRGWfcGVbs)):

### Step 1 — KV cache is an *inference* concept
It shows up at **serve / generate** time, not as the pretraining objective.
Weights \(W_Q,W_K,W_V\) are already fixed.

### Step 2 — Prefill vs decode (must say this first)
Example prompt: `A sunset is` → model continues `extremely beautiful…`

1. **Prefill:** consume **all** prompt tokens (largely in parallel), run attention, emit the **first** output token. Video: often compute-bound.  
2. **Decode:** emit **one new token at a time** until stop. Video: this is where naive recompute hurts.

Until first token → prefill; after that → decode.

### Step 3 — Prefill at matrix level (what gets created)
Input embeddings \(X\) (e.g. 3 tokens × \(d\)) multiply fixed \(W_Q,W_K,W_V\) → matrices \(Q,K,V\).

Attention scores \(\propto Q K^\top\) (then scale / softmax / causal mask) → weights.
Context \(\propto\) weights \(\times V\).
Last position’s context → logits → first generated token.

**At end of prefill:** you already computed \(K\) and \(V\) rows for every prompt token. **Cache them.**

### Step 4 — Naive decode (what wrong answers imply)
To generate the next token after `… extremely`, a naive picture recomputes \(Q,K,V\) for the **entire** sequence every step, full score matrix, full context matrix — then keeps only the last row.

That recomputes past tokens’ \(K,V\) that **cannot change** (same tokens, frozen weights).

### Step 5 — Two realizations that define KV cache

**Realization A — Only the latest context vector is needed for the next token.**  
You do **not** need the full context matrix; you need the context for the newest position.

**Realization B — That latest context still needs *full* \(K\) and *full* \(V$.**  
Scores for the new query attend over **all** past keys; weighted sum uses **all** past values.
So you need the whole \(K,V\) history — but you must not **recompute** it.

### Step 6 — Actual decode with cache
1. Take embedding of **only the new token** (1 × \(d\)), not the whole history.  
2. Compute **only** that token’s \(q_{\text{new}}, k_{\text{new}}, v_{\text{new}}\).  
3. **Append** \(k_{\text{new}}, v_{\text{new}}\) to cached \(K,V\).  
4. Use \(q_{\text{new}}\) against **full** \(K^\top\) → scores/weights for this step only (1 × seq).  
5. Mix with full \(V\) → one context vector → logits → next token.  
6. Repeat; cache grows by one \(K,V\) row per new token (per layer / head layout).

### Step 7 — FAQ interviewers love: why not a Q cache?
To predict the next token you need:
- full \(K\), full \(V\),
- but **only** \(q\) for the **new** token — not historical \(Q\).

Past queries are not reused for the next step’s scores. Hence **KV** cache, not QKV cache.

### Step 8 — Payoff vs dark side (bridge to “evil”)
| Good | Evil |
|---|---|
| Avoids redundant matmuls; compute curve much closer to **linear** in new tokens vs **quadratic** recompute of the past | Cache **memory** grows with sequence × layers × KV heads × dim |
| Lower FLOPs → better TTFT/TPOT for users | Moving KV from HBM to compute each step → bandwidth bottleneck (→ GQA/MLA, PagedAttention, quant) |

Video defers deep “dark side” / complexity plots to later questions; Phase 1 continues in L2.2–L2.5 and L6.

### 60-second interview script
1. Inference-only; prefill vs decode.  
2. Prefill builds \(Q,K,V\) for the prompt; **store \(K,V\)**.  
3. Next token needs latest context ⇒ needs full \(K,V\) + new \(q\) only.  
4. Don’t recompute past \(K,V\); append new \(k,v\).  
5. No Q-cache — only new \(q\) is required.  
6. Speeds decode (less FLOPs); costs memory/bandwidth.

---

## L2.1 The good — why KV cache exists

(Compressed version of Steps 4–6.)

### Setup
Naive decode recomputes attention over the full prefix every step.

### Problem
Quadratic waste: past tokens’ \(K/V\) are recomputed despite being unchanged.

### Root cause
Past keys/values are deterministic given past tokens and frozen weights; only the newest token adds new \(k,v\).

### Solution
**KV cache:** store past \(K,V\); append only the new token’s \(K,V\); attend with new \(q\) only.

### Payoff
Far fewer FLOPs per decode step; better chatbot latency.

## L2.2 The evil — memory growth

KV size scales with:
`layers × kv_heads × head_dim × sequence_length × bytes_per_element × 2 (K and V)`  
(Exact formula depends on GQA/MQA and implementation layout.)

### Problem
One long chat or many concurrent chats → VRAM explosion.

### Root cause
You traded compute for memory; concurrency × length multiplies the trade.

## L2.3 Unknown output length

### Setup
Scheduler must reserve or allocate KV before/during decode.

### Problem
You do not know final decode length when the request arrives.

### Root cause
Contiguous over-reservation → fragmentation / blocked users (bridge to PagedAttention).

## L2.4 Prefix caching / chunked prefill (names you will hear)

| Idea | Problem | Idea in one line |
|---|---|---|
| Prefix / prompt caching | Repeated system prompts recomputed | Reuse KV for shared prefixes |
| Chunked prefill | Huge prefills block the GPU | Split prefill into chunks; interleave with decode |
| Eviction / compression (H2O, StreamingLLM, …) | Infinite context on finite VRAM | Drop or compress less useful KV (method-specific) |

Explain each only after stating which KV evil it targets. Do not dump acronyms.

## L2.5 Bridge to Lecture 6
KV is necessary; naive allocation is evil → **PagedAttention** + continuous batching.

---

# Bonus — Inference and YCombinator (3 lessons)

Treat this as **product framing**, not new math.

### Setup
Startups sell tokens, latency SLAs, and reliability—not “Transformers.”

### Problem
Brilliant kernels that do not move unit economics or UX do not ship.

### Root cause
Inference sits on the critical path of cost of goods (GPU hours) and retention (speed).

### Solution mindset
Pick optimizations by **ROI**: which bottleneck burns money or users this week?
Order of attack is often: (1) stop OOMs / raise batch, (2) cut KV, (3) quantize, (4) speculate, (5) fancy kernels.

### Payoff
You can pitch and prioritize like an infra founder, not only like a paper reader.

---

# Lecture 3 — Attention Variants Part 1 (6 lessons)

**Primary teaching video for GQA:**
[Vizuara — LLM Interview Series #6: What Is Grouped Query Attention?](https://www.youtube.com/watch?v=mtsY7JsGQjw)

Shared setup: attention needs \(Q,K,V\). At decode, loading \(K,V\) from the cache is a major **HBM → compute** bandwidth cost.

---

## L3.0 Full interview walkthrough: GQA (follow this video’s story)

Do **not** open with “GQA shares KV across groups.”
Construct the story in this order ([video](https://www.youtube.com/watch?v=mtsY7JsGQjw)):

### Step 1 — Place attention in the whole LM
LM = **input block** (token + position embeddings) → **stacked Transformer blocks** → **output / logits**.
Inside one Transformer block: LayerNorm → **multi-head attention** → residual → LayerNorm → FFN → residual.
Zoom into MHA only after locating it.

### Step 2 — Why multi-head exists (perspectives)
Single-head attention: one score matrix → one “view” of relations in the sentence.
Multi-head: each head has its own \(Q_i,K_i,V_i\) path → its own attention scores → concatenated context.
**Need for multiple heads:** capture diverse perspectives in language
(video example: ambiguous attachment — *brush* with *artist* vs *woman*).

Paper anchor for MHA: Vaswani et al., 2017.

### Step 3 — The need for GQA starts in the KV cache
In MHA with \(h\) heads, decode stores **per-head** keys and values:
\(K_1,\ldots,K_h\) and \(V_1,\ldots,V_h\).

**Root fact:** KV cache size scales with the number of KV heads.
Larger KV ⇒ more bytes moved from HBM to compute each decode step ⇒ latency / capacity pain
(even on a fast GPU, the transfer can dominate).

So the innovation need is: **shrink KV without throwing away too much quality.**

### Step 4 — Multi-Query Attention (MQA) — the extreme fix
**Idea (Shazeer, 2019):** keep **different queries** per head (\(Q_1 \neq Q_2 \neq \cdots\)),
but force **one shared K and one shared V** across all heads
(\(K_1=K_2=\cdots=K_h\), same for \(V\)).

**What you store:** only one \(K\) and one \(V\) (not \(h\) of each).
KV no longer scales with query-head count → large memory cut (up to ~\(h\times\) in the toy counting argument).

**Why language quality suffers:** heads exist to diversify attention scores.
If all heads share the same \(K\) (and \(V\)), diversity comes only from different \(Q\)s → weaker multi-perspective modeling.
Video framing: MQA = **low KV (good)** + **poor language pattern capture (bad)**.

### Step 5 — Grouped-Query Attention (GQA) — the middle
**Paper:** Ainslie et al., EMNLP 2023.

**Idea:** do **not** share one KV across *all* heads.
Partition query heads into \(g\) groups. **Within a group**, share K and V (MQA-style).
**Across groups**, K/V differ.

Toy with 4 query heads and 2 groups (as in the video):
- Group 1: \(K_1=K_2\), \(V_1=V_2\)
- Group 2: \(K_3=K_4\), \(V_3=V_4\)
- But \(K_1 \neq K_3\), \(V_1 \neq V_3\)

**What you store:** one \(K\) and one \(V\) **per group** (here \(K_1,K_3\) and \(V_1,V_3\)), not all four.

Endpoints:
- \(g = h\) → MHA (no sharing)
- \(g = 1\) → MQA (full sharing)

### Step 6 — Spectrum (how to draw it in an interview)

| | KV memory | Language / multi-perspective quality |
|---|---|---|
| **MHA** | Highest (bad for serve) | Highest (good) |
| **GQA** | Middle | Middle |
| **MQA** | Lowest (good for serve) | Lowest (often too weak) |

Video’s practical claim: production open models rarely ship pure MQA because quality drops too much; **GQA is the common compromise** (e.g. Llama family uses GQA; number of groups is a hyperparameter—verify per model card, don’t invent).

### Step 7 — Honest drawback (video ending)
GQA is **middle for everything**: not best quality, not best KV savings.
The video teases **multi-latent attention (MLA)** as aiming for stronger quality *and* larger KV savings—cover MLA as a **different** mechanism in L3.4; do not equate MLA with GQA.

### 60-second interview script (from this video’s recap)
1. Locate MHA in the Transformer.  
2. Multiple heads → multiple perspectives.  
3. MHA stores all K/V heads → KV ∝ heads → HBM traffic hurts decode.  
4. MQA shares one K/V → tiny KV, weak diversity.  
5. GQA shares K/V **inside groups only** → middle memory, middle quality.  
6. Name Llama-style adoption + group count as a hyperparameter.  
7. Optional: MLA is a later/different compression idea.

---

## L3.1 Multi-Head Attention (MHA) — Vaswani et al., 2017

See Step 2 above. Inference pain: store all \(h\) K/V heads.

## L3.2 Multi-Query Attention (MQA) — Shazeer, 2019

See Step 4. Multi-**query** = queries differ; keys/values shared.

## L3.3 Grouped-Query Attention (GQA) — Ainslie et al., EMNLP 2023

See Steps 5–7. Public paper: quality close to MHA with speed closer to MQA after uptraining recipes in that work.

## L3.4 Latent attention (MLA-style)

### Setup
DeepSeek-style **Multi-head Latent Attention (MLA)** compresses what must be cached into a latent form (architecture-specific; see DeepSeek reports).

### Problem
Even GQA KV can be large at extreme context.

### Caution
**Not the same as GQA.** Video explicitly separates them: GQA is group-sharing of K/V heads; MLA is a different compression design. Prefer the model paper for equations.

## L3.5 Sparse attention

### Setup
Full attention is expensive as sequence length grows.

### Problem
Dense all-pairs may be unnecessary for many tokens.

### Solution family
Local+global patterns, block sparsity, etc. (architecture-specific).

## L3.6 How to answer in interviews
Use **L3.0** — never open with the acronym. Story: MHA need → KV evil → MQA extreme → GQA middle → (optional) MLA.

---

# Lecture 4 — Attention Variants Part 2 (3 lessons)

## L4.1 Sliding Window Attention (SWA)

### Setup
Each token attends only to a local window of past tokens (plus optional global tokens), as in Mistral-style designs.

### Problem
Full context attention is too costly for long sequences.

### Root cause
Most local syntax/semantics may not need full \(n\times n\) attention every layer.

### Solution
Banded / windowed attention → compute/memory scale with window size \(w\), not full \(n\) (per layer assumptions).

### Tradeoff
Information beyond the window must propagate across layers/depth (or via other mechanisms).

## L4.2 State Space Models (SSMs) — intuition

### Setup
Sequence models that mix information with linear state updates (S4 lineage), aiming for long-range modeling with better asymptotic cost than dense attention.

### Problem
Quadratic attention limits length.

### Root cause
Explicit all-pairs attention is expensive; a compressed state can carry history.

### Solution
Maintain a recurrent / convolutional state that summarizes the past.

## L4.3 Mamba

### Setup
Mamba (Gu & Dao et al., 2023) popularized selective SSMs for language modeling with hardware-aware implementations.

### Problem
Want Transformer-like quality with better long-sequence efficiency characteristics.

### Root cause
Need selectivity (input-dependent state updates) + efficient parallel scan/kernels.

### Solution
Selective SSM architecture + efficient implementation (see the Mamba paper for equations—do not invent matrix forms from memory in an interview; state the idea).

### Payoff
Alternative sequence backbone / hybrid stacks appearing in modern systems. Serving stacks (including vLLM) increasingly discuss hybrid/Mamba support—check current engine docs for your version.

---

# Lecture 5 — FlashAttention 1, 2, 3 (4 lessons)

## Shared problem statement

### Setup
Standard attention materializes large \(S = QK^\top\) (and softmax) in HBM.

### Problem
Memory traffic dominates; naive attention is slow and memory-hungry even when FLOPs look fine.

### Root cause
GPU **HBM ↔ SRAM** IO, not just arithmetic intensity.

## L5.1 FlashAttention-1 (Dao et al., NeurIPS 2022)

### Solution ideas
- Tiling: bring blocks of \(Q,K,V\) into SRAM.  
- **Online softmax** so you never need the full score matrix in HBM.  
- Recomputation in backward (training) to save memory.  
- Exact attention (same math result, different IO schedule).

### Payoff
Faster, less memory for long sequences; foundational for modern stacks.

## L5.2 FlashAttention-2

### Problem FA1 left on the table
Work partitioning / parallelism not saturating GPUs as well as possible.

### Solution
Better parallelism and work partitioning (see Hazy Research FA2 writeup): higher utilization; broader head-dim support; MQA/GQA support called out in FA2 materials.

### Payoff
~2× over FA1 in reported settings (hardware/workload dependent).

## L5.3 FlashAttention-3 (Dao et al., NeurIPS 2024)

### Setup
Hopper GPUs (H100) expose asynchrony: Tensor Cores + TMA.

### Problem
FA2 leaves substantial utilization on the table on H100 (paper discusses low utilization vs peak).

### Solution (paper’s three techniques)
1. Warp specialization to overlap data movement and compute.  
2. Interleave block matmul and softmax.  
3. FP8 path with block quantization / incoherent processing for low-precision Tensor Cores.

### Payoff
Paper reports ~1.5–2.0× vs FA2 on H100 in BF16/FP16 regimes and high FP8 throughput; treat numbers as **on that hardware/paper setting**.

**Sources:** Dao et al. FlashAttention (2022); FlashAttention-2 blog; FlashAttention-3 (NeurIPS 2024) / [tridao.me/blog/2024/flash3](https://tridao.me/blog/2024/flash3/).

## L5.4 Interview line
“FlashAttention does not change the mathematical attention; it changes the **IO schedule** so GPUs stop drowning in HBM traffic.”

---

# Lecture 6 — The anatomy of a vLLM step (6 lessons)

Public anchors: [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm), Kwon et al. SOSP 2023 (PagedAttention).

## L6.1 Why a serving engine ≠ `transformers.generate`

### Problem
HF generate is great for correctness demos; multi-tenant production needs scheduling, memory managers, and batching policies.

## L6.2 Continuous batching

### Setup
Requests arrive and finish at different times.

### Problem
Static batching leaves GPU lanes idle when short sequences finish.

### Solution
Dynamically add/remove sequences each iteration (iteration-level scheduling; Orca/vLLM lineage).

## L6.3 PagedAttention (full layers — same as interview video)

1. **Basics:** GPU memory = weights + activations + KV.  
2. **Problem:** contiguous KV reservation → internal/external fragmentation; vLLM blog cites large waste fractions in prior systems.  
3. **Root cause:** request KV forced contiguous in physical memory.  
4. **Solution:** fixed-size KV **blocks/pages** + **block table** + free list; allocate on demand; non-contiguous OK.  
5. **Payoff:** higher effective batch → throughput; optional block sharing (parallel sampling / prefix) via refcount / CoW.

## L6.4 One scheduler step (mental model)

Each engine step roughly:
1. Choose which waiting/running requests run (policy).  
2. Ensure blocks allocated for new tokens.  
3. Run model forward for the batched tokens (prefill chunks and/or decode).  
4. Sample next token IDs.  
5. Append to sequences; free blocks for finished requests; update caches.

Exact class names change across vLLM versions—learn the **roles**, not frozen filenames.

## L6.5 Prefix caching / sharing
Block tables make shared prefixes cheap to reuse (system prompts, multi-sample from one prompt).

## L6.6 What to benchmark in the Phase 1 capstone
Throughput vs latency curves, KV memory vs concurrency, with/without paging features, before/after quant — matching the public capstone theme: *speed-optimized inference server* ([inference.vizuara.ai](https://inference.vizuara.ai/)).

---

# Lecture 7 — All about Quantization (4 lessons)

## L7.1 Why quantize

### Setup
Weights (and sometimes KV/activations) in FP16/BF16 are large.

### Problem
VRAM and memory bandwidth limit batch size and tok/s.

### Root cause
Decode often memory-bound: fewer bytes/weight ⇒ more effective bandwidth.

## L7.2 Precision ladder (conceptual)

| Format | Role |
|---|---|
| FP32 | Reference / rare in modern LLM serve |
| FP16 / BF16 | Common training/serve baseline |
| INT8 | Weight (and sometimes act) quant |
| INT4 / NF4-style | Aggressive weight quant for edge / consumer GPUs |
| FP8 | Hardware-special (Hopper) paths; also in FA3 story |

## L7.3 Post-training methods you must separate

| Method | Problem it targets | Idea (do not invent formulas) |
|---|---|---|
| GPTQ | Accurate low-bit **weight** quant after training | Layer-wise quantization with second-order / Hessian-style compensation (Frantar et al.) |
| AWQ | Protect salient weights | Activation-aware weight quantization (Lin et al.) |
| GGUF / llama.cpp quants | CPU/edge distribution formats | Ecosystem packing for local runtimes |
| KV / activation quant | Cache/activation bandwidth | Separate from weight-only quant—say which tensors you mean |

### Payoff
Fit bigger batches / smaller GPUs.  
### Danger
Quality cliffs; always eval your task suite after quant.

## L7.4 Interview discipline
State: *which tensors* (weights vs KV vs activations), *bits*, *method*, *hardware*, *quality check*.

---

# Lecture 8 — Speculative Decoding

Public companion reading: [Vizuara Substack — Speculative Decoding in vLLM](https://vizuara.substack.com/p/speculative-decoding-theory-and-implementation); Leviathan et al. ICML 2023.

## L8.1 The problem (before the name)

### Setup
Target model \(M_p\) is large; each decode step is expensive and serial.

### Problem
Even with FlashAttention + PagedAttention, you still pay a full forward of \(M_p\) per accepted token (roughly).

### Root cause
Autoregressive dependence: cannot freely parallelize future tokens of the *same* sequence on the target alone.

## L8.2 Classic draft–verify (Leviathan et al.)

### Solution
1. Small **draft** model \(M_q\) proposes \(K\) future tokens cheaply.  
2. Large **target** \(M_p\) verifies those tokens **in parallel** in one forward (with rejection sampling correction so the output distribution matches \(M_p\)).  
3. Accept a prefix of drafts; sample a correction on the first rejection.

### Payoff
Fewer target forwards per output token when acceptance rate is high.  
### Tradeoff
Bad drafts ⇒ wasted verify; need memory for draft+target.

## L8.3 Modern variants (names → which problem)

| Family | Extra idea |
|---|---|
| N-gram / prompt lookup | Draft from prompt statistics (no small LM) |
| Medusa | Extra decoding heads on the target for multi-token drafts |
| EAGLE / EAGLE-3 | Draft from target features / dedicated head; high acceptance when trained well |
| MTP | Multi-token prediction training objectives that help speculation |

vLLM exposes speculation via speculative config (see current vLLM docs / Vizuara walkthrough for flags). Compatibility with prefix caching depends on version/backend—verify for your release.

## L8.4 Interview answer skeleton
1. Prefill/decode + memory-bound decode.  
2. Serial target steps are the latency wall.  
3. Draft cheap tokens → verify in parallel on target → preserve target distribution.  
4. Acceptance rate is the real dial; name one method (EAGLE3 / draft model) and the tradeoff.

---

# End-to-end Phase 1 story (one breath)

```text
Hardware roofline
  → prefill vs decode, latency vs throughput
  → KV cache (good: no recompute; evil: memory)
  → shrink KV: MQA/GQA/MLA/sparse/window/SSM
  → make attention IO-efficient: FlashAttention 1/2/3
  → serve many users: vLLM continuous batching + PagedAttention
  → shrink bytes: quantization
  → skip target steps: speculative decoding
  → measure on real devices (labs) and ship a fast server (capstone)
```

---

# Interview checklist (Phase 1)

| If they ask… | Open with… | Then… |
|---|---|---|
| Entropy | Disorder as a *count of arrangements* | Shannon surprise → CE / temp / KL |
| Diffusion | Creation too hard one-shot | Corrupt gently → undo one step from noise |
| Expressivity | Stuck in too few dimensions | Lift / widen so patterns appear (watch overfit) |
| Compression | Too big (mem/compute/ΔW/pixels) | Narrow waist → work small → project up |
| Transformer arch | Surface “attention+FFN” | Input/processor/output → 8 modules → QKV |
| Attention | “Focus on important tokens” | Need for links → QKV → causal scores → multi-head |
| RAG vs fine-tuning | Context can’t hold the corpus / need grounded answers | Retrieve vs change weights; patterns/cost/simplicity; start RAG |
| KV cache | Prefill/decode recompute waste | Memory growth evil |
| PagedAttention | Contiguous KV fragmentation | Blocks + block table |
| GQA | MHA KV bandwidth | Share KV across query groups |
| FlashAttention | HBM traffic | Tiling + online softmax |
| vLLM step | Multi-tenant batching | Schedule → allocate blocks → forward → sample |
| Quantization | Memory-bound decode | Which tensors + GPTQ/AWQ/GGUF |
| Speculative decoding | Serial target steps | Draft–verify + acceptance |

---

# Landmark sources

| Topic | Anchor |
|---|---|
| Course framing | [inference.vizuara.ai](https://inference.vizuara.ai/), [Maven](https://maven.com/vizuara/inference-workshop) |
| Entropy mental model | [YouTube `q3agYqgqklU`](https://www.youtube.com/watch?v=q3agYqgqklU); [visual notes](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-09-entropy.html) |
| Diffusion mental model | [YouTube `bTEfdB2D1Ek`](https://www.youtube.com/watch?v=bTEfdB2D1Ek); [visual notes](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-07-reverse-the-corruption.html) |
| Expressivity mental model | [YouTube `YE0udJCgDqc`](https://www.youtube.com/watch?v=YE0udJCgDqc); [visual notes](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-06-expressivity.html) |
| Compression mental model | [YouTube `hGS6RbAYLl0`](https://www.youtube.com/watch?v=hGS6RbAYLl0); [visual notes](https://vizuaraai.github.io/great-mental-models-of-ai/lecture-05-compression.html) |
| Transformer architecture teaching | [YouTube `c533te7NSpI`](https://www.youtube.com/watch?v=c533te7NSpI) |
| Attention teaching | [YouTube `cquX2tOODUI`](https://www.youtube.com/watch?v=cquX2tOODUI) |
| RAG vs fine-tuning teaching | [YouTube `cCXjumE70-g`](https://www.youtube.com/watch?v=cCXjumE70-g) |
| KV cache teaching | [YouTube `CxRGWfcGVbs`](https://www.youtube.com/watch?v=CxRGWfcGVbs) |
| PagedAttention teaching | [YouTube `-AB6m0Spo6c`](https://www.youtube.com/watch?v=-AB6m0Spo6c) |
| GQA teaching | [YouTube `mtsY7JsGQjw`](https://www.youtube.com/watch?v=mtsY7JsGQjw) |
| vLLM / PagedAttention | Kwon et al., SOSP 2023; [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm) |
| MHA / MQA / GQA | Vaswani 2017; Shazeer 2019; Ainslie et al. EMNLP 2023 |
| FlashAttention 1/2/3 | Dao et al. 2022 / FA2 / FA3 NeurIPS 2024 |
| Speculative decoding | Leviathan et al. 2023; [Vizuara Substack](https://vizuara.substack.com/p/speculative-decoding-theory-and-implementation) |

---

*Also see:* `research/llm_end_to_end_survey.md` · LaTeX twin: `research/vizuara_inference_phase1_notes.tex`

