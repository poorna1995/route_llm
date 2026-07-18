# Signal comparison for unsupervised LLM routing

Analogous to confidence-routing “signal type” tables (entropy / logprob / self-consistency / verifier), specialized to **our** signals: unsupervised \(C(q)\), \(H\), \(p_{\max}\), paraphrase \(U\), Hotpot cluster entropy — **no preference / Arena labels**.

**Raster:** `fig_signal_comparison.png`

| Signal type | Latency overhead | Requires probe | Token cost | Reliability (fit) | Best use case |
|-------------|------------------|----------------|------------|-------------------|---------------|
| Answer entropy \(H(q\mid M_{\mathrm{weak}})\) | +1 weak probe | Yes (weak) | 1× weak | **High** on MC (AUROC ~0.85); Medium on Hotpot (~0.77) | **Primary** escalate signal when letter/option probs available |
| Max prob \(p_{\max}\) / margin | ~0 extra (same probe as \(H\)) | Yes (weak) | None extra | **High** on MC (~0.85–0.86); tracks \(H\) | Drop-in complement to \(H\); nearly interchangeable on ARC/MMLU |
| Hotpot cluster entropy | +\(n\) samples (e.g. \(n{=}3\)) | Yes (weak samples) | \(n\times\) weak | Medium (AUROC ~0.77 after full-context re-probe) | Free-form multi-hop when MC letter probe does not apply |
| Paraphrase uncertainty \(U\) | +\(k\) surface probes | Yes (weak × \(k\)) | \(k\times\) weak | TBD (deferred) | Secondary when single-surface \(H\) is ambiguous |
| Query complexity \(C(q)=g(\phi(q))\) | Embedding / features only | **No** pool LLM | None on weak/strong | Medium (task-dependent; join pending) | **Pre-inference** escalate before any candidate call; cheap filter |
| Joint \(S=\boldsymbol{\lambda}\!\cdot\!\mathbf{z}\) | As components | Depends on \(\mathbf{z}\) | As components | Tuned on **calib** | Weighted rule after signals validated; \(\tau\) frozen for eval |

**Not in table (labels, not routing signals):** `weak_wrong`, `needs_strong` = weak wrong ∧ strong right — used for **AUROC validation** and PGR/CPT, not as live features.

**Contrast with supervised routers:** RouteLLM-style systems learn \(P(\text{strong wins}\mid q)\) from preferences; our rows are **defined a priori** from probes / query features.
