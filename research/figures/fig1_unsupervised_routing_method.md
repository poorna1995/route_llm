# Figure 1 — Unsupervised LLM routing (method)

Analogous to Sawant’s *confidence pipeline* (Tier-1 call → entropy/confidence gate → serve / escalate), adapted to our formulation: **no preference labels**; thresholds on **calib**; weak-model probe defines \(H\) / \(p_{\max}\).

**Raster:** `fig1_unsupervised_routing_method.png`

## Editable diagram (Mermaid)

```mermaid
flowchart LR
  Q["Query q"] --> C["Optional<br/>C(q) = g(φ(q))<br/>pre-inference"]
  Q --> W["Weak probe<br/>M_weak"]
  W --> PSI["Signals ψ<br/>H(q | M_weak)<br/>p_max, margin, U"]
  C --> G{"Escalate?<br/>H ≥ τ_H<br/>or C ≥ τ_C"}
  PSI --> G
  G -->|yes| S["Strong model<br/>M_strong"]
  G -->|no| KEEP["Keep weak<br/>answer"]
  S --> Y["Final answer y"]
  KEEP --> Y

  classDef box fill:#1e3a5f,stroke:#0f2744,color:#fff
  classDef gate fill:#f4f4f5,stroke:#1e3a5f,color:#111
  class Q,C,W,PSI,S,KEEP,Y box
  class G gate
```

## Mapping to the Medium figure

| Sawant (confidence pipeline) | Our figure |
|------------------------------|------------|
| Tier-1 model call + logprobs | Weak probe \(M_{\mathrm{weak}}\) |
| Entropy / confidence scorer | \(\psi\): \(H\), \(p_{\max}\), margin, \(U\) |
| Confidence gate (3 zones) | Binary escalate gate \(\tau_H\) / \(\tau_C\) (calib) |
| Direct serve | Keep weak answer |
| Verify / escalate | Call \(M_{\mathrm{strong}}\) |
| (not in their fig) | Optional pre-inference \(C(q)\) |

We use a **binary** weak↔strong pool (paper scope) rather than three dispatch zones; the structure is the same: cheap probe → uncertainty → gate → serve or escalate.
