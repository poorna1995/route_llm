# Learn-weight router (deferred)

Parameter-weighted routing from the problem formulation:

\[
S(q;\boldsymbol{\lambda})=\sum_j \lambda_j\, z_j(q,M_{\mathrm{weak}}),\quad
R=\mathrm{strong}\iff S\ge\tau
\]

**Not implemented yet.** Current work lives in `router/rule_based/` (thresholds + equal-weight top‑k sets).

When ready, reuse:
- `router/join.py` — joined rows with `z` and labels
- `router/metrics.py` — PGR / CPT
- fit z-score stats from `rule_based` artifacts

Do **not** train Arena/preference routers here — only light \(\lambda\) on calib over a priori signals.
