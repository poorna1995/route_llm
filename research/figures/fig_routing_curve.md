# Routing curves (eval) with frozen calib \(\tau^\star\)

**Raster:** `fig_routing_curve.png`  
**Command:** `./run.sh route curves --role eval`  
**Artifacts:** `router/artifacts/rule_based/curves_eval.json`, `markers_eval.json`, `tau_star.json`

## What is plotted

- **Left:** overall accuracy vs strong-call rate \(\alpha\) on **eval** (\(n=2597\)).
- **Right:** PGR vs \(\alpha\) on the same plot-only threshold sweep.
- Curves: \(S_H\), \(S_{\mathrm{top2}}\), \(S_{\mathrm{top4}}\), \(S_{\mathrm{top5}}\), \(S_{\mathrm{all}}\).
- **★** = operating point of the **calib-frozen** \(\tau^\star\) (CPT80) applied on eval — not retuned on eval.
- Horizontal refs: always-weak / always-strong (left); PGR \(=0.80\) and \(1.0\) (right).

## How to read

A curve farther **up and left** is better: more quality at fewer strong calls.  
CPT80 policy on calib targeted PGR \(\ge 0.80\) at minimal \(\alpha\); stars show where that frozen rule lands on eval.
