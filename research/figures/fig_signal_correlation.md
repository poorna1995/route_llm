# Signal correlations: \(C_*\) vs weak model-dependent probes

**Raster:** `fig_signal_correlation.png` · **Vector:** `fig_signal_correlation.pdf`

Pearson \(r\) on the fit join (\(n{=}2278\)): model-independent complexity scalars from `signals/query` vs weak-probe scores from `fit_weak.jsonl`.

## Panels
1. **Cross (pooled)** — \(C_*\) × \(\{H, p_{\max}, \mathrm{margin}, \mathrm{top2}\}\): near-zero (\(|r|\lesssim 0.14\)).
2. **Within \(C_*\)** — length/linguistic drive `C_query`; `C_atypical` is more distinct.
3. **Within weak MD** — \(H\), \(p_{\max}\), margin nearly redundant (\(|r|\approx 0.96\)–\(0.99\)).
4. **By dataset** — same cross pattern on ARC / MMLU / Hotpot / pooled.

## Caption (draft)
> Pearson correlations among unsupervised routing signals on the fit split. Model-independent complexity features \(C_*\) are nearly orthogonal to weak-model answer uncertainty \(H\) and \(p_{\max}\), while the model-dependent confidence family is highly collinear.
