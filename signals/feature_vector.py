"""Minimal feature vectors for unsupervised routing.

  φ(q)   — query-only          (model-independent)
  ψ(q,M) — query–model probe   (model-dependent)
  z      — concat [φ | ψ]

Extract from existing processed rows; no LLM calls.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Ordered coordinates (stable for tables / λ weights).
PHI_KEYS: tuple[str, ...] = (
    "C_length",
    "C_density",
    "C_atypical",
    "C_linguistic",
    "C_query",
)

PSI_KEYS: tuple[str, ...] = (
    "H",
    "p_max",
    "margin",
    "top2_mass",
)

Z_KEYS: tuple[str, ...] = PHI_KEYS + PSI_KEYS


def _get(d: Mapping[str, Any] | None, key: str) -> float | None:
    if not d or key not in d:
        return None
    v = d[key]
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def phi_from_complexity(complexity: Mapping[str, Any] | None) -> dict[str, float | None]:
    """φ(q) from signals/query row['complexity']."""
    return {k: _get(complexity, k) for k in PHI_KEYS}


def psi_from_scores(answer_scores: Mapping[str, Any] | None) -> dict[str, float | None]:
    """ψ(q,M) from query_model row['answer_scores']."""
    return {k: _get(answer_scores, k) for k in PSI_KEYS}


def phi_from_row(row: Mapping[str, Any]) -> dict[str, float | None]:
    return phi_from_complexity(row.get("complexity"))


def psi_from_row(row: Mapping[str, Any]) -> dict[str, float | None]:
    return psi_from_scores(row.get("answer_scores"))


def z_vectors(
    *,
    query_row: Mapping[str, Any],
    model_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build φ, ψ, and joint z from one complexity row + one probe row."""
    phi = phi_from_row(query_row)
    psi = psi_from_row(model_row)
    return {
        "query_id": model_row.get("query_id") or query_row.get("query_id"),
        "model_role": model_row.get("model_role"),
        "model_id": model_row.get("model_id"),
        "phi": phi,
        "psi": psi,
        "z": {**phi, **psi},
    }


def as_list(vec: Mapping[str, float | None], keys: Sequence[str]) -> list[float | None]:
    """Dense list in canonical key order (None if missing)."""
    return [vec.get(k) for k in keys]


def as_floats(vec: Mapping[str, float | None], keys: Sequence[str], *, fill: float = float("nan")) -> list[float]:
    """Dense float list; missing → fill (default NaN)."""
    out: list[float] = []
    for k in keys:
        v = vec.get(k)
        out.append(fill if v is None else float(v))
    return out
