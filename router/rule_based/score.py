"""Orient + z-score features; equal-weight escalate score s(q)."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# Default: higher ⇒ escalate
DEFAULT_DIR = {
    "C_length": +1,
    "C_density": +1,
    "C_atypical": +1,
    "C_linguistic": +1,
    "C_query": +1,
    "H": +1,
    "p_max": -1,
    "margin": -1,
    "top2_mass": -1,
}


def fit_zstats(rows: list[dict[str, Any]], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-feature mean/std on fit (finite values only)."""
    stats: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r["z"].get(k) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            stats[k] = {"mean": 0.0, "std": 1.0}
        else:
            std = float(vals.std())
            stats[k] = {"mean": float(vals.mean()), "std": std if std > 1e-9 else 1.0}
    return stats


def oriented_value(
    row: dict[str, Any],
    key: str,
    *,
    directions: dict[str, int] | None = None,
    zstats: dict[str, dict[str, float]] | None = None,
) -> float:
    d = (directions or DEFAULT_DIR).get(key, +1)
    raw = row["z"].get(key)
    if raw is None or not np.isfinite(raw):
        return float("nan")
    x = float(raw)
    if zstats and key in zstats:
        m, s = zstats[key]["mean"], zstats[key]["std"]
        x = (x - m) / s
    return d * x


def score_row(
    row: dict[str, Any],
    feature_set: Sequence[str],
    *,
    directions: dict[str, int] | None = None,
    zstats: dict[str, dict[str, float]] | None = None,
) -> float:
    """Equal-weight mean of oriented (optionally z-scored) features."""
    vals = [
        oriented_value(row, k, directions=directions, zstats=zstats) for k in feature_set
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def escalate_mask(
    rows: list[dict[str, Any]],
    feature_set: Sequence[str],
    tau: float,
    *,
    directions: dict[str, int] | None = None,
    zstats: dict[str, dict[str, float]] | None = None,
) -> list[bool]:
    return [
        score_row(r, feature_set, directions=directions, zstats=zstats) >= tau for r in rows
    ]


NAMED_SETS: dict[str, tuple[str, ...]] = {
    "S_H": ("H",),
    "S_p": ("p_max",),
    "S_md": ("H", "p_max", "margin"),
    "S_all": (
        "C_length",
        "C_density",
        "C_atypical",
        "C_linguistic",
        "C_query",
        "H",
        "p_max",
        "margin",
        "top2_mass",
    ),
}


def resolve_feature_sets(rank_order: Iterable[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Named sets + top-k from fit ranking order (highest AUROC first)."""
    sets = dict(NAMED_SETS)
    if rank_order:
        order = list(rank_order)
        sets["S_top2"] = tuple(order[:2])
        sets["S_top4"] = tuple(order[:4])
        sets["S_top5"] = tuple(order[:5])
    return sets
