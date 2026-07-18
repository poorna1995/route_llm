"""PGR / CPT and thin AUROC re-export for routers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "signals" / "query_model"))

from analyze import auroc  # noqa: E402


def pgr(r_router: float, r_weak: float, r_strong: float) -> float:
    gap = r_strong - r_weak
    if abs(gap) < 1e-12:
        return float("nan")
    return (r_router - r_weak) / gap


def cpt_from_curve(curve: list[dict[str, Any]], target_pgr: float) -> float:
    """Min strong-call rate α with PGR ≥ target (nan if unreachable)."""
    ok = [c for c in curve if np.isfinite(c.get("pgr", float("nan"))) and c["pgr"] >= target_pgr]
    if not ok:
        return float("nan")
    return float(min(c["alpha"] for c in ok))


def routed_accuracy(rows: list[dict[str, Any]], escalate: list[bool]) -> float:
    """Accuracy if escalate[i] → use strong else weak."""
    hits = []
    for r, esc in zip(rows, escalate):
        if esc:
            if not r["has_strong"]:
                return float("nan")
            hits.append(r["strong_ok"])
        else:
            hits.append(r["weak_ok"])
    return float(np.mean(hits)) if hits else float("nan")
