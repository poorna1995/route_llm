"""Join corpus + complexity φ + weak/strong probes → rows with z and labels."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "signals"))
sys.path.insert(0, str(ROOT / "signals" / "query_model"))

from analyze import is_correct, is_mc_row, read_jsonl  # noqa: E402
from feature_vector import Z_KEYS, as_floats, z_vectors  # noqa: E402
from store import resolve_signals_path  # noqa: E402

CORPUS_DIR = ROOT / "datasets" / "processed" / "corpus_v1"
COMPLEXITY_DIR = ROOT / "signals" / "query" / "processed"
PROBE_DIR = ROOT / "signals" / "query_model" / "processed"


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["query_id"]: r for r in rows}


def load_joined(role: str) -> list[dict[str, Any]]:
    """Inner-join all streams for one split role (fit|calib|eval)."""
    corpus = _by_id(read_jsonl(CORPUS_DIR / f"queries_{role}.jsonl"))
    phi_rows = _by_id(read_jsonl(COMPLEXITY_DIR / f"{role}.jsonl"))
    weak_path = resolve_signals_path(PROBE_DIR, role, "weak")
    strong_path = resolve_signals_path(PROBE_DIR, role, "strong")
    weak = _by_id(read_jsonl(weak_path)) if weak_path.exists() else {}
    strong = _by_id(read_jsonl(strong_path)) if strong_path.exists() else {}

    ids = sorted(set(corpus) & set(phi_rows) & set(weak))
    if not ids:
        raise FileNotFoundError(
            f"no joined rows for role={role!r} "
            f"(corpus={len(corpus)} phi={len(phi_rows)} weak={len(weak)} strong={len(strong)})"
        )

    out: list[dict[str, Any]] = []
    for qid in ids:
        corp, phi_row, w = corpus[qid], phi_rows[qid], weak[qid]
        gold = str(corp["gold"])
        metric = str(corp.get("metric", "accuracy"))
        w_pred = str((w.get("answer_scores") or {}).get("pred", ""))
        w_ok = is_correct(w_pred, gold, metric)

        s = strong.get(qid)
        has_strong = s is not None
        if has_strong:
            s_pred = str((s.get("answer_scores") or {}).get("pred", ""))
            s_ok = is_correct(s_pred, gold, metric)
            needs_strong = (not w_ok) and s_ok
            both_wrong = (not w_ok) and (not s_ok)
        else:
            s_pred, s_ok = "", False
            needs_strong = False
            both_wrong = False  # unknown without strong; do not treat as both_wrong

        zv = z_vectors(query_row=phi_row, model_row=w)
        out.append(
            {
                "query_id": qid,
                "role": role,
                "source": corp["source"],
                "is_mc": is_mc_row(corp),
                "gold": gold,
                "metric": metric,
                "weak_pred": w_pred,
                "strong_pred": s_pred,
                "weak_ok": w_ok,
                "strong_ok": s_ok,
                "weak_wrong": not w_ok,
                "needs_strong": needs_strong,
                "both_wrong": both_wrong,
                "has_strong": has_strong,
                "phi": zv["phi"],
                "psi": zv["psi"],
                "z": zv["z"],
                "z_vec": as_floats(zv["z"], Z_KEYS),
                "cost_weak": w.get("cost"),
                "cost_strong": (s or {}).get("cost"),
                "weak_model_id": w.get("model_id"),
                "strong_model_id": (s or {}).get("model_id"),
            }
        )
    return out


def slices(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    out: list[tuple[str, list[dict[str, Any]]]] = [("pooled", rows)]
    mc = [r for r in rows if r["is_mc"]]
    if mc and len(mc) != len(rows):
        out.append(("mc_pooled", mc))
    for src in sorted({r["source"] for r in rows}):
        out.append((src, [r for r in rows if r["source"] == src]))
    return out


def accuracy(rows: list[dict[str, Any]], which: str) -> float:
    if not rows:
        return float("nan")
    if which == "weak":
        return float(np.mean([r["weak_ok"] for r in rows]))
    if which == "strong":
        sub = [r for r in rows if r["has_strong"]]
        return float(np.mean([r["strong_ok"] for r in sub])) if sub else float("nan")
    raise ValueError(which)
