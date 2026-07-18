"""Calib τ sweep → PGR/CPT → freeze tau_star.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router.join import accuracy, load_joined  # noqa: E402
from router.metrics import cpt_from_curve, pgr, routed_accuracy  # noqa: E402
from router.rule_based.score import (  # noqa: E402
    DEFAULT_DIR,
    escalate_mask,
    fit_zstats,
    resolve_feature_sets,
    score_row,
)

HERE = Path(__file__).resolve().parent


def _load_cfg() -> dict[str, Any]:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")) or {}


def _load_rank(art: Path) -> dict[str, Any] | None:
    path = art / "rank_fit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sweep_set(
    rows: list[dict[str, Any]],
    feature_set: tuple[str, ...],
    *,
    directions: dict[str, int],
    zstats: dict[str, dict[str, float]],
    n_grid: int = 51,
) -> list[dict[str, Any]]:
    scores = np.array(
        [score_row(r, feature_set, directions=directions, zstats=zstats) for r in rows],
        dtype=float,
    )
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return []
    qs = [float(x) for x in np.quantile(finite, np.linspace(0, 1, n_grid))]
    # Endpoints: +inf ⇒ never escalate (α=0); -inf ⇒ always escalate (α=1)
    uniq: list[float] = []
    seen: set[float] = set()
    for t in [float("inf"), *qs, float("-inf")]:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    r_weak = accuracy(rows, "weak")
    r_strong = accuracy(rows, "strong")
    curve: list[dict[str, Any]] = []
    for tau in uniq:
        if tau == float("inf"):
            esc = [False] * len(rows)
        elif tau == float("-inf"):
            esc = [True] * len(rows)
        else:
            esc = escalate_mask(rows, feature_set, tau, directions=directions, zstats=zstats)
        alpha = float(np.mean(esc))
        r_r = routed_accuracy(rows, esc)
        curve.append(
            {
                "tau": None if not np.isfinite(tau) else tau,
                "tau_tag": "always_weak" if tau == float("inf") else (
                    "always_strong" if tau == float("-inf") else "threshold"
                ),
                "alpha": alpha,
                "accuracy": r_r,
                "pgr": pgr(r_r, r_weak, r_strong),
                "r_weak": r_weak,
                "r_strong": r_strong,
            }
        )
    return curve


def _json_safe(obj: Any) -> Any:
    """Convert NaN/Inf to null for strict JSON."""
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def pick_tau(curve: list[dict[str, Any]], *, select: str, alpha_max: float) -> dict[str, Any]:
    if not curve:
        return {"tau": float("nan"), "alpha": float("nan"), "pgr": float("nan"), "fallback": True}
    # Operating points only (exclude pure baselines for τ* unless fallback)
    ops = [c for c in curve if c.get("tau_tag") == "threshold"]
    pool = ops or curve

    def _best_cpt(target: float) -> dict[str, Any]:
        cands = [c for c in pool if np.isfinite(c.get("pgr", float("nan"))) and c["pgr"] >= target]
        if cands:
            return {**min(cands, key=lambda c: c["alpha"]), "fallback": False}
        # Unreachable: pick max-PGR operating point and flag
        best = max(pool, key=lambda c: (c["pgr"] if np.isfinite(c.get("pgr", float("nan"))) else -1e9))
        return {**best, "fallback": True}

    if select == "cpt50":
        best = _best_cpt(0.50)
        return {**best, "cpt50": cpt_from_curve(curve, 0.50), "cpt80": cpt_from_curve(curve, 0.80)}
    if select == "cpt80":
        best = _best_cpt(0.80)
        return {**best, "cpt50": cpt_from_curve(curve, 0.50), "cpt80": cpt_from_curve(curve, 0.80)}
    if select == "max_pgr_at_alpha":
        cands = [c for c in pool if c["alpha"] <= alpha_max + 1e-12]
        best = max(
            cands or pool,
            key=lambda c: (c["pgr"] if np.isfinite(c.get("pgr", float("nan"))) else -1e9),
        )
        return {
            **best,
            "fallback": not bool(cands),
            "cpt50": cpt_from_curve(curve, 0.50),
            "cpt80": cpt_from_curve(curve, 0.80),
        }
    raise ValueError(f"unknown select={select!r}")


def run_sweep(*, role: str = "calib") -> dict[str, Any]:
    cfg = _load_cfg()
    art = ROOT / str(cfg.get("artifacts_dir") or "router/artifacts/rule_based")
    art.mkdir(parents=True, exist_ok=True)
    rows = load_joined(role)
    if not all(r["has_strong"] for r in rows):
        missing = sum(1 for r in rows if not r["has_strong"])
        raise SystemExit(f"{role}: {missing}/{len(rows)} rows missing strong probes — run strong first")

    rank = _load_rank(art)
    order = (rank or {}).get("rank_order_pooled")
    sets = resolve_feature_sets(order)
    wanted = cfg.get("feature_sets") or list(sets)
    directions = {k: int(v) for k, v in (cfg.get("directions") or DEFAULT_DIR).items()}
    zstats = fit_zstats(load_joined("fit"), list(DEFAULT_DIR))

    select = str(cfg.get("select") or "cpt80")
    alpha_max = float(cfg.get("alpha_max") or 0.25)

    tau_star: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for name in wanted:
        if name not in sets:
            print(f"skip unknown set {name} (run rank first for top-k)", flush=True)
            continue
        fs = sets[name]
        curve = sweep_set(rows, fs, directions=directions, zstats=zstats)
        pick = pick_tau(curve, select=select, alpha_max=alpha_max)
        tau_val = pick.get("tau")
        tau_star[name] = {
            "features": list(fs),
            "tau": tau_val,
            "alpha": pick.get("alpha"),
            "pgr": pick.get("pgr"),
            "accuracy": pick.get("accuracy"),
            "cpt50": pick.get("cpt50"),
            "cpt80": pick.get("cpt80"),
            "select": select,
            "fallback": bool(pick.get("fallback")),
        }
        curves[name] = curve
        tau_s = f"{tau_val:.4f}" if isinstance(tau_val, (int, float)) and np.isfinite(tau_val) else str(tau_val)
        fb = "  [fallback: target PGR unreachable]" if pick.get("fallback") else ""
        cpt80 = pick.get("cpt80")
        cpt_s = f"{cpt80:.3f}" if isinstance(cpt80, (int, float)) and np.isfinite(cpt80) else "n/a"
        print(
            f"{name}: features={list(fs)} τ={tau_s} "
            f"α={pick.get('alpha'):.3f} PGR={pick.get('pgr'):.3f} "
            f"CPT80={cpt_s}{fb}",
            flush=True,
        )

    payload = {
        "role": role,
        "n": len(rows),
        "select": select,
        "tau_star": tau_star,
    }
    (art / "tau_star.json").write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    (art / "zscore_fit.json").write_text(json.dumps(zstats, indent=2), encoding="utf-8")
    (art / f"curves_{role}.json").write_text(json.dumps(_json_safe(curves)), encoding="utf-8")
    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sweep τ on calib and freeze tau_star.json")
    ap.add_argument("--role", default="calib")
    args = ap.parse_args()
    run_sweep(role=args.role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
