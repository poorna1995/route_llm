"""Fit-only feature ranking by AUROC → needs_strong."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "signals"))
sys.path.insert(0, str(ROOT / "signals" / "query_model"))

from analyze import auroc  # noqa: E402
from feature_vector import Z_KEYS  # noqa: E402
from router.join import load_joined, slices  # noqa: E402
from router.rule_based.score import DEFAULT_DIR, resolve_feature_sets  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load_cfg() -> dict[str, Any]:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")) or {}


def rank_features(rows: list[dict[str, Any]], directions: dict[str, int] | None = None) -> list[dict[str, Any]]:
    directions = directions or DEFAULT_DIR
    # Only rows with strong labels for needs_strong
    labeled = [r for r in rows if r["has_strong"]]
    results: list[dict[str, Any]] = []
    for label, subset in slices(labeled):
        y = np.array([int(r["needs_strong"]) for r in subset], dtype=np.int8)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for key in Z_KEYS:
            d = int(directions.get(key, +1))
            raw = np.array([r["z"].get(key, float("nan")) for r in subset], dtype=float)
            auc = auroc(y, d * raw)
            # also flipped (for reporting only — routing keeps config directions)
            auc_flip = auroc(y, -d * raw)
            auc_f = float(auc) if np.isfinite(auc) else float("nan")
            auc_flip_f = float(auc_flip) if np.isfinite(auc_flip) else float("nan")
            if np.isfinite(auc_f) and np.isfinite(auc_flip_f):
                best = max(auc_f, auc_flip_f)
            elif np.isfinite(auc_f):
                best = auc_f
            else:
                best = auc_flip_f
            results.append(
                {
                    "slice": label,
                    "feature": key,
                    "direction": d,
                    "auroc": auc_f,
                    "auroc_flipped": auc_flip_f,
                    "best_auroc": best,
                    "n": len(subset),
                    "n_pos": int(y.sum()),
                }
            )
    return results


def pooled_order(results: list[dict[str, Any]]) -> list[str]:
    """Rank features by pooled AUROC under default orientation (not flipped)."""
    pooled = [r for r in results if r["slice"] == "pooled"]
    pooled.sort(key=lambda r: (-(r["auroc"] if np.isfinite(r["auroc"]) else -1.0), r["feature"]))
    return [r["feature"] for r in pooled]


def run_rank(*, role: str = "fit") -> dict[str, Any]:
    cfg = _load_cfg()
    rows = load_joined(role)
    directions = {k: int(v) for k, v in (cfg.get("directions") or DEFAULT_DIR).items()}
    results = rank_features(rows, directions)
    order = pooled_order(results)
    sets = resolve_feature_sets(order)
    art = ROOT / str(cfg.get("artifacts_dir") or "router/artifacts/rule_based")
    art.mkdir(parents=True, exist_ok=True)
    payload = {
        "role": role,
        "n": len(rows),
        "n_with_strong": sum(1 for r in rows if r["has_strong"]),
        "rank_order_pooled": order,
        "feature_sets": {k: list(v) for k, v in sets.items()},
        "rows": results,
    }
    out = art / "rank_fit.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Rank z features on fit vs needs_strong")
    ap.add_argument("--role", default="fit")
    args = ap.parse_args()
    payload = run_rank(role=args.role)
    print(f"n={payload['n']} with_strong={payload['n_with_strong']}")
    print("rank_order_pooled:", payload["rank_order_pooled"])
    print("feature_sets:")
    for k, v in payload["feature_sets"].items():
        print(f"  {k}: {v}")
    print("pooled AUROC → needs_strong:")
    for r in payload["rows"]:
        if r["slice"] == "pooled":
            print(f"  {r['feature']:16s}  auroc={r['auroc']:.3f}  best={r['best_auroc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
