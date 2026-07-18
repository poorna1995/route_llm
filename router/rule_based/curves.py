"""Plot routing curves on eval (plot-only sweep); mark calib τ* operating points.

  ./run.sh route curves --role eval
  → research/figures/fig_routing_curve.png
  → router/artifacts/rule_based/curves_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router.join import accuracy, load_joined  # noqa: E402
from router.metrics import pgr, routed_accuracy  # noqa: E402
from router.rule_based.score import (  # noqa: E402
    DEFAULT_DIR,
    escalate_mask,
    fit_zstats,
    resolve_feature_sets,
)
from router.rule_based.sweep import sweep_set  # noqa: E402

HERE = Path(__file__).resolve().parent

# Primary comparison sets for the paper figure
CURVE_SETS = ("S_H", "S_top2", "S_top4", "S_top5", "S_all")

# Distinct, print-friendly colors (avoid purple-gradient AI defaults)
COLORS = {
    "S_H": "#0B6E4F",
    "S_top2": "#1B4F72",
    "S_top4": "#B85C38",
    "S_top5": "#6B4C9A",
    "S_all": "#333333",
    "S_p": "#5D6D7E",
    "S_md": "#922B21",
}


def _load_cfg() -> dict[str, Any]:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")) or {}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _sort_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ops = [c for c in curve if c.get("tau_tag") == "threshold"]
    return sorted(ops, key=lambda c: c["alpha"])


def _op_at_tau(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    tau: float,
    *,
    directions: dict[str, int],
    zstats: dict[str, dict[str, float]],
    r_weak: float,
    r_strong: float,
) -> dict[str, float]:
    esc = escalate_mask(rows, features, tau, directions=directions, zstats=zstats)
    alpha = float(np.mean(esc))
    acc = routed_accuracy(rows, esc)
    return {"alpha": alpha, "accuracy": acc, "pgr": pgr(acc, r_weak, r_strong), "tau": tau}


def run_curves(
    *,
    role: str = "eval",
    out_png: Path | None = None,
    sets: tuple[str, ...] = CURVE_SETS,
) -> dict[str, Any]:
    cfg = _load_cfg()
    art = ROOT / str(cfg.get("artifacts_dir") or "router/artifacts/rule_based")
    tau_path = art / "tau_star.json"
    if not tau_path.exists():
        raise SystemExit(f"missing {tau_path} — run: ./run.sh route sweep --role calib")

    tau_payload = json.loads(tau_path.read_text(encoding="utf-8"))
    tau_star = tau_payload["tau_star"]

    rank_path = art / "rank_fit.json"
    order = None
    if rank_path.exists():
        order = json.loads(rank_path.read_text(encoding="utf-8")).get("rank_order_pooled")
    named = resolve_feature_sets(order)

    directions = {k: int(v) for k, v in (cfg.get("directions") or DEFAULT_DIR).items()}
    zstats = fit_zstats(load_joined("fit"), list(DEFAULT_DIR))
    rows = load_joined(role)
    if not all(r["has_strong"] for r in rows):
        raise SystemExit(f"{role}: missing strong probes")

    r_weak = accuracy(rows, "weak")
    r_strong = accuracy(rows, "strong")

    curves: dict[str, Any] = {}
    markers: dict[str, Any] = {}

    for name in sets:
        if name not in named:
            print(f"skip unknown set {name}", flush=True)
            continue
        if name not in tau_star:
            print(f"skip {name}: no τ* in tau_star.json", flush=True)
            continue
        fs = tuple(tau_star[name].get("features") or named[name])
        curve = sweep_set(rows, fs, directions=directions, zstats=zstats)
        curves[name] = curve
        tau = float(tau_star[name]["tau"])
        markers[name] = {
            "features": list(fs),
            "tau_star": tau,
            "calib": {
                "alpha": tau_star[name].get("alpha"),
                "accuracy": tau_star[name].get("accuracy"),
                "pgr": tau_star[name].get("pgr"),
            },
            "eval_at_tau_star": _op_at_tau(
                rows,
                fs,
                tau,
                directions=directions,
                zstats=zstats,
                r_weak=r_weak,
                r_strong=r_strong,
            ),
        }
        m = markers[name]["eval_at_tau_star"]
        print(
            f"{name}: τ*={tau:.4f}  eval@τ* α={m['alpha']:.3f} "
            f"acc={m['accuracy']:.3f} PGR={m['pgr']:.3f}",
            flush=True,
        )

    # Persist plot-only eval sweep
    (art / f"curves_{role}.json").write_text(
        json.dumps(_json_safe(curves), indent=2), encoding="utf-8"
    )
    (art / f"markers_{role}.json").write_text(
        json.dumps(_json_safe({"role": role, "r_weak": r_weak, "r_strong": r_strong, "sets": markers}), indent=2),
        encoding="utf-8",
    )

    out_png = out_png or (ROOT / "research" / "figures" / "fig_routing_curve.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _plot(curves, markers, r_weak=r_weak, r_strong=r_strong, role=role, out_png=out_png)
    print(f"wrote {out_png.relative_to(ROOT)}", flush=True)
    return {"curves": curves, "markers": markers, "png": str(out_png)}


def _plot(
    curves: dict[str, list[dict[str, Any]]],
    markers: dict[str, dict[str, Any]],
    *,
    r_weak: float,
    r_strong: float,
    role: str,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=160)
    ax_acc, ax_pgr = axes

    for name, curve in curves.items():
        ops = _sort_curve(curve)
        if not ops:
            continue
        xs = [c["alpha"] for c in ops]
        ys_acc = [c["accuracy"] for c in ops]
        ys_pgr = [c["pgr"] for c in ops]
        color = COLORS.get(name, "#444444")
        ax_acc.plot(xs, ys_acc, color=color, lw=1.8, label=name)
        ax_pgr.plot(xs, ys_pgr, color=color, lw=1.8, label=name)

        m = (markers.get(name) or {}).get("eval_at_tau_star") or {}
        if m:
            ax_acc.scatter(
                [m["alpha"]],
                [m["accuracy"]],
                s=70,
                color=color,
                marker="*",
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )
            ax_pgr.scatter(
                [m["alpha"]],
                [m["pgr"]],
                s=70,
                color=color,
                marker="*",
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )

    # Baselines
    ax_acc.axhline(r_weak, color="#888888", ls="--", lw=1.0, label="always weak")
    ax_acc.axhline(r_strong, color="#555555", ls=":", lw=1.0, label="always strong")
    ax_pgr.axhline(0.80, color="#888888", ls="--", lw=1.0, label="PGR = 0.80 (CPT80)")
    ax_pgr.axhline(1.0, color="#555555", ls=":", lw=1.0, label="PGR = 1.0")

    ax_acc.set_xlabel("Strong-call rate α")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"Accuracy vs α ({role}; ★ = calib τ*)")
    ax_acc.set_xlim(0, 1)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(loc="lower right", fontsize=8, frameon=False)

    ax_pgr.set_xlabel("Strong-call rate α")
    ax_pgr.set_ylabel("PGR")
    ax_pgr.set_title(f"PGR vs α ({role}; ★ = calib τ*)")
    ax_pgr.set_xlim(0, 1)
    ax_pgr.set_ylim(0, 1.05)
    ax_pgr.grid(True, alpha=0.25)
    ax_pgr.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle(
        "Unsupervised rule-based routing — eval curves with frozen calib τ*",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Plot routing curves; mark calib τ*")
    ap.add_argument("--role", default="eval")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG path (default: research/figures/fig_routing_curve.png)",
    )
    ap.add_argument(
        "--sets",
        default=",".join(CURVE_SETS),
        help="comma-separated feature sets",
    )
    args = ap.parse_args()
    sets = tuple(s.strip() for s in args.sets.split(",") if s.strip())
    run_curves(role=args.role, out_png=args.out, sets=sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
