"""Apply frozen τ* on eval (no retuning)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router.join import accuracy, load_joined, slices  # noqa: E402
from router.metrics import pgr, routed_accuracy  # noqa: E402
from router.rule_based.score import DEFAULT_DIR, escalate_mask  # noqa: E402

HERE = Path(__file__).resolve().parent


def run_eval(*, role: str = "eval") -> dict[str, Any]:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")) or {}
    art = ROOT / str(cfg.get("artifacts_dir") or "router/artifacts/rule_based")
    tau_path = art / "tau_star.json"
    z_path = art / "zscore_fit.json"
    if not tau_path.exists():
        raise SystemExit(f"missing {tau_path} — run: ./run.sh route sweep")
    if not z_path.exists():
        raise SystemExit(f"missing {z_path} — run sweep first")

    payload = json.loads(tau_path.read_text(encoding="utf-8"))
    tau_star = payload["tau_star"]
    tuned_on = payload.get("role")
    if tuned_on and tuned_on != "calib":
        print(
            f"WARNING: tau_star.json was tuned on role={tuned_on!r}, not calib. "
            f"Re-run: ./run.sh route sweep --role calib",
            flush=True,
        )
    zstats = json.loads(z_path.read_text(encoding="utf-8"))
    directions = {k: int(v) for k, v in (cfg.get("directions") or DEFAULT_DIR).items()}
    rows = load_joined(role)
    if not all(r["has_strong"] for r in rows):
        raise SystemExit(f"{role}: missing strong probes")

    r_weak = accuracy(rows, "weak")
    r_strong = accuracy(rows, "strong")
    report: dict[str, Any] = {
        "role": role,
        "n": len(rows),
        "r_weak": r_weak,
        "r_strong": r_strong,
        "sets": {},
    }

    print(f"eval n={len(rows)}  weak={r_weak:.3f}  strong={r_strong:.3f}")
    print(f"{'set':<10}{'α':>8}{'acc':>8}{'PGR':>8}")
    for name, spec in tau_star.items():
        fs = tuple(spec["features"])
        tau = float(spec["tau"])
        esc = escalate_mask(rows, fs, tau, directions=directions, zstats=zstats)
        alpha = float(np.mean(esc))
        acc = routed_accuracy(rows, esc)
        pg = pgr(acc, r_weak, r_strong)
        # path cost if available
        lat = []
        for r, e in zip(rows, esc):
            cw = (r.get("cost_weak") or {}).get("latency_ms")
            cs = (r.get("cost_strong") or {}).get("latency_ms")
            if cw is None:
                continue
            lat.append(float(cw) + (float(cs) if e and cs is not None else 0.0))
        by_src = {}
        for lab, subset in slices(rows):
            esc_s = escalate_mask(subset, fs, tau, directions=directions, zstats=zstats)
            acc_s = routed_accuracy(subset, esc_s)
            rw = accuracy(subset, "weak")
            rs = accuracy(subset, "strong")
            by_src[lab] = {
                "n": len(subset),
                "alpha": float(np.mean(esc_s)),
                "accuracy": acc_s,
                "pgr": pgr(acc_s, rw, rs),
            }
        entry = {
            "features": list(fs),
            "tau": tau,
            "alpha": alpha,
            "accuracy": acc,
            "pgr": pg,
            "mean_path_latency_ms": float(np.mean(lat)) if lat else None,
            "by_slice": by_src,
        }
        report["sets"][name] = entry
        print(f"{name:<10}{alpha:8.3f}{acc:8.3f}{pg:8.3f}")

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    out = art / f"eval_{role}.json"
    out.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return report


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate frozen rule-based router on eval")
    ap.add_argument("--role", default="eval")
    args = ap.parse_args()
    run_eval(role=args.role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
