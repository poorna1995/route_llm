#!/usr/bin/env python3
"""CLI for rule-based router.

  ./run.sh route rank
  ./run.sh route sweep --role calib
  ./run.sh route eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule-based unsupervised router")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rank = sub.add_parser("rank", help="Rank z features on fit vs needs_strong")
    p_rank.add_argument("--role", default="fit")

    p_sweep = sub.add_parser("sweep", help="Sweep τ on calib; write tau_star.json")
    p_sweep.add_argument("--role", default="calib")

    p_eval = sub.add_parser("eval", help="Apply frozen τ* on eval")
    p_eval.add_argument("--role", default="eval")

    p_join = sub.add_parser("join-check", help="Sanity-check join counts")
    p_join.add_argument("--role", default="fit")

    args = ap.parse_args()

    if args.cmd == "rank":
        from router.rule_based.rank import run_rank

        payload = run_rank(role=args.role)
        print(f"wrote rank → router/artifacts/rule_based/rank_fit.json")
        print("order:", payload["rank_order_pooled"])
        return 0

    if args.cmd == "sweep":
        from router.rule_based.sweep import run_sweep

        run_sweep(role=args.role)
        return 0

    if args.cmd == "eval":
        from router.rule_based.evaluate import run_eval

        run_eval(role=args.role)
        return 0

    if args.cmd == "join-check":
        from router.join import load_joined

        rows = load_joined(args.role)
        n_s = sum(1 for r in rows if r["has_strong"])
        print(f"{args.role}: joined={len(rows)} with_strong={n_s}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
