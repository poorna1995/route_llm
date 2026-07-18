#!/usr/bin/env python3
"""List / restore timestamped backups of {role}_{model_role}.jsonl probes.

Usage:
  ./run.sh query-model-restore --list --stem fit_weak
  ./run.sh query-model-restore --stem fit_strong                 # latest
  ./run.sh query-model-restore --stem fit_strong --backup fit_strong.20260716T143000Z.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from store import (  # noqa: E402
    backup_dir,
    list_backups,
    read_jsonl,
    restore_from_backup,
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="List or restore query_model probe backups")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument(
        "--stem",
        required=True,
        help="file stem without .jsonl, e.g. fit_weak or fit_strong",
    )
    ap.add_argument("--list", action="store_true", help="list backups (newest first)")
    ap.add_argument(
        "--backup",
        default=None,
        help="backup filename or path (default: newest)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be restored without writing",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ROOT / cfg["output_dir"]
    stem = args.stem.removesuffix(".jsonl")

    backups = list_backups(out_dir, stem)
    if args.list or args.dry_run and not args.backup:
        print(f"backups for {stem} in {backup_dir(out_dir)}:")
        if not backups:
            print("  (none)")
            return 1 if args.list else 0
        for i, p in enumerate(backups):
            n = len(read_jsonl(p))
            mark = " <- latest" if i == 0 else ""
            print(f"  {p.name}  rows={n}  bytes={p.stat().st_size}{mark}")
        if args.list:
            return 0

    if args.dry_run:
        src = backups[0] if args.backup is None else Path(args.backup)
        print(f"would restore {stem}.jsonl from {src}")
        return 0

    target = restore_from_backup(out_dir, stem, backup=args.backup)
    n = len(read_jsonl(target))
    print(json.dumps({"restored": str(target), "rows": n, "from": args.backup or "latest"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
