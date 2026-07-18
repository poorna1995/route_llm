"""Durable on-disk layout for query×model probes.

Layout (permanent — weak/strong never share a write target):
  processed/{role}_{model_role}.jsonl   e.g. fit_weak.jsonl, fit_strong.jsonl
  processed/backups/{stem}.{utc}.jsonl  timestamped snapshots before each write

Writes are atomic (tmp + rename) and default to upsert-by-query_id so
--limit / --sources smoke runs cannot wipe other queries or the other model.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_KEEP_BACKUPS = 20


def signal_path(out_dir: Path, role: str, model_role: str) -> Path:
    return out_dir / f"{role}_{model_role}.jsonl"


def backup_dir(out_dir: Path) -> Path:
    return out_dir / "backups"


def resolve_signals_path(out_dir: Path, role: str, model_role: str) -> Path:
    """Prefer role_modelRole file; fall back to legacy {role}.jsonl if present."""
    preferred = signal_path(out_dir, role, model_role)
    if preferred.exists():
        return preferred
    legacy = out_dir / f"{role}.jsonl"
    if legacy.exists():
        return legacy
    return preferred


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_backups(out_dir: Path, stem: str) -> list[Path]:
    """Newest-first backups for a signal stem (e.g. fit_weak)."""
    bdir = backup_dir(out_dir)
    if not bdir.exists():
        return []
    prefix = f"{stem}."
    files = [
        p
        for p in bdir.iterdir()
        if p.is_file() and p.name.startswith(prefix) and p.suffix == ".jsonl"
    ]
    return sorted(files, key=lambda p: p.name, reverse=True)


def prune_backups(out_dir: Path, stem: str, *, keep: int = DEFAULT_KEEP_BACKUPS) -> None:
    for old in list_backups(out_dir, stem)[keep:]:
        old.unlink(missing_ok=True)


def backup_signal_file(
    path: Path,
    *,
    keep: int = DEFAULT_KEEP_BACKUPS,
) -> Path | None:
    """Copy existing signal file into processed/backups/ before overwrite.

    Returns backup path, or None if there was nothing to back up.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    out_dir = path.parent
    bdir = backup_dir(out_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = bdir / f"{path.stem}.{ts}.jsonl"
    # Avoid collision if two writes in the same second
    n = 0
    while dest.exists():
        n += 1
        dest = bdir / f"{path.stem}.{ts}_{n}.jsonl"
    shutil.copy2(path, dest)
    prune_backups(out_dir, path.stem, keep=keep)
    return dest


def restore_from_backup(
    out_dir: Path,
    stem: str,
    *,
    backup: Path | str | None = None,
    keep: int = DEFAULT_KEEP_BACKUPS,
) -> Path:
    """Restore `{stem}.jsonl` from a backup (latest if backup is None).

    Backs up the current live file first (if any), then atomically replaces it.
    """
    target = out_dir / f"{stem}.jsonl"
    if backup is None:
        cands = list_backups(out_dir, stem)
        if not cands:
            raise FileNotFoundError(f"no backups for {stem!r} under {backup_dir(out_dir)}")
        src = cands[0]
    else:
        src = Path(backup)
        if not src.is_absolute():
            # allow bare filename or relative under backups/
            alt = backup_dir(out_dir) / src.name
            src = alt if alt.exists() else (out_dir / src)
        if not src.exists():
            raise FileNotFoundError(f"backup not found: {backup}")

    # Snapshot whatever is live before restore
    backup_signal_file(target, keep=keep)
    rows = read_jsonl(src)
    write_jsonl_atomic(target, rows, backup=False)
    return target


def write_jsonl_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    backup: bool = True,
    keep_backups: int = DEFAULT_KEEP_BACKUPS,
) -> Path | None:
    """Atomically write JSONL. Optionally snapshot the previous file first."""
    bak: Path | None = None
    if backup:
        bak = backup_signal_file(path, keep=keep_backups)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return bak


def upsert_by_query_id(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace rows whose query_id appears in new_rows; keep all others."""
    replace = {r["query_id"] for r in new_rows}
    kept = [r for r in existing if r["query_id"] not in replace]
    return kept + new_rows
