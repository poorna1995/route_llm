#!/usr/bin/env python3
"""Verify query_model outputs and run S1/S2 alignment analysis (WORKFLOW.md §9–10)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import normalize_answer  # noqa: E402
from store import resolve_signals_path  # noqa: E402

# Canonical AUROC direction: score increasing ⇒ more likely wrong (WORKFLOW §15.7).
SIGNAL_SPECS: list[tuple[str, str, int]] = [
    ("entropy", "H", +1),
    ("entropy", "perplexity_H", +1),
    ("confidence", "p_max", -1),
    ("confidence", "surprisal", +1),
    ("confidence", "inv_p_max", +1),
    ("margin", "margin", -1),
    ("top2", "top2_mass", -1),
]

S2_REPRESENTATIVES = ("H", "p_max", "margin", "top2_mass")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def corpus_source_counts(corpus: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(r["source"] for r in corpus).items()))


def is_mc_row(row: dict[str, Any]) -> bool:
    task = str(row.get("task_type", ""))
    if task.endswith("_mc"):
        return True
    return str(row.get("metric", "")) != "em"


def is_correct(pred: str, gold: str, metric: str) -> bool:
    if metric == "em":
        return normalize_answer(pred) == normalize_answer(gold)
    return str(pred).strip().upper() == str(gold).strip().upper()


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC via Mann–Whitney U (no sklearn)."""
    y = y_true.astype(np.int8)
    s = scores.astype(np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # Average ranks for ties (sorted-order positions i..j-1).
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[i]] == s[order[j]]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def verify(
    *,
    signals_path: Path,
    corpus_path: Path,
    manifest_path: Path | None,
    role: str,
    model_role: str,
) -> dict[str, Any]:
    issues: list[str] = []
    if not signals_path.exists():
        raise SystemExit(f"missing signals: {signals_path}")

    corpus = read_jsonl(corpus_path)
    corpus_by_id = {r["query_id"]: r for r in corpus}
    expected_by_source = corpus_source_counts(corpus)
    expected_total = len(corpus)

    signals = read_jsonl(signals_path)
    role_rows = [r for r in signals if r.get("model_role") == model_role]

    report: dict[str, Any] = {
        "signals_path": str(signals_path),
        "corpus_path": str(corpus_path),
        "n_signals_file": len(signals),
        "n_role_rows": len(role_rows),
        "n_corpus": expected_total,
        "expected_total": expected_total,
        "role": role,
        "model_role": model_role,
        "issues": issues,
        "ok": True,
    }

    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["manifest"] = {
            "backend": manifest.get("backend"),
            "created_utc": manifest.get("created_utc"),
            "counts": manifest.get("counts"),
            "models": manifest.get("models"),
        }
        if manifest.get("backend") == "mock":
            issues.append("manifest backend is mock (not a real HF run)")
        manifest_count = (manifest.get("counts") or {}).get(role)
        if manifest_count is not None and manifest_count != len(role_rows):
            issues.append(
                f"manifest counts[{role!r}]={manifest_count} != role rows {len(role_rows)}"
            )

    if not role_rows:
        issues.append(f"no rows with model_role={model_role!r}")

    if len(role_rows) != expected_total:
        issues.append(f"role row count {len(role_rows)} != corpus {expected_total}")

    model_roles_in_file = sorted({r.get("model_role") for r in signals})
    report["model_roles_in_file"] = model_roles_in_file

    src_counts = Counter(r.get("source") for r in role_rows)
    report["source_counts"] = dict(sorted(src_counts.items()))
    report["expected_source_counts"] = expected_by_source
    for src, exp in expected_by_source.items():
        if src_counts.get(src, 0) != exp:
            issues.append(f"{src}: got {src_counts.get(src, 0)}, expected {exp}")

    dupes = [k for k, v in Counter(r["query_id"] for r in role_rows).items() if v > 1]
    if dupes:
        issues.append(f"duplicate query_id count: {len(dupes)}")

    missing = sorted(set(corpus_by_id) - {r["query_id"] for r in role_rows})
    extra = sorted({r["query_id"] for r in role_rows} - set(corpus_by_id))
    if missing:
        issues.append(f"missing query_ids: {len(missing)} (e.g. {missing[:3]})")
    if extra:
        issues.append(f"unknown query_ids: {len(extra)} (e.g. {extra[:3]})")

    required = ("query_id", "model_role", "answer_scores", "probe")
    for i, row in enumerate(role_rows[:5]):
        for key in required:
            if key not in row:
                issues.append(f"row {i} missing field {key!r}")
                break
        scores = row.get("answer_scores") or {}
        for field in ("H", "p_max", "pred"):
            if field not in scores:
                issues.append(f"row {i} answer_scores missing {field!r}")
                break

    # Mock runs often have perfectly uniform fake MC probs.
    for row in role_rows[:20]:
        if row.get("probe", {}).get("kind") != "mc_letter":
            continue
        probs = (row.get("answer_scores") or {}).get("probs") or {}
        if probs and len(set(probs.values())) == 1:
            issues.append(
                f"query {row['query_id'][:12]} has uniform MC probs (likely mock backend)"
            )
            break

    report["issues"] = issues
    report["ok"] = not issues
    return report


def join_rows(
    signals: list[dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
    *,
    model_role: str,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for row in signals:
        if row.get("model_role") != model_role:
            continue
        qid = row["query_id"]
        if qid not in corpus_by_id:
            continue
        corp = corpus_by_id[qid]
        scores = row["answer_scores"]
        pred = str(scores.get("pred", ""))
        gold = str(corp["gold"])
        metric = str(corp.get("metric", "accuracy"))
        correct = is_correct(pred, gold, metric)
        paraphrase = row.get("paraphrase") or {}
        joined.append(
            {
                "query_id": qid,
                "source": corp["source"],
                "task_type": corp.get("task_type"),
                "metric": metric,
                "is_mc": is_mc_row(corp),
                "pred": pred,
                "gold": gold,
                "correct": correct,
                "wrong": not correct,
                "scores": scores,
                "paraphrase": paraphrase,
            }
        )
    return joined


def analysis_slices(joined: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    slices: list[tuple[str, list[dict[str, Any]]]] = [("pooled", joined)]
    mc = [r for r in joined if r["is_mc"]]
    if mc and len(mc) != len(joined):
        slices.append(("mc_pooled", mc))
    for src in sorted({r["source"] for r in joined}):
        slices.append((src, [r for r in joined if r["source"] == src]))
    return slices


def signal_specs_for(joined: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    specs = list(SIGNAL_SPECS)
    if any("U" in (r.get("paraphrase") or {}) for r in joined):
        specs.append(("paraphrase", "U", +1))
    return specs


def s1_table(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, subset in analysis_slices(joined):
        n = len(subset)
        acc = sum(1 for r in subset if r["correct"]) / n
        rows.append({"slice": label, "n": n, "accuracy": acc})
    return rows


def s2_table(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = signal_specs_for(joined)

    for label, subset in analysis_slices(joined):
        if len(subset) < 2:
            continue
        y_sub = np.array([int(r["wrong"]) for r in subset], dtype=np.int8)
        if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            continue
        for family, field, direction in specs:
            if family == "paraphrase":
                raw = np.array(
                    [float(r["paraphrase"].get(field, float("nan"))) for r in subset]
                )
            else:
                raw = np.array([float(r["scores"].get(field, float("nan"))) for r in subset])
            auc = auroc(y_sub, direction * raw)
            rows.append(
                {
                    "slice": label,
                    "family": family,
                    "signal": field,
                    "direction": direction,
                    "n": len(subset),
                    "n_wrong": int(y_sub.sum()),
                    "auroc": auc,
                    "s2_representative": field in S2_REPRESENTATIVES or field == "U",
                }
            )
    return rows


def print_verify(report: dict[str, Any]) -> None:
    print(f"signals: {report['signals_path']}")
    print(f"corpus:  {report['corpus_path']} ({report['n_corpus']} queries)")
    print(
        f"rows:    {report['n_role_rows']} for model_role={report['model_role']!r} "
        f"(file has {report['n_signals_file']}, expected {report['expected_total']})"
    )
    if "manifest" in report:
        m = report["manifest"]
        print(f"manifest backend: {m.get('backend')}  created: {m.get('created_utc')}")
    print("source_counts:", report.get("source_counts"))
    if report["ok"]:
        print("VERIFY: OK")
    else:
        print("VERIFY: FAILED")
        for issue in report["issues"]:
            print(f"  - {issue}")


def print_s1(rows: list[dict[str, Any]], *, model_role: str) -> None:
    print(f"\n=== S1 accuracy ({model_role} pred vs gold) ===")
    print(f"{'slice':<16} {'n':>6} {'accuracy':>10}")
    for r in rows:
        print(f"{r['slice']:<16} {r['n']:>6} {r['accuracy']:>10.3f}")


def print_s2(rows: list[dict[str, Any]], *, model_role: str, representatives_only: bool) -> None:
    title = f"=== S2 AUROC → {model_role}_wrong"
    if representatives_only:
        title += " (family representatives)"
    print("\n" + title + " ===")
    print(f"{'slice':<16} {'family':<12} {'signal':<14} {'n':>6} {'wrong':>6} {'auroc':>8}")
    for r in rows:
        if representatives_only and not r["s2_representative"]:
            continue
        print(
            f"{r['slice']:<16} {r['family']:<12} {r['signal']:<14} "
            f"{r['n']:>6} {r['n_wrong']:>6} {r['auroc']:>8.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify {role}_{model_role}.jsonl and run S1/S2 tables"
    )
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--role", default="fit", help="corpus/signals role (fit|calib|eval)")
    ap.add_argument("--model-role", default="weak")
    ap.add_argument("--signals", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--s2-all-signals", action="store_true", help="print all signals, not just reps")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    corpus_dir = ROOT / cfg["corpus_dir"]
    out_dir = ROOT / cfg["output_dir"]
    art_dir = ROOT / cfg["artifact_dir"]

    signals_path = args.signals or resolve_signals_path(out_dir, args.role, args.model_role)
    corpus_path = args.corpus or (corpus_dir / f"queries_{args.role}.jsonl")
    manifest_path = args.manifest or (art_dir / "manifest.json")
    print(f"signals: {signals_path}", flush=True)

    if not args.skip_verify:
        vreport = verify(
            signals_path=signals_path,
            corpus_path=corpus_path,
            manifest_path=manifest_path,
            role=args.role,
            model_role=args.model_role,
        )
        print_verify(vreport)
        if args.verify_only:
            return 0 if vreport["ok"] else 1
        if not vreport["ok"]:
            print("\nAnalysis skipped: fix verification issues first (or pass --skip-verify).", flush=True)
            return 1

    corpus_by_id = {r["query_id"]: r for r in read_jsonl(corpus_path)}
    signals = read_jsonl(signals_path)
    joined = join_rows(signals, corpus_by_id, model_role=args.model_role)

    s1 = s1_table(joined)
    s2 = s2_table(joined)
    print_s1(s1, model_role=args.model_role)
    print_s2(s2, model_role=args.model_role, representatives_only=not args.s2_all_signals)

    if args.json_out:
        payload = {
            "role": args.role,
            "model_role": args.model_role,
            "s1": s1,
            "s2": s2,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
