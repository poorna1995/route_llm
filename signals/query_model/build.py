#!/usr/bin/env python3
"""Build model-dependent signals H(Y|q,M), U(q|M) for fit/calib/eval.

Stages (WORKFLOW.md §5):
  A  load corpus + model pool
  B  paraphrase question surfaces (optional)
  C  probe each M ∈ {weak, strong}
  D  checkpoint flush → processed/{role}_{model_role}.jsonl (atomic upsert) + manifest

Resume (default): skip query_ids already present in the output file; flush every
--save-every new probes so a crash/interrupt keeps finished work.

Usage:
  .venv/bin/python signals/query_model/build.py --mock --limit 8
  ./run.sh query-model --backend hf --roles fit --model-roles weak --no-paraphrase
  ./run.sh query-model --backend hf --roles fit --model-roles strong --limit 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from backend import get_backend  # noqa: E402
from features import (  # noqa: E402
    cluster_entropy,
    parse_mc,
    paraphrase_u,
    scores_from_probs,
    split_hotpot_prompt,
)
from paraphrase import get_paraphraser, prompts_for_surfaces  # noqa: E402
from store import (  # noqa: E402
    read_jsonl,
    signal_path,
    upsert_by_query_id,
    write_jsonl_atomic,
)

ROLES = ("fit", "calib", "eval")

# Short names in datasets/config.yaml → HF repo ids (for records / future hf backend)
HF_MODEL_IDS: dict[str, str] = {
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_pool(
    datasets_cfg: dict[str, Any],
    experiment: str,
    *,
    model_roles: set[str] | None = None,
) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (experiment_name, [(role, short_name, hf_id), ...])."""
    pool = (datasets_cfg.get("model_pool") or {}).get(experiment)
    if not pool:
        raise SystemExit(f"unknown experiment: {experiment}")
    name = pool.get("name") or experiment
    out: list[tuple[str, str, str]] = []
    for role in ("weak", "strong"):
        if model_roles and role not in model_roles:
            continue
        short = pool[role]
        hf = HF_MODEL_IDS.get(short, short)
        out.append((role, short, hf))
    if not out:
        raise SystemExit("empty model pool after --model-roles filter")
    return name, out


def is_mc_row(row: dict[str, Any]) -> bool:
    return str(row.get("task_type", "")).endswith("_mc") or bool(parse_mc(row["prompt"])[1])


def probe_mc(
    backend,
    *,
    prompt: str,
    query_id: str,
    model_id: str,
    labels: list[str],
) -> dict[str, Any]:
    probs = backend.score_letters(prompt, labels, query_id=query_id, model_id=model_id)
    return scores_from_probs(probs)


def probe_hotpot(
    backend,
    *,
    prompt: str,
    query_id: str,
    model_id: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    hcfg = cfg.get("hotpot") or {}
    answers = backend.sample_answers(
        prompt,
        int(hcfg.get("n_samples", 5)),
        query_id=query_id,
        model_id=model_id,
        temperature=float(hcfg.get("temperature", 0.7)),
        max_new_tokens=int(hcfg.get("max_new_tokens", 48)),
        seed=int(hcfg.get("seed", 42)),
    )
    return cluster_entropy(answers)


def mc_pred(backend, prompt: str, labels: list[str], *, query_id: str, model_id: str) -> str:
    return probe_mc(backend, prompt=prompt, query_id=query_id, model_id=model_id, labels=labels)["pred"]


def hotpot_pred(
    backend,
    prompt: str,
    *,
    query_id: str,
    model_id: str,
    cfg: dict[str, Any],
    n_samples: int = 1,
) -> str:
    hcfg = cfg.get("hotpot") or {}
    answers = backend.sample_answers(
        prompt,
        n_samples,
        query_id=query_id,
        model_id=model_id,
        temperature=float(hcfg.get("temperature", 0.7)),
        max_new_tokens=int(hcfg.get("max_new_tokens", 48)),
        seed=int(hcfg.get("seed", 42)),
    )
    if n_samples <= 1:
        return answers[0]
    return cluster_entropy(answers)["pred"]


def build_record(
    row: dict[str, Any],
    *,
    model_role: str,
    model_id: str,
    experiment_name: str,
    backend,
    paraphraser,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    prompt = row["prompt"]
    query_id = row["query_id"]
    mc = is_mc_row(row)
    question, labels, texts = parse_mc(prompt)
    context_prefix = "" if mc else split_hotpot_prompt(prompt)[0]

    pcfg = cfg.get("paraphrase") or {}
    paraphrase_on = bool(pcfg.get("enabled", True))
    k = int(pcfg.get("k", 3))

    if paraphrase_on and k > 1:
        surfaces = paraphraser.surfaces(question, k, query_id=query_id)
        surface_prompts = prompts_for_surfaces(
            prompt,
            surfaces,
            labels=labels,
            texts=texts,
            is_mc=mc,
            context_prefix=context_prefix,
        )
    else:
        surface_prompts = [prompt]

    # Per-query latency / token trace (backend accumulates each forward).
    if hasattr(backend, "clear_trace"):
        backend.clear_trace()

    if mc:
        answer_scores = probe_mc(
            backend, prompt=surface_prompts[0], query_id=query_id, model_id=model_id, labels=labels
        )
        preds = [
            mc_pred(backend, p, labels, query_id=query_id, model_id=model_id) for p in surface_prompts
        ]
        probe_kind = "mc_letter"
    else:
        answer_scores = probe_hotpot(
            backend, prompt=surface_prompts[0], query_id=query_id, model_id=model_id, cfg=cfg
        )
        preds = [
            hotpot_pred(backend, p, query_id=query_id, model_id=model_id, cfg=cfg, n_samples=1)
            for p in surface_prompts
        ]
        probe_kind = "freeform_clusters"

    cost = backend.pop_trace() if hasattr(backend, "pop_trace") else {}
    # Optional $ estimate from config (USD per 1M tokens); 0 / missing → omit.
    prices = (cfg.get("pricing") or {}).get(model_id) or (cfg.get("pricing") or {}).get(model_role)
    if prices and cost:
        pt = int(cost.get("prompt_tokens", 0))
        ct = int(cost.get("completion_tokens", 0))
        in_rate = float(prices.get("input_per_mtok", 0.0))
        out_rate = float(prices.get("output_per_mtok", 0.0))
        cost["usd_est"] = round((pt * in_rate + ct * out_rate) / 1_000_000.0, 8)

    rec: dict[str, Any] = {
        "query_id": query_id,
        "model_id": model_id,
        "model_role": model_role,
        "experiment": experiment_name,
        "role": row["role"],
        "source": row["source"],
        "answer_scores": answer_scores,
        "probe": {"kind": probe_kind},
        "cost": cost,
    }
    if paraphrase_on and k > 1:
        rec["paraphrase"] = paraphrase_u(preds)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Build model-dependent query×model signals")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifacts", type=Path, default=None)
    ap.add_argument(
        "--experiment",
        choices=("primary", "ablation", "local_strong", "local_strong_gemma"),
        default=None,
    )
    ap.add_argument("--roles", default="fit,calib,eval", help="comma-separated")
    ap.add_argument("--sources", default=None, help="comma-separated filter")
    ap.add_argument("--model-roles", default=None, help="weak,strong subset (e.g. weak only)")
    ap.add_argument("--backend", choices=("mock", "hf"), default=None, help="probe backend")
    ap.add_argument("--mock", action="store_true", help="alias for --backend mock")
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per role")
    ap.add_argument(
        "--replace-all",
        action="store_true",
        help="rewrite the target {role}_{model_role}.jsonl fully (default: upsert by query_id)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="re-probe every query even if query_id already exists in the output file",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="flush completed probes to disk every N new queries (0 = only at end of role×model)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="log every N queries (0 disables)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.mock:
        backend_name = "mock"
    elif args.backend:
        backend_name = args.backend
    else:
        backend_name = cfg.get("backend", "mock")
    if args.no_paraphrase:
        cfg.setdefault("paraphrase", {})["enabled"] = False

    datasets_cfg = load_yaml(ROOT / cfg["datasets_config"])
    experiment = args.experiment or cfg.get("experiment", "primary")
    model_roles = None
    if args.model_roles:
        model_roles = {r.strip() for r in args.model_roles.split(",") if r.strip()}
    experiment_name, pool = resolve_pool(datasets_cfg, experiment, model_roles=model_roles)

    corpus_dir = args.corpus or (ROOT / cfg["corpus_dir"])
    out_dir = args.out or (ROOT / cfg["output_dir"])
    art_dir = args.artifacts or (ROOT / cfg["artifact_dir"])

    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
    sources = None
    if args.sources:
        sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    backend = get_backend(backend_name, cfg)
    paraphraser = get_paraphraser("mock")

    counts: dict[str, int] = {}
    files_written: list[str] = []
    for role in roles:
        path = corpus_dir / f"queries_{role}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing corpus: {path}")
        rows = read_jsonl(path)
        if sources:
            rows = [r for r in rows if r["source"] in sources]
        if args.limit is not None:
            rows = rows[: args.limit]

        n_rows = len(rows)
        progress_every = max(0, int(args.progress_every))
        save_every = max(0, int(args.save_every))

        # One file per model_role — weak/strong never overwrite each other.
        for model_role, _short, hf_id in pool:
            out_path = signal_path(out_dir, role, model_role)
            resume = not args.replace_all and not args.no_resume
            disk_rows: list[dict[str, Any]] = [] if args.replace_all else read_jsonl(out_path)
            done_ids = {r["query_id"] for r in disk_rows} if resume else set()
            n_skip_plan = sum(1 for r in rows if r["query_id"] in done_ids)
            n_todo = n_rows - n_skip_plan
            print(
                f"probing {role}/{model_role}: {n_rows} queries [{backend_name}] → {hf_id}"
                f"  (resume={'on' if resume else 'off'}, already_done={n_skip_plan}, todo={n_todo})",
                flush=True,
            )

            new_records: list[dict[str, Any]] = []
            n_skipped = 0
            n_probed = 0
            probed_since_flush = 0
            sum_ms = 0.0
            sum_tok = 0
            first_write = True
            last_bak: Path | None = None
            last_records: list[dict[str, Any]] = disk_rows

            def flush_checkpoint(*, reason: str) -> None:
                nonlocal disk_rows, first_write, probed_since_flush, last_bak, last_records
                if not new_records and out_path.exists() and not args.replace_all:
                    last_records = disk_rows
                    return
                if args.replace_all or not disk_rows:
                    records = list(new_records)
                    merge_note = "replace_all" if args.replace_all else "new_file"
                else:
                    records = upsert_by_query_id(disk_rows, new_records)
                    kept = len(records) - len(new_records)
                    merge_note = f"upsert kept={kept} new={len(new_records)}"
                    disk_rows = records
                bak = write_jsonl_atomic(out_path, records, backup=first_write)
                if bak is not None:
                    last_bak = bak
                first_write = False
                probed_since_flush = 0
                last_records = records
                bak_note = f", backup={bak.name}" if bak else ""
                print(
                    f"  checkpoint ({reason}): {out_path.name} → {len(records)} records"
                    f" ({merge_note}{bak_note})",
                    flush=True,
                )

            try:
                for i, row in enumerate(rows, 1):
                    qid = row["query_id"]
                    if qid in done_ids:
                        n_skipped += 1
                        if progress_every and (
                            i == 1 or i % progress_every == 0 or i == n_rows
                        ):
                            print(
                                f"  {role}/{model_role} {i}/{n_rows} skip {row['source']} "
                                f"{qid[:12]}  (probed={n_probed} skipped={n_skipped})",
                                flush=True,
                            )
                        continue

                    rec = build_record(
                        row,
                        model_role=model_role,
                        model_id=hf_id,
                        experiment_name=experiment_name,
                        backend=backend,
                        paraphraser=paraphraser,
                        cfg=cfg,
                    )
                    new_records.append(rec)
                    done_ids.add(qid)
                    n_probed += 1
                    probed_since_flush += 1
                    c = rec.get("cost") or {}
                    sum_ms += float(c.get("latency_ms", 0.0))
                    sum_tok += int(c.get("total_tokens", 0))
                    if progress_every and (i == 1 or i % progress_every == 0 or i == n_rows):
                        ms = float(c.get("latency_ms", 0.0))
                        tok = int(c.get("total_tokens", 0))
                        denom = max(n_probed, 1)
                        print(
                            f"  {role}/{model_role} {i}/{n_rows} {row['source']} "
                            f"{qid[:12]}  {ms:.0f}ms  {tok} tok"
                            f"  (avg {sum_ms / denom:.0f}ms, {sum_tok / denom:.0f} tok;"
                            f" probed={n_probed} skipped={n_skipped})",
                            flush=True,
                        )
                    if save_every and probed_since_flush >= save_every:
                        flush_checkpoint(reason=f"every {save_every}")
            except BaseException as exc:
                # Persist finished probes before dying (Ctrl-C, OOM killer leftovers, bugs).
                if new_records and probed_since_flush:
                    flush_checkpoint(reason=f"on_{type(exc).__name__}")
                raise
            else:
                if probed_since_flush or (
                    not out_path.exists() and (new_records or args.replace_all)
                ):
                    flush_checkpoint(reason="final")
                elif n_skipped and not n_probed:
                    print(
                        f"  {role}/{model_role}: nothing new to probe "
                        f"(all {n_skipped} already in {out_path.name})",
                        flush=True,
                    )

            key = f"{role}_{model_role}"
            counts[key] = len(last_records)
            try:
                files_written.append(str(out_path.relative_to(ROOT)))
            except ValueError:
                files_written.append(str(out_path))
            bak_note = f", backup={last_bak.name}" if last_bak else ""
            print(
                f"wrote {out_path.name}: {counts[key]} records "
                f"(probed={n_probed} skipped={n_skipped}{bak_note})",
                flush=True,
            )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "experiment": experiment,
        "experiment_name": experiment_name,
        "models": [{"role": r, "hf_id": h} for r, _s, h in pool],
        "counts": counts,
        "files": files_written,
        "layout": "{role}_{model_role}.jsonl",
        "paraphrase_k": (cfg.get("paraphrase") or {}).get("k"),
        "limit": args.limit,
        "replace_all": bool(args.replace_all),
        "resume": not bool(args.replace_all or args.no_resume),
        "save_every": max(0, int(args.save_every)),
        "sources_filter": sorted(sources) if sources else None,
        "config_sha16": hashlib.sha256(args.config.read_bytes()).hexdigest()[:16],
    }
    write_json(art_dir / "manifest.json", manifest)
    print("done", json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
