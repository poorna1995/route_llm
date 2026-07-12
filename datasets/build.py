#!/usr/bin/env python3
"""Build a leakage-safe offline routing corpus (HotpotQA + ARC-C + MMLU).

Usage (repo root):
  PYTHONPATH=datasets .venv/bin/python datasets/build.py
  PYTHONPATH=datasets .venv/bin/python datasets/build.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


# ── helpers ──────────────────────────────────────────────────────────

def sha16(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def qid(source: str, config: str, split: str, raw_id: str) -> str:
    return hashlib.sha256(f"{source}|{config}|{split}|{raw_id}".encode()).hexdigest()[:24]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def sample(items: list, k: int | None, seed: int, key_fn=None) -> list:
    if k is None or k >= len(items):
        out = list(items)
        random.Random(seed).shuffle(out)
        return out
    if not key_fn:
        return random.Random(seed).sample(items, k)
    buckets: dict[Any, list] = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets, key=str)
    out: list = []
    idx = {k: 0 for k in keys}
    while len(out) < k:
        moved = False
        for key in keys:
            i = idx[key]
            if i < len(buckets[key]):
                out.append(buckets[key][i])
                idx[key] = i + 1
                moved = True
                if len(out) >= k:
                    break
        if not moved:
            break
    rng.shuffle(out)
    return out[:k]


def mc_prompt(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    labels = labels or [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question.strip(), ""]
    for lab, ch in zip(labels, choices):
        lines.append(f"{lab}. {ch}")
    lines += ["", "Answer with the option letter only."]
    return "\n".join(lines)


def normalize_arc_choices(labels: list[str], texts: list[str], answer_key: str) -> tuple[list[str], list[str], str]:
    """Map 1/2/3/4 (and A/B/...) onto A/B/C/... so gold matches the prompt."""
    new_labels = [chr(ord("A") + i) for i in range(len(texts))]
    key = str(answer_key).strip()
    if key in labels:
        gold = new_labels[labels.index(key)]
    else:
        # already a letter matching position, or unexpected — best-effort
        gold = key if key in new_labels else new_labels[0]
    return new_labels, texts, gold


def row(
    *,
    source: str,
    config: str,
    hf_split: str,
    role: str,
    task_type: str,
    prompt: str,
    gold: str,
    metric: str,
    raw_id: str,
    meta: dict | None = None,
) -> dict:
    p, g = prompt.strip(), str(gold).strip()
    return {
        "query_id": qid(source, config, hf_split, raw_id),
        "source": source,
        "config": config,
        "hf_split": hf_split,
        "role": role,  # eval | fit | calib
        "task_type": task_type,
        "prompt": p,
        "gold": g,
        "metric": metric,
        "meta": meta or {},
        "text_hash": sha16(p + "||" + g),
    }


# ── builders ─────────────────────────────────────────────────────────

def build_arc(cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    out: list[dict] = []
    for role, rc in cfg["roles"].items():
        split = rc["split"]
        ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split))
        ds = sample(ds, rc.get("max_examples"), seed + sum(map(ord, role)) % 10_000)
        for ex in ds:
            labels, texts = list(ex["choices"]["label"]), list(ex["choices"]["text"])
            labels, texts, gold = normalize_arc_choices(labels, texts, ex["answerKey"])
            out.append(
                row(
                    source="arc_challenge",
                    config="ARC-Challenge",
                    hf_split=split,
                    role=role,
                    task_type="science_mc",
                    prompt=mc_prompt(ex["question"], texts, labels),
                    gold=gold,
                    metric="accuracy",
                    raw_id=str(ex["id"]),
                    meta={"raw_id": str(ex["id"]), "n_choices": len(texts)},
                )
            )
    return out


def build_hotpot(cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    if cfg.get("hf_config", "distractor") != "distractor":
        raise ValueError("only distractor (has gold on validation)")
    policy = cfg.get("context_policy", "question_only")
    roles_cfg = cfg.get("roles", {})

    def to_row(ex: dict, role: str, hf_split: str) -> dict:
        q = ex["question"].strip()
        if policy == "question_only":
            prompt = f"Question: {q}\n\nAnswer briefly."
        else:
            raise ValueError(f"unsupported context_policy={policy}")
        return row(
            source="hotpotqa",
            config="distractor",
            hf_split=hf_split,
            role=role,
            task_type="multi_hop_qa",
            prompt=prompt,
            gold=ex["answer"],
            metric="em",
            raw_id=str(ex["id"]),
            meta={
                "raw_id": str(ex["id"]),
                "level": ex.get("level"),
                "type": ex.get("type"),
                "context_policy": policy,
            },
        )

    out: list[dict] = []

    # --- train split: partition into fit / calib with NO overlap ---
    train_roles = {
        role: rc
        for role, rc in roles_cfg.items()
        if rc.get("split") == "train"
    }
    if train_roles:
        if "test" in {rc.get("split") for rc in train_roles.values()}:
            raise ValueError("hotpot test has no gold")
        train_ds = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="train"))
        # Stratify then carve calib first, then fit from remainder (researcher-safe).
        strat = sample(
            train_ds,
            None,
            seed,
            key_fn=lambda ex: (ex.get("level"), ex.get("type")),
        )
        n_calib = int((train_roles.get("calib") or {}).get("max_examples") or 0)
        n_fit = int((train_roles.get("fit") or {}).get("max_examples") or 0)
        if "calib" in train_roles and "fit" in train_roles and n_calib + n_fit > len(strat):
            raise ValueError("hotpot train: calib+fit exceeds available rows")
        cursor = 0
        if "calib" in train_roles:
            chunk = strat[cursor : cursor + n_calib] if n_calib else []
            cursor += len(chunk)
            out.extend(to_row(ex, "calib", "train") for ex in chunk)
        if "fit" in train_roles:
            chunk = strat[cursor : cursor + n_fit] if n_fit else strat[cursor:]
            out.extend(to_row(ex, "fit", "train") for ex in chunk)

    # --- other splits (e.g. validation → eval) ---
    for role, rc in roles_cfg.items():
        split = rc["split"]
        if split == "train":
            continue  # already handled
        if split == "test":
            raise ValueError("hotpot test has no gold")
        ds = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split=split))
        ds = sample(
            ds,
            rc.get("max_examples"),
            seed + 17,
            key_fn=lambda ex: (ex.get("level"), ex.get("type")),
        )
        out.extend(to_row(ex, role, split) for ex in ds)

    return out


def build_mmlu(cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    out: list[dict] = []
    seen_text: set[str] = set()  # drop duplicate question text (noise in MMLU)
    subjects = cfg["subjects"]

    def load(subject: str, split: str):
        if split == "auxiliary_train":
            raise ValueError("mmlu auxiliary_train forbidden (leakage)")
        return list(load_dataset("cais/mmlu", subject, split=split))

    def add(ex, subject: str, split: str, role: str, idx: int) -> None:
        choices = list(ex["choices"])
        ans = ex["answer"]
        gold = chr(ord("A") + int(ans)) if not isinstance(ans, str) or str(ans).isdigit() else str(ans)
        prompt = mc_prompt(ex["question"], choices)
        th = sha16(prompt + "||" + gold)
        if th in seen_text:
            return
        seen_text.add(th)
        raw = f"{subject}|{split}|{idx}|{sha16(ex['question'])}"
        out.append(
            row(
                source="mmlu",
                config=subject,
                hf_split=split,
                role=role,
                task_type="knowledge_mc",
                prompt=prompt,
                gold=gold,
                metric="accuracy",
                raw_id=raw,
                meta={"subject": subject, "row_index": idx},
            )
        )

    for role, rc in cfg["roles"].items():
        split = rc["split"]
        per = rc.get("per_subject")
        for subject in subjects:
            rows = load(subject, split)
            # stable index before sampling
            indexed = list(enumerate(rows))
            if per is not None:
                indexed = sample(indexed, int(per), seed + sum(map(ord, role + subject)) % 10_000)
            for idx, ex in indexed:
                add(ex, subject, split, role, idx)

    return out


BUILDERS = {
    "arc_challenge": build_arc,
    "hotpotqa": build_hotpot,
    "mmlu": build_mmlu,
}


# ── leakage checks ───────────────────────────────────────────────────

def check(rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("empty corpus")
    ids = [r["query_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate query_id")

    by_role: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["source"] == "mmlu" and r["hf_split"] == "auxiliary_train":
            raise SystemExit("forbidden: mmlu auxiliary_train")
        if not r["prompt"] or not r["gold"]:
            raise SystemExit(f"empty prompt/gold: {r['query_id']}")
        by_role[r["role"]].add(r["query_id"])
        if r["role"] in {"eval", "fit", "calib"}:
            by_hash[r["text_hash"]].add(r["role"])

    for a, b in (("fit", "eval"), ("calib", "eval"), ("fit", "calib")):
        overlap = by_role[a] & by_role[b]
        if overlap:
            raise SystemExit(f"{a}/{b} id overlap: {list(overlap)[:3]}")

    for h, roles in by_hash.items():
        if len(roles) > 1:
            raise SystemExit(f"text leakage across {sorted(roles)} hash={h}")


# ── main ─────────────────────────────────────────────────────────────

def apply_smoke(cfg: dict) -> dict:
    """Shrink caps for a fast dry-run; same schema and leakage rules."""
    ds = cfg["datasets"]
    ds["arc_challenge"]["roles"]["fit"]["max_examples"] = 32
    ds["arc_challenge"]["roles"]["calib"]["max_examples"] = 16
    ds["arc_challenge"]["roles"]["eval"]["max_examples"] = 32
    ds["hotpotqa"]["roles"]["fit"]["max_examples"] = 48
    ds["hotpotqa"]["roles"]["calib"]["max_examples"] = 16
    ds["hotpotqa"]["roles"]["eval"]["max_examples"] = 32
    ds["mmlu"]["subjects"] = ds["mmlu"]["subjects"][:2]
    ds["mmlu"]["roles"]["eval"]["per_subject"] = 8
    cfg["version"] = cfg.get("version", "corpus_v1") + "_smoke"
    cfg["output_dir"] = "datasets/processed/corpus_smoke"
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg = apply_smoke(cfg)

    seed = int(cfg["seed"])
    rows: list[dict] = []
    for name, ds_cfg in cfg["datasets"].items():
        if not ds_cfg.get("enabled", True):
            continue
        if name not in BUILDERS:
            raise SystemExit(f"unknown dataset {name}; add builder in build.py")
        print(f"building {name} ...", flush=True)
        rows.extend(BUILDERS[name](ds_cfg, seed))

    check(rows)

    out = args.out or Path(cfg["output_dir"])
    if not out.is_absolute():
        out = REPO / out

    by_role: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)

    for role, role_rows in by_role.items():
        write_jsonl(out / f"queries_{role}.jsonl", role_rows)
    write_jsonl(out / "queries.jsonl", rows)

    split_ids = {role: sorted(r["query_id"] for r in rs) for role, rs in by_role.items()}
    write_json(out / "split_ids.json", split_ids)

    counts_src: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts_src[r["source"]][r["role"]] += 1

    manifest = {
        "version": cfg.get("version"),
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "smoke": bool(args.smoke),
        "counts_by_role": {k: len(v) for k, v in by_role.items()},
        "counts_by_source_role": {s: dict(v) for s, v in counts_src.items()},
        "fingerprints": {
            role: hashlib.sha256("\n".join(ids).encode()).hexdigest()
            for role, ids in split_ids.items()
        },
        "model_pool": cfg.get("model_pool", []),
    }
    write_json(out / "manifest.json", manifest)
    print(f"wrote {out}")
    print("counts:", manifest["counts_by_role"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
