"""Row-local model-independent complexity features (no shared state, no LLMs)."""

from __future__ import annotations

import re
import statistics
import zlib
from typing import Any

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_OPTION = re.compile(r"^([A-E])\.\s+(.*)$", re.MULTILINE)
_SENTENCE = re.compile(r"[.!?]+")


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def parse_prompt(prompt: str) -> tuple[str, list[str]]:
    """Split MC prompt into question stem + option texts; else (full text, [])."""
    opts = [(m.group(1), m.group(2).strip()) for m in _OPTION.finditer(prompt)]
    if not opts:
        # Strip Hotpot wrapper if present
        q = prompt
        if q.startswith("Question:"):
            q = q[len("Question:") :].strip()
        for suffix in ("\n\nAnswer briefly.", "\nAnswer briefly."):
            if q.endswith(suffix):
                q = q[: -len(suffix)].strip()
        return q, []

    first = opts[0][0]
    # Question = text before first "A." / "B." line
    idx = prompt.find(f"\n{first}. ")
    if idx < 0:
        idx = prompt.find(f"{first}. ")
    question = prompt[:idx].strip() if idx >= 0 else prompt
    for junk in ("\n\nAnswer with the option letter only.", "Answer with the option letter only."):
        question = question.replace(junk, "").strip()
    return question, [t for _, t in opts]


def mattr(tokens: list[str], window: int) -> float:
    n = len(tokens)
    if n == 0:
        return float("nan")
    if n < window:
        return len(set(tokens)) / n
    ratios = [
        len(set(tokens[i : i + window])) / window for i in range(n - window + 1)
    ]
    return statistics.fmean(ratios)


def compression_ratio(text: str, level: int) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return float("nan")
    return len(zlib.compress(raw, level=level)) / len(raw)


def structural(prompt: str, *, mattr_window: int, zlib_level: int) -> dict[str, float]:
    question, options = parse_prompt(prompt)
    q_toks = words(question)
    opt_lens = [len(words(o)) for o in options]
    prompt_len = len(words(prompt))
    q_len = len(q_toks)
    opt_sum = sum(opt_lens) if opt_lens else 0
    return {
        "prompt_token_len": float(prompt_len),
        "question_token_len": float(q_len),
        "mean_option_token_len": statistics.fmean(opt_lens) if opt_lens else 0.0,
        "std_option_token_len": statistics.pstdev(opt_lens) if len(opt_lens) > 1 else 0.0,
        "question_option_ratio": (q_len / opt_sum) if opt_sum else float(q_len),
        "mattr": mattr(words(prompt), mattr_window),
        "compression_ratio": compression_ratio(prompt, zlib_level),
    }


_BLOOM_LEVEL = {
    "understand": 2,
    "apply": 3,
    "analyze": 4,
    "evaluate": 5,
    "create": 6,
}


def bloom_depth(question: str, bloom_cfg: dict[str, Any]) -> int:
    """Max revised-Bloom level from process cues on the question stem.

    Returns 0 if no cue matches. Level 1 (Remember) is intentionally unused:
    bare interrogatives are near-universal in QA and are not demand signals.
    """
    if not bloom_cfg:
        return 0
    text = question.lower()
    toks = set(words(question))
    best = 0
    for phrase in bloom_cfg.get("apply_phrases") or []:
        if phrase.lower() in text:
            best = max(best, 3)
    for phrase in bloom_cfg.get("create_phrases") or []:
        if phrase.lower() in text:
            best = max(best, 6)
    for name, level in _BLOOM_LEVEL.items():
        for w in bloom_cfg.get(name) or []:
            if w.lower() in toks:
                best = max(best, level)
                break
    return best


def linguistic_cues(prompt: str, cfg: dict[str, Any]) -> dict[str, float]:
    question, _ = parse_prompt(prompt)
    text = prompt.lower()
    toks = set(words(prompt))
    q_toks = set(words(question))
    ling = cfg.get("linguistic") or {}

    def hit_count(lexicon: list[str], token_set: set[str], haystack: str) -> int:
        return sum(1 for w in lexicon if w.lower() in token_set or w.lower() in haystack)

    multihop = ling.get("multihop") or []
    domains = ling.get("domains") or {}

    domain_hits = 0
    for terms in domains.values():
        if any(t.lower() in toks or t.lower() in text for t in terms):
            domain_hits += 1

    markers = cfg.get("requirement_markers") or []
    n_req = sum(text.count(m.lower()) for m in markers)
    # numbered steps 1. 2. or (1) (2)
    n_req += len(re.findall(r"(?:^|\s)(?:\d+[\.)]|[\-\*])\s+\w", prompt))
    # count on stem only so MC "A. / B." markers do not inflate
    n_sentences = max(1, len([s for s in _SENTENCE.split(question) if s.strip()]))
    depth = bloom_depth(question, ling.get("bloom") or {})

    return {
        "n_requirements": float(n_req),
        # ordinal Bloom depth (0 or 2..6); name kept for C_linguistic wiring
        "reasoning_depth_score": float(depth),
        "multihop_score": float(hit_count(multihop, q_toks, question.lower())),
        "domain_breadth": float(domain_hits),
        "n_question_marks": float(prompt.count("?")),
        "n_sentences": float(n_sentences),
    }


def task_form(row: dict[str, Any]) -> dict[str, Any]:
    is_mc = 1 if str(row.get("task_type", "")).endswith("_mc") else 0
    n_choices = int((row.get("meta") or {}).get("n_choices") or 0)
    if is_mc and n_choices == 0:
        _, opts = parse_prompt(row["prompt"])
        n_choices = len(opts)
    return {
        "is_mc": is_mc,
        "n_choices": n_choices,
        "source": row["source"],
        "task_type": row["task_type"],
    }


def extract_row(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Stage A: structural + linguistic_cues + task_form from one corpus row."""
    return {
        "query_id": row["query_id"],
        "role": row["role"],
        "source": row["source"],
        "structural": structural(
            row["prompt"],
            mattr_window=int(cfg.get("mattr_window", 50)),
            zlib_level=int(cfg.get("zlib_level", 6)),
        ),
        "linguistic_cues": linguistic_cues(row["prompt"], cfg),
        "task_form": task_form(row),
    }
