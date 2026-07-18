"""Answer uncertainty H and paraphrase U — pure math, no LLM."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any

_OPTION = re.compile(r"^([A-E])\.\s+(.*)$", re.MULTILINE)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ANSWER_SUFFIXES = ("\n\nAnswer briefly.", "\nAnswer briefly.")


def _strip_answer_suffix(text: str) -> str:
    for suffix in _ANSWER_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text.strip()


def split_hotpot_prompt(prompt: str) -> tuple[str, str]:
    """Return (context_prefix, question). context_prefix is empty for question-only prompts."""
    marker = "\n\nQuestion:"
    if prompt.startswith("Question:"):
        return "", _strip_answer_suffix(prompt[len("Question:") :].strip())
    if marker in prompt:
        prefix, rest = prompt.rsplit(marker, 1)
        return prefix.strip(), _strip_answer_suffix(rest.strip())
    return "", _strip_answer_suffix(prompt)


def rebuild_hotpot_prompt(context_prefix: str, question: str) -> str:
    q_block = f"Question: {question.strip()}\n\nAnswer briefly."
    if context_prefix:
        return f"{context_prefix.rstrip()}\n\n{q_block}"
    return q_block


def parse_mc(prompt: str) -> tuple[str, list[str], list[str]]:
    """Return (question_stem, option_labels, option_texts). Empty labels => free-form."""
    opts = [(m.group(1), m.group(2).strip()) for m in _OPTION.finditer(prompt)]
    if not opts:
        _prefix, q = split_hotpot_prompt(prompt)
        return q, [], []

    first = opts[0][0]
    idx = prompt.find(f"\n{first}. ")
    if idx < 0:
        idx = prompt.find(f"{first}. ")
    question = prompt[:idx].strip() if idx >= 0 else prompt
    for junk in ("\n\nAnswer with the option letter only.", "Answer with the option letter only."):
        question = question.replace(junk, "").strip()
    labels = [a for a, _ in opts]
    texts = [t for _, t in opts]
    return question, labels, texts


def rebuild_mc_prompt(question: str, labels: list[str], texts: list[str]) -> str:
    lines = [question.strip(), ""]
    for lab, text in zip(labels, texts):
        lines.append(f"{lab}. {text}")
    lines += ["", "Answer with the option letter only."]
    return "\n".join(lines)


def normalize_answer(text: str) -> str:
    """SQuAD / HotpotQA exact-match normalization.

    Lowercase, strip punctuation, drop articles (a/an/the), collapse whitespace.
    Used for metric==\"em\" scoring and free-form answer clustering.
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def scores_from_probs(probs: dict[str, float]) -> dict[str, Any]:
    """Option-letter distribution → answer_scores (MC)."""
    if not probs:
        raise ValueError("empty probs")
    items = sorted(probs.items(), key=lambda kv: kv[0])
    letters = [k for k, _ in items]
    p = [float(v) for _, v in items]
    s = sum(p)
    if s <= 0:
        p = [1.0 / len(p)] * len(p)
    else:
        p = [x / s for x in p]

    ranked = sorted(zip(p, letters), reverse=True)
    p_max = ranked[0][0]
    pred = ranked[0][1]
    p_second = ranked[1][0] if len(ranked) > 1 else 0.0
    h = -sum(x * math.log(x) for x in p if x > 0)

    return {
        "H": h,
        "p_max": p_max,
        "pred": pred,
        "margin": p_max - p_second,
        "top2_mass": p_max + p_second,
        "perplexity_H": math.exp(h),
        "inv_p_max": 1.0 / p_max if p_max > 0 else float("inf"),
        "surprisal": -math.log(p_max) if p_max > 0 else float("inf"),
        "probs": {letters[i]: p[i] for i in range(len(letters))},
        "n_options": len(letters),
    }


def cluster_entropy(answers: list[str]) -> dict[str, Any]:
    """Discrete cluster entropy over normalized answer strings (Hotpot v1).

    Stores raw ``samples`` so H can be recomputed offline if normalization changes,
    without re-querying the model.
    """
    if not answers:
        raise ValueError("empty answers")
    raw = [str(a) for a in answers]
    normed = [normalize_answer(a) for a in raw]
    counts = Counter(normed)
    n = len(normed)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    pred_norm, pred_count = counts.most_common(1)[0]
    # recover a representative string from the mode cluster
    pred = next(a for a, z in zip(raw, normed) if z == pred_norm)
    return {
        "H": h,
        "p_max": pred_count / n,
        "pred": pred,
        "margin": pred_count / n - (sorted(counts.values(), reverse=True)[1] / n if len(counts) > 1 else 0.0),
        "top2_mass": sum(sorted((c / n for c in counts.values()), reverse=True)[:2]),
        "perplexity_H": math.exp(h),
        "inv_p_max": n / pred_count,
        "surprisal": -math.log(pred_count / n),
        "probs": None,
        "n_options": len(counts),
        "n_samples": n,
        "n_clusters": len(counts),
        "samples": raw,
    }


def paraphrase_u(preds: list[str]) -> dict[str, Any]:
    if not preds:
        raise ValueError("empty preds")
    counts = Counter(preds)
    mode, mode_n = counts.most_common(1)[0]
    agreement = mode_n / len(preds)
    return {
        "U": 1.0 - agreement,
        "agreement": agreement,
        "k": len(preds),
        "preds": preds,
        "mode": mode,
    }
