"""Question-text paraphrase surfaces for U(q|M). Mock in S0; T5 later."""

from __future__ import annotations

import hashlib
from typing import Protocol

from features import rebuild_hotpot_prompt, rebuild_mc_prompt


class Paraphraser(Protocol):
    def surfaces(self, question: str, k: int, *, query_id: str) -> list[str]:
        ...


class MockParaphraser:
    """Deterministic rewrites — original + (k-1) variants."""

    _PREFIXES = ("In other words:", "Restated:", "Briefly:")

    def surfaces(self, question: str, k: int, *, query_id: str) -> list[str]:
        k = max(1, k)
        out = [question.strip()]
        h = hashlib.sha256(query_id.encode()).hexdigest()
        for i in range(1, k):
            prefix = self._PREFIXES[(int(h[i : i + 2], 16) + i) % len(self._PREFIXES)]
            out.append(f"{prefix} {question.strip()}")
        return out


def prompts_for_surfaces(
    base_prompt: str,
    surfaces: list[str],
    *,
    labels: list[str],
    texts: list[str],
    is_mc: bool,
    context_prefix: str = "",
) -> list[str]:
    if not is_mc:
        return [rebuild_hotpot_prompt(context_prefix, s) for s in surfaces]
    return [rebuild_mc_prompt(s, labels, texts) for s in surfaces]


def get_paraphraser(name: str = "mock") -> Paraphraser:
    if name == "mock":
        return MockParaphraser()
    raise NotImplementedError(f"paraphraser {name!r} not implemented — use mock")
