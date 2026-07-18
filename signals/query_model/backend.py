"""Probe candidate model M on query q — mock (S0) or HuggingFace (real)."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Protocol

# Avoid Triton compile on aarch64 without python3-dev (GB10 / DGX Spark)
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
# Faster Hub shard downloads when hf_transfer is installed (must be set before hub import).
# No-op for an already-running process; takes effect on the next query-model launch.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _empty_trace() -> dict[str, float | int]:
    return {
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_calls": 0,
    }


def aggregate_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-call probe events into one cost block."""
    out = _empty_trace()
    kinds: list[str] = []
    for e in events:
        out["latency_ms"] = float(out["latency_ms"]) + float(e.get("latency_ms", 0.0))
        out["prompt_tokens"] = int(out["prompt_tokens"]) + int(e.get("prompt_tokens", 0))
        out["completion_tokens"] = int(out["completion_tokens"]) + int(e.get("completion_tokens", 0))
        out["n_calls"] = int(out["n_calls"]) + int(e.get("n_calls", 1))
        if e.get("kind"):
            kinds.append(str(e["kind"]))
    out["total_tokens"] = int(out["prompt_tokens"]) + int(out["completion_tokens"])
    out["latency_ms"] = round(float(out["latency_ms"]), 3)
    if kinds:
        out["kinds"] = kinds
    return out


class ModelBackend(Protocol):
    def clear_trace(self) -> None: ...

    def pop_trace(self) -> dict[str, Any]: ...

    def score_letters(self, prompt: str, letters: list[str], *, query_id: str, model_id: str) -> dict[str, float]:
        ...

    def sample_answers(
        self,
        prompt: str,
        n: int,
        *,
        query_id: str,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        ...


class MockBackend:
    """Deterministic fake probes keyed by (query_id, model_id) — schema / CI only."""

    def __init__(self) -> None:
        self._trace: list[dict[str, Any]] = []

    def clear_trace(self) -> None:
        self._trace = []

    def pop_trace(self) -> dict[str, Any]:
        agg = aggregate_trace(self._trace)
        self._trace = []
        return agg

    def _seed(self, query_id: str, model_id: str, salt: str = "") -> int:
        h = hashlib.sha256(f"{query_id}|{model_id}|{salt}".encode()).hexdigest()
        return int(h[:8], 16)

    def score_letters(self, prompt: str, letters: list[str], *, query_id: str, model_id: str) -> dict[str, float]:
        if not letters:
            raise ValueError("no letters")
        t0 = time.perf_counter()
        rng = self._seed(query_id, model_id, "mc")
        logits = [((rng >> (i * 5)) % 97) + 1 for i in range(len(letters))]
        logits[0] += (rng % 11)
        z = sum(math.exp(x / 10.0) for x in logits)
        probs = {L: math.exp(logits[i] / 10.0) / z for i, L in enumerate(letters)}
        # Approximate tokens for schema/tests (whitespace split).
        pt = max(1, len(prompt.split()))
        self._trace.append(
            {
                "kind": "mc_letter",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "prompt_tokens": pt,
                "completion_tokens": 0,
                "n_calls": 1,
            }
        )
        return probs

    def sample_answers(
        self,
        prompt: str,
        n: int,
        *,
        query_id: str,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        t0 = time.perf_counter()
        pool = ["paris", "london", "berlin", "rome", "madrid"]
        out: list[str] = []
        base = self._seed(query_id, model_id, f"hotpot|{seed}")
        for i in range(n):
            out.append(pool[(base + i * 3) % len(pool)])
        pt = max(1, len(prompt.split()))
        self._trace.append(
            {
                "kind": "freeform_sample",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "prompt_tokens": pt,
                "completion_tokens": n * min(8, max_new_tokens),
                "n_calls": 1,
            }
        )
        return out


def _letter_token_id(tokenizer, letter: str) -> int:
    """First-token id for option letter at generation position."""
    for text in (letter, f" {letter}", f"\n{letter}"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if not ids:
        raise ValueError(f"cannot tokenize letter {letter!r}")
    return ids[-1]


class HFBackend:
    """Real LLM probes: option-letter logprobs (MC) + short samples (Hotpot)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self._cache: dict[str, tuple[Any, Any]] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._trace: list[dict[str, Any]] = []

    def clear_trace(self) -> None:
        self._trace = []

    def pop_trace(self) -> dict[str, Any]:
        agg = aggregate_trace(self._trace)
        self._trace = []
        return agg

    def _record(
        self,
        *,
        kind: str,
        t0: float,
        prompt_tokens: int,
        completion_tokens: int = 0,
        n_calls: int = 1,
    ) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        self._trace.append(
            {
                "kind": kind,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "n_calls": int(n_calls),
            }
        )

    def _from_pretrained(self, cls, model_id: str, **kwargs):
        """Prefer local cache so a bad/expired HF token cannot block a warm cache."""
        try:
            return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
        except Exception as e:
            print(f"local cache miss for {model_id} ({type(e).__name__}); trying Hub …", flush=True)
            return cls.from_pretrained(model_id, **kwargs)

    def _load(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]
        print(f"loading {model_id} on {self.device} ...", flush=True)
        tok = self._from_pretrained(AutoTokenizer, model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        offload_dir = Path(self.cfg.get("offload_dir") or ".cache/hf_offload")
        offload_dir.mkdir(parents=True, exist_ok=True)
        load_kw: dict[str, Any] = {
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "attn_implementation": "eager",
            "offload_folder": str(offload_dir),
            "offload_state_dict": True,
        }
        if self.device == "cuda":
            load_kw["device_map"] = "auto"
        if self.device == "cuda" and ("70b" in model_id.lower() or "70B" in model_id):
            # accelerate expects GPU keys as ints (0, 1, ...), not "cuda:0"
            load_kw["max_memory"] = {
                0: str(self.cfg.get("max_memory_cuda") or "26GiB"),
                "cpu": str(self.cfg.get("max_memory_cpu") or "90GiB"),
            }
        # 70B: prefer 4-bit via BitsAndBytesConfig (transformers≥4.x / 5.x reject bare load_in_4bit=)
        if "70b" in model_id.lower() or "70B" in model_id:
            prequant_id = str(
                self.cfg.get("strong_prequant_model_id")
                or os.getenv("STRONG_PREQUANT_MODEL_ID", "")
            ).strip()
            if prequant_id:
                try:
                    print(f"trying pre-quantized strong model: {prequant_id}", flush=True)
                    prequant_tok = AutoTokenizer.from_pretrained(prequant_id, use_fast=True)
                    if prequant_tok.pad_token is None:
                        prequant_tok.pad_token = prequant_tok.eos_token
                    model = AutoModelForCausalLM.from_pretrained(prequant_id, **load_kw)
                    if self.device == "cpu" and not hasattr(model, "hf_device_map"):
                        model = model.to(self.device)
                    model.eval()
                    self._cache[model_id] = (model, prequant_tok)
                    return model, prequant_tok
                except Exception as e:
                    print(
                        f"pre-quantized load failed for {prequant_id}: {e}; falling back",
                        flush=True,
                    )

            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig

                bnb_kw = dict(load_kw)
                bnb_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("using bitsandbytes 4-bit (nf4) load", flush=True)
                model = self._from_pretrained(AutoModelForCausalLM, model_id, **bnb_kw)
            except ImportError:
                print("bitsandbytes unavailable; loading 70B in bf16 (needs ~140GB)", flush=True)
                model = self._from_pretrained(AutoModelForCausalLM, model_id, **load_kw)
        else:
            model = self._from_pretrained(AutoModelForCausalLM, model_id, **load_kw)
        if self.device == "cpu" and not hasattr(model, "hf_device_map"):
            model = model.to(self.device)
        model.eval()
        self._cache[model_id] = (model, tok)
        return model, tok

    def _chat_prompt(self, tokenizer, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score_letters(self, prompt: str, letters: list[str], *, query_id: str, model_id: str) -> dict[str, float]:
        if not letters:
            raise ValueError("no letters")
        model, tok = self._load(model_id)
        text = self._chat_prompt(tok, prompt)
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        out: dict[str, float] = {}
        for L in letters:
            tid = _letter_token_id(tok, L)
            out[L] = math.exp(log_probs[tid].item())
        self._record(kind="mc_letter", t0=t0, prompt_tokens=prompt_tokens, completion_tokens=0)
        return out

    def sample_answers(
        self,
        prompt: str,
        n: int,
        *,
        query_id: str,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        """Sample n free-form answers.

        For temperature > 0, uses one ``generate(..., num_return_sequences=n)``
        call (batched decode) instead of n sequential generates.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        model, tok = self._load(model_id)
        text = self._chat_prompt(tok, prompt)
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Greedy is deterministic — one decode, replicate if n > 1.
        if temperature <= 0:
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.eos_token_id,
                    do_sample=False,
                )
            completion = int(out_ids.shape[1] - prompt_len)
            self._record(
                kind="freeform_sample",
                t0=t0,
                prompt_tokens=prompt_len,
                completion_tokens=completion,
            )
            ans = tok.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()
            return [ans] * n

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=n,
            )
        # Batched generate: count completion tokens across all sequences.
        completion = int((out_ids.shape[1] - prompt_len) * out_ids.shape[0])
        self._record(
            kind="freeform_sample",
            t0=t0,
            prompt_tokens=prompt_len * n,  # billed / accounted per sequence
            completion_tokens=completion,
        )
        return [
            tok.decode(out_ids[i, prompt_len:], skip_special_tokens=True).strip()
            for i in range(out_ids.shape[0])
        ]


def get_backend(name: str, cfg: dict[str, Any] | None = None) -> ModelBackend:
    if name == "mock":
        return MockBackend()
    if name == "hf":
        return HFBackend(cfg)
    raise ValueError(f"unknown backend: {name}")
