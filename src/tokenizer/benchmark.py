"""Tokenizer benchmarking utilities for Phase 3 comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .tokenizer import Tokenizer


@dataclass(frozen=True)
class TokenizerBenchmarkResult:
    tokenizer_name: str
    num_chars: int
    num_tokens: int
    chars_per_token: float
    unique_tokens: int
    unique_token_ratio: float
    encode_tokens_per_sec: float


def benchmark_tokenizer(
    tokenizer_name: str,
    tokenizer: Tokenizer,
    text: str,
    runs: int = 3,
) -> TokenizerBenchmarkResult:
    """Benchmark a tokenizer on a given text sample."""
    if runs <= 0:
        raise ValueError("runs must be > 0")

    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    if token_count == 0:
        raise ValueError("Tokenizer produced zero tokens for provided text")

    elapsed = 0.0
    for _ in range(runs):
        start = perf_counter()
        tokenizer.encode(text)
        elapsed += perf_counter() - start

    avg_elapsed = elapsed / runs
    unique_count = len(set(tokens))

    return TokenizerBenchmarkResult(
        tokenizer_name=tokenizer_name,
        num_chars=len(text),
        num_tokens=token_count,
        chars_per_token=len(text) / token_count,
        unique_tokens=unique_count,
        unique_token_ratio=unique_count / token_count,
        encode_tokens_per_sec=token_count / avg_elapsed if avg_elapsed > 0 else float("inf"),
    )
