"""Tests for tokenizer benchmark utility."""

from __future__ import annotations

import pytest

from src.tokenizer.benchmark import benchmark_tokenizer
from src.tokenizer.tokenizer import Tokenizer


class _WhitespaceTokenizer(Tokenizer):
    def encode(self, text: str) -> list[int]:
        if not text.strip():
            return []
        return [idx for idx, _ in enumerate(text.split(), start=1)]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


def test_benchmark_tokenizer_metrics() -> None:
    tokenizer = _WhitespaceTokenizer()
    text = "alpha beta gamma alpha"

    result = benchmark_tokenizer("ws", tokenizer, text, runs=2)

    assert result.tokenizer_name == "ws"
    assert result.num_chars == len(text)
    assert result.num_tokens == 4
    assert result.unique_tokens == 4
    assert 0.0 < result.chars_per_token
    assert 0.0 < result.unique_token_ratio <= 1.0
    assert result.encode_tokens_per_sec > 0.0


def test_benchmark_tokenizer_rejects_zero_runs() -> None:
    tokenizer = _WhitespaceTokenizer()
    with pytest.raises(ValueError, match="runs must be > 0"):
        benchmark_tokenizer("ws", tokenizer, "text", runs=0)


def test_benchmark_tokenizer_rejects_zero_tokens() -> None:
    tokenizer = _WhitespaceTokenizer()
    with pytest.raises(ValueError, match="zero tokens"):
        benchmark_tokenizer("ws", tokenizer, "   ", runs=1)
