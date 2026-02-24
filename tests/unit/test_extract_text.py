"""Unit tests for extract_text utilities."""

import pytest

from src.data.pipeline.extract_text import parse_size


def test_parse_size_tokens_megabytes():
    # 1MT -> 1,000,000 tokens * 4 bytes per token
    assert parse_size("1MT") == 4_000_000


def test_parse_size_tokens_kilobytes():
    # 250KT -> 250,000 tokens * 4 bytes per token
    assert parse_size("250KT") == 1_000_000


def test_parse_size_tokens_plain():
    # 100T -> 100 tokens * 4 bytes per token
    assert parse_size("100T") == 400


def test_parse_size_tokens_invalid():
    with pytest.raises(ValueError):
        parse_size("tenMT")
