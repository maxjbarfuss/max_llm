"""Tests for BPE tokenizer wrapping tiktoken."""

from __future__ import annotations

import pytest

from src.tokenizer import TokenizerFactory
from src.tokenizer.bpe_tokenizer import BPETokenizer


class TestBPETokenizerConstruction:
    def test_default_construction_gpt2(self) -> None:
        tok = BPETokenizer(encoding="gpt2")
        assert tok.vocab_size == 50257

    def test_default_construction_cl100k(self) -> None:
        tok = BPETokenizer(encoding="cl100k_base")
        assert tok.vocab_size == 100277

    def test_unknown_encoding_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown encoding"):
            BPETokenizer(encoding="not_a_real_encoding")


class TestBPETokenizerEncodeDecode:
    @pytest.fixture
    def tok(self) -> BPETokenizer:
        return BPETokenizer(encoding="gpt2")

    def test_encode_returns_list_of_ints(self, tok: BPETokenizer) -> None:
        ids = tok.encode("Hello, world!")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_nonempty_text(self, tok: BPETokenizer) -> None:
        ids = tok.encode("The quick brown fox")
        assert len(ids) >= 3  # At minimum 3 tokens for 4 words

    def test_decode_roundtrip(self, tok: BPETokenizer) -> None:
        text = "Hello, world! This is a test."
        assert tok.decode(tok.encode(text)) == text

    def test_decode_roundtrip_multiline(self, tok: BPETokenizer) -> None:
        text = "Line one.\nLine two.\nLine three."
        assert tok.decode(tok.encode(text)) == text

    def test_encode_empty_string(self, tok: BPETokenizer) -> None:
        ids = tok.encode("")
        assert ids == []

    def test_decode_empty_list(self, tok: BPETokenizer) -> None:
        assert tok.decode([]) == ""

    def test_all_token_ids_in_vocab_range(self, tok: BPETokenizer) -> None:
        ids = tok.encode("Hello, world! The quick brown fox jumps over the lazy dog.")
        assert all(0 <= i < tok.vocab_size for i in ids)

    def test_encode_unicode(self, tok: BPETokenizer) -> None:
        text = "Héllo wörld — café"
        result = tok.decode(tok.encode(text))
        assert result == text


class TestBPETokenizerCountTokens:
    @pytest.fixture
    def tok(self) -> BPETokenizer:
        return BPETokenizer(encoding="gpt2")

    def test_count_tokens_matches_encode_length(self, tok: BPETokenizer) -> None:
        text = "Hello, world! The quick brown fox."
        assert tok.count_tokens(text) == len(tok.encode(text))

    def test_count_tokens_empty(self, tok: BPETokenizer) -> None:
        assert tok.count_tokens("") == 0

    def test_longer_text_more_tokens(self, tok: BPETokenizer) -> None:
        short = "Hello"
        long_text = "Hello, world! The quick brown fox jumps over the lazy dog."
        assert tok.count_tokens(long_text) > tok.count_tokens(short)


class TestBPETokenizerVocabSize:
    def test_vocab_size_gpt2(self) -> None:
        tok = BPETokenizer(encoding="gpt2")
        assert tok.vocab_size == 50257

    def test_vocab_size_is_int(self) -> None:
        tok = BPETokenizer(encoding="gpt2")
        assert isinstance(tok.vocab_size, int)


class TestBPETokenizerFactory:
    def test_factory_creates_bpe_tokenizer(self) -> None:
        tok = TokenizerFactory.create("bpe")
        assert isinstance(tok, BPETokenizer)

    def test_factory_bpe_with_encoding_kwarg(self) -> None:
        tok = TokenizerFactory.create("bpe", encoding="gpt2")
        assert isinstance(tok, BPETokenizer)
        assert tok.vocab_size == 50257

    def test_factory_bpe_in_available_list(self) -> None:
        assert "bpe" in TokenizerFactory.list_available()
