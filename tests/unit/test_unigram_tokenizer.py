"""Tests for Unigram tokenizer backed by sentencepiece."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tokenizer import TokenizerFactory, UnigramTokenizer


def _write_corpus(path: Path) -> None:
    path.write_text(
        "the quick brown fox jumps over the lazy dog\n"
        "hello world from max llm tokenizer benchmark\n"
        "tiny stories and wikitext provide useful corpus variety\n",
        encoding="utf-8",
    )


class TestUnigramTokenizerTraining:
    def test_train_and_roundtrip(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"
        _write_corpus(corpus_path)

        model_path = UnigramTokenizer.train_model(
            input_path=corpus_path,
            model_prefix=tmp_path / "spm_unigram",
            vocab_size=64,
        )

        tok = UnigramTokenizer(model_path=model_path)
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_missing_model_path_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.model"
        with pytest.raises(FileNotFoundError, match="Unigram model not found"):
            UnigramTokenizer(model_path=missing)


class TestUnigramTokenizerFactory:
    def test_factory_creates_unigram(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"
        _write_corpus(corpus_path)

        model_path = UnigramTokenizer.train_model(
            input_path=corpus_path,
            model_prefix=tmp_path / "spm_factory",
            vocab_size=64,
        )

        tok = TokenizerFactory.create("unigram", model_path=model_path)
        assert isinstance(tok, UnigramTokenizer)
        assert tok.count_tokens("hello world") > 0
