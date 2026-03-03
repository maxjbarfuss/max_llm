"""HuggingFace tokenizers-backed BPE tokenizer for custom small-vocab models."""

from __future__ import annotations

from tokenizers import Tokenizer as HFTokenizer

from .tokenizer import Tokenizer


class HFBPETokenizer(Tokenizer):
    """BPE tokenizer backed by a saved HuggingFace ``tokenizers`` vocab file.

    Args:
        vocab_path: Path to a ``.json`` file produced by ``tokenizer.save()``.
    """

    def __init__(self, vocab_path: str, **_kwargs: object) -> None:
        self._tok = HFTokenizer.from_file(vocab_path)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tok.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return self._tok.decode(tokens)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text).ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
