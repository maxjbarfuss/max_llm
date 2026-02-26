"""BPE tokenizer wrapping tiktoken encodings.

Supported encodings:
    - "gpt2"        — 50 257 token vocabulary (byte-level BPE, GPT-2 style)
    - "cl100k_base" — 100 277 token vocabulary (GPT-3.5 / GPT-4 / text-embedding-ada-002)
    - "o200k_base"  — 200 019 token vocabulary (GPT-4o)

Usage:
    tok = BPETokenizer()                  # defaults to gpt2
    tok = BPETokenizer(encoding="cl100k_base")
    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)
"""

from __future__ import annotations

import tiktoken

from .tokenizer import Tokenizer

_SUPPORTED_ENCODINGS: frozenset[str] = frozenset(
    {
        "gpt2",
        "cl100k_base",
        "o200k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    }
)


class BPETokenizer(Tokenizer):
    """Byte-pair encoding tokenizer backed by tiktoken.

    Args:
        encoding: tiktoken encoding name (default: "gpt2").

    Raises:
        ValueError: If the encoding name is not in the supported set.
    """

    def __init__(self, encoding: str = "gpt2", **_kwargs: object) -> None:
        """Create a BPE tokenizer.

        Args:
            encoding: tiktoken encoding name.
            **_kwargs: Ignored extra keyword arguments (for config-schema
                compatibility with other tokenizers that accept e.g. ``mode``).
        """
        if encoding not in _SUPPORTED_ENCODINGS:
            raise ValueError(
                f"Unknown encoding '{encoding}'. " f"Supported: {sorted(_SUPPORTED_ENCODINGS)}"
            )
        self._enc = tiktoken.get_encoding(encoding)
        self._encoding_name = encoding

    # ------------------------------------------------------------------
    # Tokenizer ABC
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text to BPE token IDs.

        Args:
            text: Input text (any Unicode string).

        Returns:
            List of integer token IDs in [0, vocab_size).
        """
        if not text:
            return []
        return list(self._enc.encode(text))

    def decode(self, tokens: list[int]) -> str:
        """Decode BPE token IDs back to text.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        if not tokens:
            return ""
        return self._enc.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text without constructing the list.

        Args:
            text: Input text string.

        Returns:
            Token count (integer ≥ 0).
        """
        if not text:
            return 0
        return len(self._enc.encode(text))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the underlying encoding."""
        return self._enc.n_vocab

    @property
    def encoding_name(self) -> str:
        """Name of the tiktoken encoding in use."""
        return self._encoding_name
