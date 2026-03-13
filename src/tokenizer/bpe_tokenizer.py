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
        if encoding not in _SUPPORTED_ENCODINGS:
            raise ValueError(
                f"Unknown encoding '{encoding}'. " f"Supported: {sorted(_SUPPORTED_ENCODINGS)}"
            )
        self._enc = tiktoken.get_encoding(encoding)
        self._encoding_name = encoding

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self._enc.encode(text))

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return self._enc.decode(tokens)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def encoding_name(self) -> str:
        return self._encoding_name
