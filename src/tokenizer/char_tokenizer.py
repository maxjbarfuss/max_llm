"""Character-level tokenizer over the 128-character ASCII range.

Each character maps directly to its ASCII ordinal (0–127).
Characters outside this range are silently dropped during encoding.
"""

from __future__ import annotations

_VOCAB_SIZE = 128


class CharTokenizer:
    """Bijective tokenizer mapping ASCII characters to token IDs.

    Token IDs are identical to ASCII ordinals: encode('A') == [65].
    Non-ASCII characters (ordinal >= 128) are dropped during encode.
    Out-of-range IDs are dropped during decode.
    """

    def __init__(self) -> None:
        self.vocab_size: int = _VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs.

        Args:
            text: Input string. Non-ASCII characters are silently dropped.

        Returns:
            List of integer token IDs in [0, 127].
        """
        return [ord(c) for c in text if ord(c) < _VOCAB_SIZE]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string.

        Args:
            ids: Token IDs. Values outside [0, 127] are silently dropped.

        Returns:
            Decoded string.
        """
        return "".join(chr(i) for i in ids if 0 <= i < _VOCAB_SIZE)

    def __len__(self) -> int:
        return self.vocab_size
