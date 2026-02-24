"""Character-level tokenizer over the 128-character ASCII range.

Each character maps directly to its ASCII ordinal (0–127).
Characters outside this range are silently dropped during encoding.

The CharTokenizer provides a simple, deterministic encoding where:
- encode('A') == [65]
- decode([65]) == 'A'
- Non-ASCII characters (ordinal >= 128) are silently dropped
"""

from __future__ import annotations

from .tokenizer import Tokenizer

# ASCII range: 0–127 (128 characters total)
_VOCAB_SIZE = 128


class CharTokenizer(Tokenizer):
    """Bijective tokenizer mapping ASCII characters to token IDs.

    Properties:
        - Deterministic: same input always produces same output
        - Fast: O(n) encode/decode with single pass
        - Simple: token ID = character ordinal
        - Lossless for ASCII: roundtrip guaranteed for [0, 127]

    Examples:
        >>> tok = CharTokenizer()
        >>> tok.encode("hello")
        [104, 101, 108, 108, 111]
        >>> tok.decode([104, 101, 108, 108, 111])
        'hello'
        >>> tok.count_tokens("hello")
        5
    """

    vocab_size: int = _VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        """Encode text to ASCII token IDs.

        Non-ASCII characters (ordinal >= 128) are silently discarded.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs in [0, 127].

        Time Complexity: O(n) where n is len(text)
        """
        return [ord(c) for c in text if ord(c) < _VOCAB_SIZE]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text.

        IDs outside [0, 127] are silently discarded.

        Args:
            tokens: Token IDs to decode.

        Returns:
            Decoded text string.

        Time Complexity: O(n) where n is len(tokens)
        """
        return "".join(chr(i) for i in tokens if 0 <= i < _VOCAB_SIZE)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text without full encoding.

        Equivalent to len(self.encode(text)) but may be faster
        in some implementations.

        Args:
            text: Input text string.

        Returns:
            Number of valid ASCII tokens.

        Time Complexity: O(n) where n is len(text)
        """
        return sum(1 for c in text if ord(c) < _VOCAB_SIZE)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
