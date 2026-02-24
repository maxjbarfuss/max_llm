"""Character-level tokenizer with multiple encoding modes.

Supports various encoding strategies:
- 'codepoint': Direct Unicode codepoint mapping (configurable vocab_size)
- 'utf8': UTF-8 byte-level encoding (vocab_size=256)
- 'utf16': UTF-16 code unit encoding (vocab_size=65536)
- 'utf32': UTF-32 encoding, same as full codepoint (vocab_size=1114112)

Each mode offers different tradeoffs:
- 'codepoint' with vocab_size=128: ASCII-only, fast, minimal parameters
- 'codepoint' with vocab_size=65536: Unicode BMP, good coverage
- 'utf8': Universal, byte-level, vocab_size=256 (good for compression)
- 'utf16': Full Unicode, larger vocab
- 'utf32': Full Unicode, largest vocab

Examples:
    >>> tok = CharTokenizer(mode='codepoint', vocab_size=128)  # ASCII-only
    >>> tok.encode('A') == [65]

    >>> tok = CharTokenizer(mode='utf8')  # Byte-level
    >>> tok.encode('A') == [65]
    >>> tok.encode('€')  # Euro sign, 3 bytes in UTF-8
    [226, 130, 172]
"""

from __future__ import annotations

from typing import Literal

from .tokenizer import Tokenizer

# Default settings
_DEFAULT_MODE = "codepoint"
_DEFAULT_VOCAB_SIZE = 128

# Vocab sizes for each mode
_MODE_VOCAB_SIZES = {
    "utf8": 256,
    "utf16": 65536,
    "utf32": 1114112,  # Full Unicode range U+0000 to U+10FFFF
}


class CharTokenizer(Tokenizer):
    """Configurable character-level tokenizer.

    Supports multiple encoding modes:
    - 'codepoint': Direct Unicode codepoint (configurable vocab_size)
    - 'utf8': UTF-8 byte-level encoding (256 tokens)
    - 'utf16': UTF-16 code units (65536 tokens)
    - 'utf32': UTF-32 / full codepoints (1,114,112 tokens)

    Args:
        mode: Encoding mode (default: 'codepoint')
        vocab_size: For 'codepoint' mode, max codepoint value (default: 128)

    Examples:
        >>> # ASCII-only for speed
        >>> tok = CharTokenizer(mode='codepoint', vocab_size=128)
        >>> tok.encode("hello")
        [104, 101, 108, 108, 111]

        >>> # UTF-8 byte-level (universal, compact vocab)
        >>> tok = CharTokenizer(mode='utf8')
        >>> tok.encode("hello")
        [104, 101, 108, 108, 111]
        >>> tok.encode("戦")  # Japanese char -> 3 UTF-8 bytes
        [230, 136, 166]

        >>> # UTF-16 for Unicode BMP
        >>> tok = CharTokenizer(mode='utf16')
        >>> tok.encode("戦")
        [25126]

        >>> # UTF-32 / full Unicode
        >>> tok = CharTokenizer(mode='utf32')
        >>> tok.encode("😀")  # Emoji outside BMP
        [128512]
    """

    def __init__(
        self,
        mode: Literal["codepoint", "utf8", "utf16", "utf32"] = _DEFAULT_MODE,
        vocab_size: int | None = None,
    ):
        """Initialize CharTokenizer with specified mode.

        Args:
            mode: Encoding mode ('codepoint', 'utf8', 'utf16', 'utf32')
            vocab_size: For 'codepoint' mode only, max vocab size (default: 128)
        """
        self.mode = mode

        # Determine vocab size based on mode
        if mode in _MODE_VOCAB_SIZES:
            self.vocab_size = _MODE_VOCAB_SIZES[mode]
            if vocab_size is not None:
                raise ValueError(
                    f"vocab_size cannot be specified for mode='{mode}' "
                    f"(fixed at {self.vocab_size})"
                )
        else:  # codepoint mode
            self.vocab_size = vocab_size if vocab_size is not None else _DEFAULT_VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs based on the configured mode.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.

        Time Complexity: O(n) where n is len(text)
        """
        if self.mode == "utf8":
            # Encode as UTF-8 bytes
            return list(text.encode("utf-8"))
        elif self.mode == "utf16":
            # Encode as UTF-16 code units (16-bit values)
            utf16_bytes = text.encode("utf-16-le")  # Little-endian, no BOM
            # Convert byte pairs to 16-bit integers
            return [
                utf16_bytes[i] | (utf16_bytes[i + 1] << 8) for i in range(0, len(utf16_bytes), 2)
            ]
        elif self.mode == "utf32":
            # Encode as UTF-32 (full codepoints)
            return [ord(c) for c in text]
        else:  # codepoint mode with vocab_size limit
            return [ord(c) for c in text if ord(c) < self.vocab_size]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text based on the configured mode.

        Args:
            tokens: Token IDs to decode.

        Returns:
            Decoded text string.

        Time Complexity: O(n) where n is len(tokens)
        """
        if self.mode == "utf8":
            # Decode from UTF-8 bytes
            byte_array = bytes(tokens)
            return byte_array.decode("utf-8", errors="ignore")
        elif self.mode == "utf16":
            # Decode from UTF-16 code units
            # Convert 16-bit integers back to byte pairs
            byte_list = []
            for token in tokens:
                byte_list.append(token & 0xFF)  # Low byte
                byte_list.append((token >> 8) & 0xFF)  # High byte
            byte_array = bytes(byte_list)
            return byte_array.decode("utf-16-le", errors="ignore")
        elif self.mode == "utf32":
            # Decode from UTF-32 (full codepoints)
            return "".join(chr(i) for i in tokens if 0 <= i <= 0x10FFFF)
        else:  # codepoint mode
            return "".join(chr(i) for i in tokens if 0 <= i < self.vocab_size)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text without full encoding.

        Args:
            text: Input text string.

        Returns:
            Number of tokens (depends on encoding mode).

        Time Complexity: O(n) where n is len(text)
        """
        if self.mode == "utf8":
            return len(text.encode("utf-8"))
        elif self.mode == "utf16":
            return len(text.encode("utf-16-le")) // 2
        elif self.mode == "utf32":
            return len(text)
        else:  # codepoint mode
            return sum(1 for c in text if ord(c) < self.vocab_size)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
