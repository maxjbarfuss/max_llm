"""Unit tests for CharTokenizer encoding modes (UTF-8, UTF-16, UTF-32)."""

import pytest

from src.tokenizer import CharTokenizer


class TestCharTokenizerUTF8Mode:
    """UTF-8 byte-level encoding tests."""

    def test_vocab_size_utf8(self):
        """UTF-8 mode has vocab_size=256 (byte range)."""
        tok = CharTokenizer(mode="utf8")
        assert tok.vocab_size == 256

    def test_ascii_utf8_same_as_codepoint(self):
        """ASCII characters encode identically in UTF-8 and codepoint modes."""
        tok_utf8 = CharTokenizer(mode="utf8")
        tok_cp = CharTokenizer(mode="codepoint", vocab_size=128)

        text = "Hello, world! 123"
        assert tok_utf8.encode(text) == tok_cp.encode(text)

    def test_utf8_multibyte_encoding(self):
        """Non-ASCII characters encode as multiple UTF-8 bytes."""
        tok = CharTokenizer(mode="utf8")

        # Euro sign (€) is 3 bytes in UTF-8: 0xE2 0x82 0xAC
        tokens = tok.encode("€")
        assert tokens == [0xE2, 0x82, 0xAC]
        assert len(tokens) == 3

        # Japanese character (戦) is 3 bytes in UTF-8
        tokens = tok.encode("戦")
        assert len(tokens) == 3

    def test_utf8_roundtrip_multilingual(self):
        """UTF-8 mode preserves multilingual text through roundtrip."""
        tok = CharTokenizer(mode="utf8")

        texts = [
            "Hello, world!",
            "Café résumé",  # French
            "戦場のヴァルキュリア",  # Japanese
            "Привет мир",  # Russian
            "مرحبا",  # Arabic
            "你好世界",  # Chinese
            "🎉🚀😀",  # Emoji
        ]

        for text in texts:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            assert decoded == text, f"Failed roundtrip for: {text}"

    def test_utf8_count_tokens_bytes(self):
        """count_tokens returns byte count for UTF-8."""
        tok = CharTokenizer(mode="utf8")

        assert tok.count_tokens("") == 0
        assert tok.count_tokens("Hello") == 5  # ASCII: 1 byte each
        assert tok.count_tokens("€") == 3  # Euro: 3 bytes
        assert tok.count_tokens("戦") == 3  # Japanese: 3 bytes

    def test_utf8_empty_roundtrip(self):
        """Empty string roundtrips correctly in UTF-8 mode."""
        tok = CharTokenizer(mode="utf8")
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_utf8_mixed_content(self):
        """Mixed ASCII and non-ASCII content encodes correctly."""
        tok = CharTokenizer(mode="utf8")

        text = "Hello 世界!"
        tokens = tok.encode(text)
        assert tok.decode(tokens) == text

        # "Hello " is 6 bytes, "世" is 3 bytes, "界" is 3 bytes, "!" is 1 byte
        # Total: 6 + 3 + 3 + 1 = 13 bytes
        assert len(tokens) == 13


class TestCharTokenizerUTF16Mode:
    """UTF-16 code unit encoding tests."""

    def test_vocab_size_utf16(self):
        """UTF-16 mode has vocab_size=65536 (16-bit range)."""
        tok = CharTokenizer(mode="utf16")
        assert tok.vocab_size == 65536

    def test_ascii_utf16_same_as_codepoint(self):
        """ASCII characters encode identically in UTF-16 and codepoint modes."""
        tok_utf16 = CharTokenizer(mode="utf16")
        tok_cp = CharTokenizer(mode="codepoint", vocab_size=65536)

        text = "Hello, world!"
        assert tok_utf16.encode(text) == tok_cp.encode(text)

    def test_utf16_bmp_single_unit(self):
        """BMP characters (U+0000 to U+FFFF) encode as single UTF-16 units."""
        tok = CharTokenizer(mode="utf16")

        # Japanese character in BMP
        tokens = tok.encode("戦")
        assert len(tokens) == 1
        assert tokens[0] == 25126  # U+6226

    def test_utf16_roundtrip_bmp(self):
        """UTF-16 mode preserves BMP text through roundtrip."""
        tok = CharTokenizer(mode="utf16")

        texts = [
            "Hello",
            "Café",
            "戦場のヴァルキュリア",
            "Привет",
            "你好",
        ]

        for text in texts:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            assert decoded == text, f"Failed roundtrip for: {text}"

    def test_utf16_surrogate_pairs(self):
        """UTF-16 handles emoji and surrogate pairs correctly."""
        tok = CharTokenizer(mode="utf16")

        # Emoji outside BMP require surrogate pairs in UTF-16
        text = "🎉"  # U+1F389
        tokens = tok.encode(text)
        # UTF-16 encodes this as 2 code units (surrogate pair)
        assert len(tokens) == 2

        # Roundtrip should work
        assert tok.decode(tokens) == text

    def test_utf16_count_tokens(self):
        """count_tokens returns UTF-16 code unit count."""
        tok = CharTokenizer(mode="utf16")

        assert tok.count_tokens("") == 0
        assert tok.count_tokens("Hello") == 5
        assert tok.count_tokens("戦") == 1  # BMP character: 1 code unit
        assert tok.count_tokens("🎉") == 2  # Outside BMP: 2 code units (surrogate pair)


class TestCharTokenizerUTF32Mode:
    """UTF-32 / full Unicode codepoint encoding tests."""

    def test_vocab_size_utf32(self):
        """UTF-32 mode has vocab_size=1114112 (full Unicode range)."""
        tok = CharTokenizer(mode="utf32")
        assert tok.vocab_size == 1114112  # U+0000 to U+10FFFF

    def test_ascii_utf32_same_as_codepoint(self):
        """UTF-32 is equivalent to codepoint mode."""
        tok_utf32 = CharTokenizer(mode="utf32")
        tok_cp = CharTokenizer(mode="codepoint", vocab_size=1114112)

        text = "Hello"
        assert tok_utf32.encode(text) == tok_cp.encode(text)

    def test_utf32_one_codepoint_per_char(self):
        """Each character encodes as single token (full codepoint)."""
        tok = CharTokenizer(mode="utf32")

        assert tok.encode("A") == [65]
        assert tok.encode("戦") == [25126]  # U+6226
        assert tok.encode("🎉") == [127881]  # U+1F389

    def test_utf32_roundtrip_full_unicode(self):
        """UTF-32 preserves all Unicode text through roundtrip."""
        tok = CharTokenizer(mode="utf32")

        texts = [
            "Hello, world!",
            "Café résumé",
            "戦場のヴァルキュリア",
            "Привет мир",
            "你好世界",
            "🎉🚀😀💯",  # Emoji outside BMP
            "𝕳𝖊𝖑𝖑𝖔",  # Mathematical bold text
        ]

        for text in texts:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            assert decoded == text, f"Failed roundtrip for: {text}"

    def test_utf32_count_tokens_is_char_count(self):
        """count_tokens returns character count for UTF-32."""
        tok = CharTokenizer(mode="utf32")

        assert tok.count_tokens("") == 0
        assert tok.count_tokens("Hello") == 5
        assert tok.count_tokens("戦場") == 2
        assert tok.count_tokens("🎉🚀😀") == 3  # 3 emoji = 3 characters

    def test_utf32_emoji_handling(self):
        """UTF-32 handles emoji as single tokens."""
        tok = CharTokenizer(mode="utf32")

        tokens = tok.encode("😀")
        assert len(tokens) == 1
        assert tokens[0] == 128512  # U+1F600


class TestCharTokenizerCodepointMode:
    """Codepoint mode with custom vocab_size tests."""

    def test_codepoint_default_vocab(self):
        """Default codepoint mode uses vocab_size=128."""
        tok = CharTokenizer()
        assert tok.mode == "codepoint"
        assert tok.vocab_size == 128

    def test_codepoint_custom_vocab(self):
        """Codepoint mode accepts custom vocab_size."""
        tok = CharTokenizer(mode="codepoint", vocab_size=256)
        assert tok.vocab_size == 256

        # Extended ASCII character (ordinal 200)
        assert tok.encode(chr(200)) == [200]

    def test_codepoint_drops_out_of_range(self):
        """Characters >= vocab_size are dropped."""
        tok = CharTokenizer(mode="codepoint", vocab_size=128)

        text = "hello世界"
        tokens = tok.encode(text)
        assert tokens == [104, 101, 108, 108, 111]  # Only ASCII kept
        assert tok.decode(tokens) == "hello"

    def test_codepoint_bmp_coverage(self):
        """vocab_size=65536 covers Unicode BMP."""
        tok = CharTokenizer(mode="codepoint", vocab_size=65536)

        text = "Hello 世界 café"
        tokens = tok.encode(text)
        assert tok.decode(tokens) == text


class TestCharTokenizerModeValidation:
    """Mode parameter validation tests."""

    def test_utf8_rejects_vocab_size(self):
        """UTF-8 mode raises error if vocab_size is specified."""
        with pytest.raises(ValueError, match="vocab_size cannot be specified"):
            CharTokenizer(mode="utf8", vocab_size=512)

    def test_utf16_rejects_vocab_size(self):
        """UTF-16 mode raises error if vocab_size is specified."""
        with pytest.raises(ValueError, match="vocab_size cannot be specified"):
            CharTokenizer(mode="utf16", vocab_size=512)

    def test_utf32_rejects_vocab_size(self):
        """UTF-32 mode raises error if vocab_size is specified."""
        with pytest.raises(ValueError, match="vocab_size cannot be specified"):
            CharTokenizer(mode="utf32", vocab_size=512)

    def test_codepoint_accepts_vocab_size(self):
        """Codepoint mode accepts vocab_size parameter."""
        tok = CharTokenizer(mode="codepoint", vocab_size=512)
        assert tok.vocab_size == 512


class TestCharTokenizerFactoryWithModes:
    """TokenizerFactory tests with new modes."""

    def test_factory_creates_default_codepoint(self):
        """Factory creates codepoint mode by default."""
        from src.tokenizer import TokenizerFactory

        tok = TokenizerFactory.create("char")
        assert tok.mode == "codepoint"
        assert tok.vocab_size == 128

    def test_factory_accepts_mode_kwarg(self):
        """Factory passes mode kwarg to CharTokenizer."""
        from src.tokenizer import TokenizerFactory

        tok = TokenizerFactory.create("char", mode="utf8")
        assert tok.mode == "utf8"
        assert tok.vocab_size == 256

    def test_factory_accepts_vocab_size_kwarg(self):
        """Factory passes vocab_size kwarg to CharTokenizer."""
        from src.tokenizer import TokenizerFactory

        tok = TokenizerFactory.create("char", mode="codepoint", vocab_size=256)
        assert tok.vocab_size == 256


class TestCharTokenizerComparison:
    """Compare encoding efficiency across modes."""

    def test_encoding_comparison_ascii(self):
        """ASCII text encodes identically across all modes."""
        text = "Hello, world!"

        tok_cp = CharTokenizer(mode="codepoint", vocab_size=128)
        tok_utf8 = CharTokenizer(mode="utf8")
        tok_utf16 = CharTokenizer(mode="utf16")
        tok_utf32 = CharTokenizer(mode="utf32")

        tokens_cp = tok_cp.encode(text)
        tokens_utf8 = tok_utf8.encode(text)
        tokens_utf16 = tok_utf16.encode(text)
        tokens_utf32 = tok_utf32.encode(text)

        # All should produce same tokens for ASCII
        assert tokens_cp == tokens_utf8 == tokens_utf16 == tokens_utf32

    def test_encoding_comparison_multilingual(self):
        """Compare token counts for multilingual text."""
        text = "Hello 戦場 🎉"  # ASCII + Japanese + Emoji

        tok_utf8 = CharTokenizer(mode="utf8")
        tok_utf16 = CharTokenizer(mode="utf16")
        tok_utf32 = CharTokenizer(mode="utf32")

        # UTF-8: Variable bytes per character
        # "Hello " = 6, "戦" = 3, "場" = 3, " " = 1, "🎉" = 4 = 17 tokens
        assert tok_utf8.count_tokens(text) == 17

        # UTF-16: BMP = 1 unit, emoji = 2 units (surrogate pair)
        # "Hello 戦場 " = 9, "🎉" = 2 = 11 tokens
        assert tok_utf16.count_tokens(text) == 11

        # UTF-32: 1 codepoint per character
        # 10 characters = 10 tokens (Hello=5, space=1, 戦場=2, space=1, 🎉=1)
        assert tok_utf32.count_tokens(text) == 10
