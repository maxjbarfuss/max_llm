"""Unit tests for the character-level tokenizer."""

import pytest

from src.tokenizer import CharTokenizer, TokenizerFactory


class TestCharTokenizerBasics:
    """Core encode/decode behaviour."""

    def test_vocab_size(self):
        """Tokenizer exposes the 128-character ASCII vocab size."""
        tok = CharTokenizer()
        assert tok.vocab_size == 128

    def test_len(self):
        """len(tokenizer) equals vocab_size."""
        tok = CharTokenizer()
        assert len(tok) == 128

    def test_encode_empty(self):
        """Encoding an empty string returns an empty list."""
        tok = CharTokenizer()
        assert tok.encode("") == []

    def test_decode_empty(self):
        """Decoding an empty list returns an empty string."""
        tok = CharTokenizer()
        assert tok.decode([]) == ""

    def test_encode_returns_ordinals(self):
        """Each ASCII character encodes to its ordinal value."""
        tok = CharTokenizer()
        assert tok.encode("ABC") == [65, 66, 67]
        assert tok.encode("abc") == [97, 98, 99]
        assert tok.encode(" ") == [32]

    def test_decode_returns_chars(self):
        """Each valid token ID decodes to its corresponding character."""
        tok = CharTokenizer()
        assert tok.decode([65, 66, 67]) == "ABC"
        assert tok.decode([72, 101, 108, 108, 111]) == "Hello"

    def test_count_tokens_empty(self):
        """Counting tokens in empty string returns 0."""
        tok = CharTokenizer()
        assert tok.count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Token count matches encoded length for ASCII strings."""
        tok = CharTokenizer()
        assert tok.count_tokens("hello") == 5
        assert tok.count_tokens("Hello, world!") == 13


class TestCharTokenizerRoundtrip:
    """Encode → decode roundtrip guarantees."""

    def test_roundtrip_simple(self):
        """ASCII text survives encode → decode unchanged."""
        tok = CharTokenizer()
        for text in ["Hello, world!", "the quick brown fox", "0123 test\n"]:
            assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_all_ascii(self):
        """All 128 ASCII characters survive roundtrip."""
        tok = CharTokenizer()
        text = "".join(chr(i) for i in range(128))
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_empty(self):
        """Empty string survives roundtrip."""
        tok = CharTokenizer()
        assert tok.decode(tok.encode("")) == ""


class TestCharTokenizerEdgeCases:
    """Boundary and out-of-range behaviour."""

    def test_non_ascii_dropped_on_encode(self):
        """Characters with ordinal >= 128 are silently dropped during encode."""
        tok = CharTokenizer()
        # 'é' (233), '中' (20013) are both > 127
        assert tok.encode("caf\xe9") == [99, 97, 102]  # 'café' → 'caf'
        assert tok.encode("a\u4e2db") == [97, 98]  # 'a中b' → 'ab'

    def test_non_ascii_text_partial_roundtrip(self):
        """ASCII portions of mixed text roundtrip; non-ASCII is stripped."""
        tok = CharTokenizer()
        ids = tok.encode("hello \xff world")
        assert tok.decode(ids) == "hello  world"  # \xff dropped, space preserved

    def test_non_ascii_count_tokens(self):
        """Token count only counts valid ASCII characters."""
        tok = CharTokenizer()
        assert tok.count_tokens("café") == 3  # 'é' not counted
        assert tok.count_tokens("a\u4e2db") == 2  # '中' not counted

    def test_encode_ids_in_valid_range(self):
        """All IDs produced by encode are in [0, 127]."""
        tok = CharTokenizer()
        ids = tok.encode("".join(chr(i) for i in range(128)))
        assert all(0 <= i <= 127 for i in ids)

    def test_decode_ignores_out_of_range_ids(self):
        """IDs outside [0, 127] are silently dropped during decode."""
        tok = CharTokenizer()
        assert tok.decode([65, 200, 66]) == "AB"  # 200 dropped
        assert tok.decode([-1, 65]) == "A"  # -1 dropped
        assert tok.decode([128, 129, 130]) == ""  # all out of range

    def test_encode_single_chars(self):
        """Single-character encoding matches ord() exactly."""
        tok = CharTokenizer()
        for i in range(128):
            assert tok.encode(chr(i)) == [i]

    def test_encode_preserves_order(self):
        """Encode produces tokens in text order."""
        tok = CharTokenizer()
        text = "reverse"
        ids = tok.encode(text)
        assert ids == [ord(c) for c in text]


class TestTokenizerFactory:
    """TokenizerFactory registration and creation."""

    def test_create_default(self):
        """create() without args creates char tokenizer."""
        tok = TokenizerFactory.create()
        assert isinstance(tok, CharTokenizer)

    def test_create_char(self):
        """create('char') creates a CharTokenizer."""
        tok = TokenizerFactory.create("char")
        assert isinstance(tok, CharTokenizer)

    def test_create_unknown_raises(self):
        """create() with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            TokenizerFactory.create("nonexistent")

    def test_list_available(self):
        """list_available() includes 'char'."""
        available = TokenizerFactory.list_available()
        assert "char" in available
        assert isinstance(available, list)
        assert available == sorted(available)  # sorted output

    def test_create_returns_new_instance(self):
        """Each create() call returns a fresh instance."""
        tok1 = TokenizerFactory.create("char")
        tok2 = TokenizerFactory.create("char")
        assert tok1 is not tok2
        assert type(tok1) is type(tok2)

    def test_register_custom(self):
        """register() adds a custom tokenizer to the factory."""
        from src.tokenizer.tokenizer import Tokenizer

        class DummyTokenizer(Tokenizer):
            def encode(self, text: str) -> list[int]:
                return []

            def decode(self, tokens: list[int]) -> str:
                return ""

            def count_tokens(self, text: str) -> int:
                return 0

        TokenizerFactory.register("dummy", DummyTokenizer)
        tok = TokenizerFactory.create("dummy")
        assert isinstance(tok, DummyTokenizer)
        assert "dummy" in TokenizerFactory.list_available()

    def test_register_duplicate_raises(self):
        """register() with already-registered name raises ValueError."""
        from src.tokenizer.tokenizer import Tokenizer

        class DummyTokenizer(Tokenizer):
            def encode(self, text: str) -> list[int]:
                return []

            def decode(self, tokens: list[int]) -> str:
                return ""

            def count_tokens(self, text: str) -> int:
                return 0

        TokenizerFactory.register("test_dup", DummyTokenizer)
        with pytest.raises(ValueError, match="already registered"):
            TokenizerFactory.register("test_dup", DummyTokenizer)
