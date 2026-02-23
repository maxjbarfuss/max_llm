"""Unit tests for the character-level tokenizer."""

from src.tokenizer import CharTokenizer


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
        assert tok.encode("a\u4e2db") == [97, 98]       # 'a中b' → 'ab'

    def test_non_ascii_text_partial_roundtrip(self):
        """ASCII portions of mixed text roundtrip; non-ASCII is stripped."""
        tok = CharTokenizer()
        ids = tok.encode("hello \xff world")
        assert tok.decode(ids) == "hello  world"  # \xff dropped, space preserved

    def test_encode_ids_in_valid_range(self):
        """All IDs produced by encode are in [0, 127]."""
        tok = CharTokenizer()
        ids = tok.encode("".join(chr(i) for i in range(128)))
        assert all(0 <= i <= 127 for i in ids)

    def test_decode_ignores_out_of_range_ids(self):
        """IDs outside [0, 127] are silently dropped during decode."""
        tok = CharTokenizer()
        assert tok.decode([65, 200, 66]) == "AB"  # 200 dropped
        assert tok.decode([-1, 65]) == "A"          # -1 dropped

    def test_encode_single_chars(self):
        """Single-character encoding matches ord() exactly."""
        tok = CharTokenizer()
        for i in range(128):
            assert tok.encode(chr(i)) == [i]
