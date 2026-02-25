"""Unit tests for document boundary detectors (Phase 2)."""

import pytest

from src.data.datasets.boundary import (
    BlankLineBoundary,
    NoBoundary,
    PatternBoundary,
    TinyStoriesBoundary,
    WikiTextBoundary,
    get_boundary_detector,
)


class TestNoBoundary:
    def test_never_returns_boundary(self):
        """NoBoundary.is_boundary always returns False."""
        det = NoBoundary()
        for line in ["", "hello", "<|endoftext|>", " = Title = ", "===", "\n"]:
            assert det.is_boundary(line) is False


class TestWikiTextBoundary:
    def test_top_level_article_header(self):
        """Single = on each side is a top-level article boundary."""
        det = WikiTextBoundary()
        assert det.is_boundary(" = Valkyria Chronicles III = ")
        assert det.is_boundary("= Title =")
        assert det.is_boundary(" = Another Article = \n")

    def test_section_header_not_boundary(self):
        """Double == or deeper section headers are not article boundaries."""
        det = WikiTextBoundary()
        assert not det.is_boundary(" == Section == ")
        assert not det.is_boundary(" === Subsection === ")

    def test_plain_text_not_boundary(self):
        """Regular text lines are not boundaries."""
        det = WikiTextBoundary()
        assert not det.is_boundary("This is a normal sentence.")
        assert not det.is_boundary("")
        assert not det.is_boundary("   ")


class TestTinyStoriesBoundary:
    def test_endoftext_token_is_boundary(self):
        """Exact <|endoftext|> token marks a story boundary."""
        det = TinyStoriesBoundary()
        assert det.is_boundary("<|endoftext|>")

    def test_endoftext_with_surrounding_whitespace(self):
        """<|endoftext|> with leading/trailing whitespace is still a boundary."""
        det = TinyStoriesBoundary()
        assert det.is_boundary("  <|endoftext|>  ")
        assert det.is_boundary("<|endoftext|>\n")

    def test_blank_line_is_boundary(self):
        """Blank/empty lines are boundaries (karpathy tinystories variant)."""
        det = TinyStoriesBoundary()
        assert det.is_boundary("")
        assert det.is_boundary("   ")
        assert det.is_boundary("\t")
        assert det.is_boundary("\n")

    def test_plain_text_not_boundary(self):
        """Normal story text is not a boundary."""
        det = TinyStoriesBoundary()
        assert not det.is_boundary("Once upon a time there was a bunny.")

    def test_partial_token_not_boundary(self):
        """Partial or embedded token strings are not boundaries."""
        det = TinyStoriesBoundary()
        assert not det.is_boundary("endoftext")
        assert not det.is_boundary("some text <|endoftext|> more text")


class TestBlankLineBoundary:
    def test_empty_string_is_boundary(self):
        """Empty string is a boundary."""
        det = BlankLineBoundary()
        assert det.is_boundary("")

    def test_whitespace_only_is_boundary(self):
        """Lines with only whitespace are boundaries."""
        det = BlankLineBoundary()
        assert det.is_boundary("   ")
        assert det.is_boundary("\t")
        assert det.is_boundary("\t  \t")
        assert det.is_boundary("\n")

    def test_text_with_whitespace_not_boundary(self):
        """Lines with any non-whitespace are not boundaries."""
        det = BlankLineBoundary()
        assert not det.is_boundary("a")
        assert not det.is_boundary(" text ")
        assert not det.is_boundary("\t x \t")


class TestPatternBoundary:
    def test_matching_line_is_boundary(self):
        """Line matching the pattern is a boundary."""
        det = PatternBoundary(r"^---$")
        assert det.is_boundary("---")

    def test_non_matching_line_is_not_boundary(self):
        """Line not matching the pattern is not a boundary."""
        det = PatternBoundary(r"^---$")
        assert not det.is_boundary("hello")
        assert not det.is_boundary("--- extra")

    def test_custom_regex_pattern(self):
        """PatternBoundary respects arbitrary regex patterns."""
        det = PatternBoundary(r"^\d{4}-\d{2}-\d{2}")
        assert det.is_boundary("2024-01-15 Some event")
        assert not det.is_boundary("not a date")


class TestGetBoundaryDetector:
    def test_no_args_returns_no_boundary(self):
        """Default (no args) returns NoBoundary."""
        assert isinstance(get_boundary_detector(), NoBoundary)

    def test_pattern_arg_returns_pattern_boundary(self):
        """Explicit pattern returns PatternBoundary."""
        det = get_boundary_detector(pattern=r"^---")
        assert isinstance(det, PatternBoundary)

    def test_wikitext_returns_wikitext_boundary(self):
        """dataset='wikitext' returns WikiTextBoundary."""
        assert isinstance(get_boundary_detector(dataset="wikitext"), WikiTextBoundary)

    def test_tinystories_returns_tinystories_boundary(self):
        """dataset='tinystories' returns TinyStoriesBoundary."""
        assert isinstance(get_boundary_detector(dataset="tinystories"), TinyStoriesBoundary)

    def test_blank_line_returns_blank_line_boundary(self):
        """dataset='blank_line' returns BlankLineBoundary."""
        assert isinstance(get_boundary_detector(dataset="blank_line"), BlankLineBoundary)

    def test_unknown_dataset_raises(self):
        """Unknown dataset name raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_boundary_detector(dataset="bogus_dataset")

    def test_pattern_takes_precedence_over_dataset(self):
        """When both pattern and dataset are given, pattern wins."""
        det = get_boundary_detector(dataset="wikitext", pattern=r"^---")
        assert isinstance(det, PatternBoundary)
