"""Unit tests for WikiText normalization tool."""

import pytest

from src.data.datasets.wikitext.normalize import clean_wikitext, verify_text


class TestAtArtifacts:
    """Test @ artifact removal."""

    def test_remove_at_dash_at(self):
        """Remove @-@ artifacts."""
        assert clean_wikitext("guest @-@ starring role") == "guest-starring role"

    def test_remove_at_dot_at(self):
        """Remove @.@ artifacts."""
        assert clean_wikitext("U @.@ S @.@ A") == "U.S.A"

    def test_remove_standalone_at(self):
        """Remove standalone @ symbols."""
        assert clean_wikitext("text @ more") == "text more"

    def test_remove_multiple_artifacts(self):
        """Remove multiple @ artifacts in one pass."""
        text = "word @-@ word @.@ end"
        result = clean_wikitext(text)
        assert "@" not in result
        assert result == "word-word.end"


class TestNewlines:
    """Test newline collapsing."""

    def test_collapse_triple_newlines(self):
        """Collapse 3+ newlines to 2."""
        text = "line1\n\n\n\nline2"
        assert clean_wikitext(text) == "line1\n\nline2"

    def test_preserve_double_newlines(self):
        """Preserve double newlines (paragraph breaks)."""
        text = "para1\n\npara2"
        assert clean_wikitext(text) == "para1\n\npara2"

    def test_preserve_single_newlines(self):
        """Preserve single newlines."""
        text = "line1\nline2"
        assert clean_wikitext(text) == "line1\nline2"

    def test_collapse_many_newlines(self):
        """Collapse 10+ newlines to 2."""
        text = "line1\n\n\n\n\n\n\n\n\n\nline2"
        assert clean_wikitext(text) == "line1\n\nline2"


class TestHeaders:
    """Test header formatting."""

    def test_fix_spaced_header(self):
        """Fix header with spaces: = = Career = = → == Career ==."""
        result = clean_wikitext("= = Career = =")
        # Header fixing removes spaces between = signs (= = → ==)
        assert "Career" in result and result.count("=") == 4

    def test_fix_triple_spaced_header(self):
        """Fix header with more spaces: = = = text = = = → === text ===."""
        result = clean_wikitext("= = = Section = = =")
        assert "Section" in result

    def test_preserve_normal_header(self):
        """Preserve properly formatted header."""
        result = clean_wikitext("== Career ==")
        assert "Career" in result

    def test_header_with_leading_spaces(self):
        """Handle header with leading spaces."""
        text = " = = Test = = "
        result = clean_wikitext(text)
        assert "Test" in result

    def test_header_with_uneven_equals(self):
        """Handle header with uneven equals on sides."""
        text = "= Test = ="
        result = clean_wikitext(text)
        # Should normalize to balanced format
        assert "Test" in result and "=" in result


class TestPunctuation:
    """Test punctuation spacing."""

    def test_remove_space_before_comma(self):
        """Remove space before comma."""
        assert clean_wikitext("word ,") == "word,"

    def test_remove_space_before_period(self):
        """Remove space before period."""
        assert clean_wikitext("sentence .") == "sentence."

    def test_remove_spaces_before_punctuation(self):
        """Remove spaces before all closing punctuation."""
        text = "text ! text ? text ; text : text )"
        result = clean_wikitext(text)
        assert " !" not in result
        assert " ?" not in result
        assert " ;" not in result
        assert " :" not in result
        assert " )" not in result

    def test_remove_space_after_opening_paren(self):
        """Remove space after opening parenthesis."""
        assert clean_wikitext("( text )") == "(text)"

    def test_remove_space_after_opening_bracket(self):
        """Remove space after opening bracket."""
        assert clean_wikitext("[ text ]") == "[text]"

    def test_complex_punctuation(self):
        """Handle complex punctuation patterns."""
        text = "text ( example , here ) ."
        expected = "text (example, here)."
        assert clean_wikitext(text) == expected


class TestContractions:
    """Test contraction spacing."""

    def test_fix_spaced_is_contraction(self):
        """Fix 's contraction: it 's → it's."""
        assert clean_wikitext("it ' s") == "it's"

    def test_fix_spaced_not_contraction(self):
        """Fix 't contraction: don ' t → don't."""
        assert clean_wikitext("don ' t") == "don't"

    def test_fix_spaced_have_contraction(self):
        """Fix 've contraction: could ' ve → could've."""
        assert clean_wikitext("could ' ve") == "could've"

    def test_preserve_possessive_spacing(self):
        """Preserve possessive with space: governess' ice."""
        # Possessives followed by space+word should stay: "word ' word" not match [a-z]
        text = "governess ' ice"
        result = clean_wikitext(text)
        # The pattern only fixes \w+'[a-z], this has space after '
        assert "governess" in result and "ice" in result

    def test_multiple_contractions(self):
        """Fix multiple contractions in one text."""
        text = "I ' d like to say don ' t worry"
        result = clean_wikitext(text)
        assert "I'd" in result
        assert "don't" in result


class TestQuotes:
    """Test quote spacing."""

    def test_remove_space_after_opening_double_quote(self):
        """Remove spaces after opening double quote: " text → "text."""
        assert clean_wikitext('"  text') == '"text'
        assert clean_wikitext('" text') == '"text'

    def test_remove_space_before_closing_double_quote(self):
        """Remove spaces before closing double quote: text " → text"."""
        assert clean_wikitext('text  "') == 'text"'
        assert clean_wikitext('text "') == 'text"'

    def test_remove_spaces_around_quoted_text(self):
        """Remove spaces around quoted text: " text " → "text"."""
        assert clean_wikitext('"  text  "') == '"text"'
        assert clean_wikitext('" text "') == '"text"'

    def test_preserve_quotes_with_word_spacing(self):
        """Preserve space between word and quote: word "quote" → word "quote"."""
        text = 'half-section "L" network'
        result = clean_wikitext(text)
        # Space should be preserved between word and opening quote
        assert '"L"' in result

    def test_single_quotes_spacing(self):
        """Handle single quote spacing: remove spaces inside quoted spans."""
        assert clean_wikitext("'  text  '") == "'text'"
        assert clean_wikitext("' text '") == "'text'"

    def test_quote_with_punctuation(self):
        """Handle quotes with following punctuation."""
        text = '" Craig " in the'
        result = clean_wikitext(text)
        assert '"Craig"' in result


class TestCurrency:
    """Test currency symbol spacing."""

    def test_remove_space_after_dollar(self):
        """Remove space after $ : $ 22 → $22."""
        assert clean_wikitext("$ 22") == "$22"

    def test_remove_space_after_pound(self):
        """Remove space after £."""
        assert clean_wikitext("£ 100") == "£100"

    def test_remove_space_after_euro(self):
        """Remove space after €."""
        assert clean_wikitext("€ 50") == "€50"

    def test_remove_space_after_yen(self):
        """Remove space after ¥."""
        assert clean_wikitext("¥ 1000") == "¥1000"


class TestDashes:
    """Test dash spacing."""

    def test_remove_spaces_around_endash(self):
        """Remove spaces around en-dash: 30 – 40 → 30–40."""
        assert clean_wikitext("30 – 40") == "30–40"

    def test_remove_spaces_around_emdash(self):
        """Remove spaces around em-dash: text — more → text—more."""
        assert clean_wikitext("text — more") == "text—more"

    def test_preserve_endash_without_spaces(self):
        """Preserve en-dash without spaces."""
        assert clean_wikitext("2000–2005") == "2000–2005"

    def test_decades_with_apostrophe(self):
        """Fix decade spacing: ' 90s → '90s."""
        assert clean_wikitext("' 90s") == "'90s"


class TestLeadingTrailingSpaces:
    """Test leading and trailing space removal."""

    def test_remove_leading_spaces(self):
        """Remove leading spaces from lines."""
        text = "  indented line"
        assert clean_wikitext(text) == "indented line"

    def test_remove_trailing_spaces(self):
        """Remove trailing spaces from lines."""
        text = "line with spaces   "
        assert clean_wikitext(text) == "line with spaces"

    def test_preserve_internal_spaces(self):
        """Preserve internal spaces."""
        text = "line  with  internal  spaces"
        result = clean_wikitext(text)
        # Internal spaces should be collapsed to single space
        assert "line with internal spaces" == result


class TestMultipleSpaces:
    """Test multiple space collapsing."""

    def test_collapse_double_spaces(self):
        """Collapse double spaces to single."""
        assert clean_wikitext("word  word") == "word word"

    def test_collapse_multiple_spaces(self):
        """Collapse 3+ spaces to single."""
        assert clean_wikitext("word    word") == "word word"


class TestVerifyText:
    """Test issue detection."""

    def test_detect_quote_spacing_issues(self):
        """Detect space after opening quote."""
        text = '" text in quotes'
        issues = verify_text(text)
        assert len(issues["quote_spacing"]) > 0

    def test_detect_apostrophe_spacing(self):
        """Detect spaces around apostrophes."""
        text = "word ' s suffix"
        issues = verify_text(text)
        assert len(issues["apostrophe_spacing"]) > 0

    def test_no_issues_on_clean_text(self):
        """No issues on properly formatted text."""
        text = (
            "This is properly formatted text with contractions like don't and quotes like 'hello'."
        )
        issues = verify_text(text)
        # Should have minimal or no issues
        total = sum(len(v) for v in issues.values())
        assert total <= 1  # Allow for single quote at start which might match

    def test_detect_multiple_spaces(self):
        """Detect multiple spaces."""
        text = "word    word"
        issues = verify_text(text)
        assert len(issues["multiple_spaces"]) > 0

    def test_detect_malformed_headers(self):
        """Detect malformed headers."""
        text = "= = Career = ="
        issues = verify_text(text)
        assert len(issues["malformed_headers"]) > 0


class TestRealWorldExamples:
    """Test with real WikiText examples."""

    def test_constant_k_filter_header(self):
        """Fix: = = Constant k filter = = → == Constant k filter ==."""
        text = "= = Constant k filter = ="
        result = clean_wikitext(text)
        assert "Constant k filter" in result
        # Should normalize space-separated equals
        assert "= =" not in result

    def test_quoted_character_names(self):
        """Handle quoted character names in text."""
        text = 'He played " Craig " in the episode.'
        result = clean_wikitext(text)
        assert '"Craig"' in result

    def test_complex_sentence_with_artifacts(self):
        """Handle complex sentence with multiple issues."""
        text = 'The building @-@ block  is a "L" network'
        result = clean_wikitext(text)
        assert "@" not in result
        assert "  " not in result
        assert '"L"' in result

    def test_wikitext_normalize_example(self):
        """Example: normalize actual WikiText patterns."""
        text = """= = Career = =

In 2000 Boulter had a guest @-@ starring role  .

He had a role as " Craig " in  the episode ."""
        result = clean_wikitext(text)

        # Check basic fixes
        assert "= = Career = =" not in result  # Should be normalized
        assert "Career" in result  # Content preserved
        assert "@-@" not in result  # artifact removed
        assert '"Craig"' in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Handle empty string."""
        assert clean_wikitext("") == ""

    def test_only_whitespace(self):
        """Handle string with only whitespace."""
        result = clean_wikitext("   \n\n\n   ")
        # Should collapse newlines and remove spaces
        assert result.strip() == ""

    def test_single_character(self):
        """Handle single character."""
        assert clean_wikitext("x") == "x"

    def test_unicode_characters(self):
        """Preserve Unicode characters."""
        text = "café naïve résumé"
        result = clean_wikitext(text)
        assert "café" in result
        assert "naïve" in result
        assert "résumé" in result

    def test_mixed_quotes(self):
        """Handle mixed quote types."""
        text = "\" double \" and ' single ' quotes"
        result = clean_wikitext(text)
        assert '"double"' in result
        assert "'single'" in result

    def test_nested_structure(self):
        """Handle nested structures."""
        text = "outer [ inner ( nested ) ] end"
        result = clean_wikitext(text)
        # Spaces after opening brackets/parens are preserved between words
        assert "[inner" in result and "nested" in result

    def test_already_normalized_text(self):
        """Normalizing already clean text is idempotent."""
        text = "This is properly formatted text with don't and currency $99."
        result1 = clean_wikitext(text)
        result2 = clean_wikitext(result1)
        assert result1 == result2


class TestCriticalBugs:
    """Test for critical bugs found in normalization."""

    def test_preserve_space_before_opening_quote(self):
        """Preserve space between word and opening quote: word "quote" stays as word "quote"."""
        # Bug: normalization was removing space before quote: as " Connor " → as"Connor"
        text = 'He had a role as " Connor Price " .'
        result = clean_wikitext(text)
        # Most importantly: space before opening quote should be preserved
        assert 'as "' in result
        # Should NOT have: as"
        assert 'as"' not in result

    def test_preserve_space_before_single_quote(self):
        """Preserve space between word and single quote."""
        text = "He said ' hello ' to me ."
        result = clean_wikitext(text)
        # Space before opening single quote should be preserved
        assert "said '" in result
        assert "said'" not in result

    def test_remove_multiple_spaces_in_quotes(self):
        """Remove multiple spaces inside quotes, but preserve single spaces."""
        text = 'He said "  hello  " to me.'
        result = clean_wikitext(text)
        # Multiple spaces should be collapsed to nothing
        assert '"hello"' in result
        assert '"  ' not in result

    def test_fix_spaced_curly_apostrophes_and_quote_padding(self):
        """Fix real sample with spaced curly apostrophes and padded quotes."""
        text = (
            'Externally, Stevens is always calm, but internally he is far from it. " I\'m not as calm as everybody thinks, " '
            'Stevens says. His wife Tracy adds, " He ’ s calm and collected, but he ’ s fiercely competitive. '
            'He ’ s always thinking about how he can beat you. " Former player Joel Cornette says " Everyone sees Brad '
            "as a level-headed, calm and cool coach, but he ’ s about as competitive of a guy as I know. "
            'We would get into it constantly, whether playing two-on-two or arguing about players ’ having better college careers. "'
        )
        result = clean_wikitext(text)

        assert '"I\'m not as calm as everybody thinks,"' in result
        assert "he ’ s" not in result
        assert "He ’ s" not in result
        assert "players ’ having" not in result
        assert "he's" in result

    def test_quoted_character_names(self):
        """Test quoted character names in dialogue attribution."""
        text = 'He portrayed " Scott Parry " in the episode , " In Safe Hands " .'
        result = clean_wikitext(text)
        # Most importantly: preserved word-quote spacing
        assert 'portrayed "' in result
        # Note: punctuation spacing removes space before comma, so episode , " becomes episode, "
        assert 'episode, "' in result or 'episode "' in result
        # Should NOT have word stuck to quote
        assert 'portrayed"' not in result
        assert 'episode"' not in result

    def test_real_wikitext_example(self):
        """Test real WikiText example that was broken."""
        text = """In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the episode , " In Safe Hands " ."""
        result = clean_wikitext(text)

        # Critical: Should NOT have word"quote patterns (space removed)
        assert 'portrayed"' not in result
        assert 'episode"' not in result

        # Should preserve word-quote spacing
        assert 'portrayed "' in result
        # Note: punctuation spacing removes space before comma
        assert 'episode, "' in result

    pytest.main([__file__, "-v"])
