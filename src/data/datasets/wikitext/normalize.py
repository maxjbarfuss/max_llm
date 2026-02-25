"""Normalize WikiText-103 formatting.

Fixes:
- @ artifacts (@ @-@ @.@)
- Spaces before/after punctuation
- Quote spacing
- Apostrophe handling
- Currency spacing
- En-dash/em-dash spacing
- Leading paragraph spaces
- Extra newlines (3+ → 2)
- Header formatting (= = → ==)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rich.console import Console

console = Console()


def clean_wikitext(text: str) -> str:
    """Normalize WikiText formatting while preserving document structure."""

    # ========== REMOVE @ ARTIFACTS ==========
    text = text.replace(" @-@ ", "-").replace("@-@", "-")
    text = text.replace(" @.@ ", ".").replace("@.@", ".").replace("@", "")

    # ========== COLLAPSE EXTRA NEWLINES ==========
    # 3+ newlines → 2 newlines (preserves paragraph breaks)
    text = re.sub(r"\n\n\n+", "\n\n", text)

    # ========== FIX HEADER FORMATTING ==========
    # Convert "= = Career = =" to "== Career =="
    def fix_header_line(line: str) -> str:
        stripped = line.strip()
        if not stripped or stripped[0] != "=" or stripped[-1] != "=":
            return line

        # First, collapse space-separated equals: = = → ==
        # Replace " =" with "=" and "= " with "=" (but keep one leading =)
        while " = " in stripped:
            stripped = stripped.replace(" = ", "=")
        stripped = stripped.replace(" =", "=").replace("= ", "=")

        # Now count leading and trailing equals
        i = 0
        while i < len(stripped) and stripped[i] == "=":
            i += 1

        j = len(stripped) - 1
        while j >= 0 and stripped[j] == "=":
            j -= 1

        if i <= j:
            # Extract parts
            left_count = i
            content = stripped[i : j + 1].strip()

            # Use left side count for both sides (standard format)
            level = left_count
            left_eqs = "=" * level
            right_eqs = "=" * level

            return f"{left_eqs} {content} {right_eqs}"
        return line

    lines = [fix_header_line(line) for line in text.split("\n")]
    text = "\n".join(lines)

    # ========== FIX PUNCTUATION SPACING ==========
    # Spaces before closing punctuation: , . ! ? ; : ) ] }
    text = re.sub(r"[ \t]+([,\.!?;:\)\]\}])", r"\1", text)

    # Spaces after opening punctuation: ( [ {
    text = re.sub(r"([\(\[\{])[ \t]+", r"\1", text)

    # ========== FIX QUOTE & APOSTROPHE SPACING ==========
    # Fix contractions with straight or curly apostrophes:
    # Victoria ' s → Victoria's, he ’ s → he's, could ' ve → could've
    # Limit to short suffixes (1-3 chars) to avoid matching dialogue quotes like "said ' hello'"
    text = re.sub(r"(\w)[ \t]+[’'][ \t]*([a-z]{1,3})\b", r"\1'\2", text, flags=re.IGNORECASE)

    # Fix possessive marker separated by spaces before a following word:
    # players ’ having → players' having
    text = re.sub(r"(\b\w+s)[ \t]+[’'][ \t]+([a-z]\w*)", r"\1' \2", text, flags=re.IGNORECASE)

    # Fix decades: ' 90s → '90s
    text = re.sub(r"'[ \t]+(\d)", r"'\1", text)

    # Trim padding inside paired double quotes only, preserving external spacing.
    text = re.sub(r'"([^"\n]*)"', lambda m: f'"{m.group(1).strip()}"', text)

    # Handle unmatched edge cases used in tests: opening quote with inner leading spaces,
    # and closing quote at end/punctuation with inner trailing spaces.
    text = re.sub(r'(^|[\s\(\[\{])"[ \t]+', r'\1"', text)
    text = re.sub(r'(\S)[ \t]+"(?=$|[ \t]*[,\.!?;:\)\]\}])', r'\1"', text)

    # Trim padding inside paired single quotes only when they are quote delimiters,
    # not apostrophes inside words.
    text = re.sub(r"(?<!\w)'([^'\n]*)'(?!\w)", lambda m: f"'{m.group(1).strip()}'", text)

    # ========== FIX CURRENCY SPACING ==========
    # Remove space after currency symbols: $ 22 → $22
    text = re.sub(r"([$£€¥])\s+", r"\1", text)

    # ========== FIX DASH SPACING ==========
    # En-dash/em-dash spacing: 30 – 40 → 30–40 (remove spaces around dashes)
    text = re.sub(r"[ \t]+(–|—)[ \t]+", r"\1", text)

    # ========== FIX MULTIPLE SPACES ==========
    # Collapse multiple spaces/tabs (but NOT newlines) to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # ========== FIX LEADING PARAGRAPH SPACES ==========
    # Remove leading spaces at start of lines (but keep content)
    text = re.sub(r"(?m)^[ \t]+", "", text)

    # ========== FIX TRAILING SPACES ==========
    # Remove trailing spaces from lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    return text


def verify_text(text: str, context_lines: int = 50) -> dict:
    """Scan text and identify potential formatting issues.

    Returns dict with counts and sample lines for each issue type.
    """
    lines = text.split("\n")
    issues: dict[str, list[tuple[int, str]]] = {
        "quote_spacing": [],
        "apostrophe_spacing": [],
        "word_quote_missing_space": [],
        "multiple_spaces": [],
        "malformed_headers": [],
    }

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        # Quote spacing: space after opening quote or before closing quote
        if re.search(r'"[ \t]+\w', line) or re.search(r'\w[ \t]+"', line):
            issues["quote_spacing"].append((i + 1, line[:context_lines]))

        # Apostrophe with spaces around it
        if re.search(r"(\w)[ \t]+'[ \t]+", line):
            issues["apostrophe_spacing"].append((i + 1, line[:context_lines]))

        # Missing space after quote before word: word"word or "word"word
        if re.search(r'(\w)"[a-z]|"[a-z]"[a-z]', line):
            issues["word_quote_missing_space"].append((i + 1, line[:context_lines]))

        # Double spaces (not inside quotes)
        if re.search(r'(?<!")  +(?!")', line):
            issues["multiple_spaces"].append((i + 1, line[:context_lines]))

        # Malformed headers: = = Career = = or = = text
        if re.search(r"^[ \t]*=[ \t]+=", line) or re.search(r"=[ \t]+[a-z].*=[ \t]=", line):
            issues["malformed_headers"].append((i + 1, line[:context_lines]))

    return issues


def display_issues(issues: dict, title: str = "Issues Found") -> None:
    """Display issues in a formatted table."""
    total = sum(len(v) for v in issues.values())

    if total == 0:
        console.print(f"[green]✓ {title}: No issues found![/green]")
        return

    console.print(f"\n[red bold]{title}: {total} issues\n[/red bold]")

    for issue_type, findings in issues.items():
        if not findings:
            continue

        console.print(
            f"[yellow]{issue_type.replace('_', ' ').title()}:[/yellow] {len(findings)} found"
        )

        # Show first 3 examples
        for line_num, content in findings[:3]:
            # Highlight the line with context
            preview = content[:70].replace('"', '[red]"[/red]').replace("'", "[red]'[/red]")
            console.print(f"  Line {line_num}: {preview}")
            if len(content) > 70:
                console.print("    ...")

        if len(findings) > 3:
            console.print(f"  ... and {len(findings) - 3} more")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize WikiText-103 formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python normalize_wikitext.py input.txt output.txt
  python normalize_wikitext.py --verify input.txt
  python normalize_wikitext.py --in-place data.txt -v
        """,
    )
    parser.add_argument("input_file", help="Input text file to normalize")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output file (default: input_file with .normalized suffix)",
    )
    parser.add_argument("--in-place", action="store_true", help="Modify file in place")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify original file for issues only (no normalization)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show statistics and samples")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        return

    # Read file
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        original = f.read()

    # Verify mode: just scan and report
    if args.verify:
        console.print(f"\n[cyan bold]Scanning: {input_path.name}[/cyan bold]")
        issues_before = verify_text(original)
        display_issues(issues_before, "Issues Found in Original")

        # Also show what normalization would fix
        normalized = clean_wikitext(original)
        issues_after = verify_text(normalized)
        display_issues(issues_after, "Issues After Normalization")

        # Summary
        fixed_count = sum(len(v) for v in issues_before.values()) - sum(
            len(v) for v in issues_after.values()
        )
        console.print(f"\n[green]Would fix {fixed_count} issues[/green]")
        return

    # Normal mode: normalize
    normalized = clean_wikitext(original)

    # Determine output path
    if args.in_place:
        output_path = input_path
    elif args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}.normalized")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(normalized)

    # Report statistics
    orig_size = len(original)
    norm_size = len(normalized)
    orig_nl = original.count("\n")
    norm_nl = normalized.count("\n")

    console.print(f"\n[green]✓ Normalized to:[/green] {output_path}")
    console.print(f"  Original: {orig_size:,} bytes | {orig_nl:,} newlines")
    console.print(f"  Normalized: {norm_size:,} bytes | {norm_nl:,} newlines")
    pct = 100 * (orig_size - norm_size) / orig_size
    console.print(f"  Reduction: {orig_size - norm_size:,} bytes ([yellow]{pct:.1f}%[/yellow])")
    console.print(f"  Newlines removed: {orig_nl - norm_nl:,}")

    if args.verbose:
        console.print("\n[cyan]=== Remaining Issues ===[/cyan]")
        issues = verify_text(normalized)
        display_issues(issues, "Potential Issues in Normalized Output")

        console.print("\n[cyan]=== Sample (first 30 lines) ===[/cyan]")
        for i, line in enumerate(normalized.split("\n")[:30]):
            if line.strip():
                console.print(f"  {i:3d}: {line[:75]}")
            else:
                console.print(f"  {i:3d}: [dim][empty][/dim]")


if __name__ == "__main__":
    main()
