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

        # Remove spaces around = symbols: = = → ==
        for _ in range(10):
            if " =" not in stripped and "= " not in stripped[1:-1]:
                break
            stripped = (
                stripped[:1] + stripped[1:].replace(" =", "=").replace("= ", "=") + stripped[-1:]
            )

        # Extract opening =s, content, closing =s
        i = 0
        while i < len(stripped) and stripped[i] == "=":
            i += 1
        j = len(stripped) - 1
        while j >= 0 and stripped[j] == "=":
            j -= 1

        if i <= j:
            left_eqs = stripped[:i]
            content = stripped[i : j + 1].strip()
            right_eqs = stripped[j + 1 :]
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
    # Fix contractions: Victoria 's → Victoria's, Didn ' t → Didn't
    text = re.sub(r"(\w)[ \t]+'[ \t]*([a-z])", r"\1'\2", text)

    # Fix decades: ' 90s → '90s
    text = re.sub(r"'[ \t]+(\d)", r"'\1", text)

    # Remove spaces inside quotes: " text " → "text", ' text ' → 'text'
    text = re.sub(r'"[ \t]+', '"', text)  # Remove space after opening "
    text = re.sub(r'[ \t]+"', '"', text)  # Remove space before closing "
    text = re.sub(r"'[ \t]+", "'", text)  # Remove space after opening '
    text = re.sub(r"[ \t]+'", "'", text)  # Remove space before closing '

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize WikiText-103 formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python normalize_wikitext.py input.txt output.txt
  python normalize_wikitext.py --in-place data.txt
        """,
    )
    parser.add_argument("input_file", help="Input text file to normalize")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output file (default: input_file with .normalized suffix)",
    )
    parser.add_argument("--in-place", action="store_true", help="Modify file in place")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show statistics")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Determine output path
    if args.in_place:
        output_path = input_path
    elif args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}.normalized")

    # Read and normalize
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        original = f.read()

    normalized = clean_wikitext(original)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(normalized)

    # Report statistics
    orig_size = len(original)
    norm_size = len(normalized)
    orig_nl = original.count("\n")
    norm_nl = normalized.count("\n")

    print(f"✓ Normalized to: {output_path}")
    print(f"  Original: {orig_size:,} bytes | {orig_nl:,} newlines")
    print(f"  Normalized: {norm_size:,} bytes | {norm_nl:,} newlines")
    print(
        f"  Reduction: {orig_size - norm_size:,} bytes ({100*(orig_size-norm_size)/orig_size:.1f}%)"
    )
    print(f"  Newlines removed: {orig_nl - norm_nl:,}")

    if args.verbose:
        print("\n=== Sample (first 30 lines) ===")
        for i, line in enumerate(normalized.split("\n")[:30]):
            if line.strip():
                print(f"  {i:3d}: {line[:75]}")
            else:
                print(f"  {i:3d}: [empty]")


if __name__ == "__main__":
    main()
