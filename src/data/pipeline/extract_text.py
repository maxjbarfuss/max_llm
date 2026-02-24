#!/usr/bin/env python3
"""Extract a subset of a text dataset for quick training iteration.

Respects document boundaries to avoid cutting mid-document. The boundary
strategy is dataset-specific (WikiText article headers, TinyStories end-of-story
tokens, etc.) and is resolved by the BoundaryDetector factory.

Usage:
    # Extract complete WikiText articles
    python -m src.data.pipeline.extract_text \\
        --input /slow/wikitext/train_normalized.txt \\
        --output data/fast/wikitext_10mb.txt \\
        --size 10M --dataset wikitext

    # Extract complete TinyStories stories
    python -m src.data.pipeline.extract_text \\
        --input /slow/tinystories/train.txt \\
        --output data/fast/tinystories_10mb.txt \\
        --size 10M --dataset tinystories

    # Custom boundary pattern (regex)
    python -m src.data.pipeline.extract_text \\
        --input data.txt --output subset.txt --size 5M \\
        --boundary-pattern "^<\\|endoftext\\|>"

    # No boundary detection (cut at byte limit)
    python -m src.data.pipeline.extract_text \\
        --input data.txt --output subset.txt --size 10M
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rich.console import Console

from src.data.datasets.boundary import NoBoundary, get_boundary_detector

console = Console()


def parse_size(size_str: str) -> int:
    """Parse size string like '10M', '100K', '1G' into bytes.

    For token-based sizes, assumes ~4 bytes per token on average.
    """
    size_str = size_str.upper().strip()

    # Handle token counts (assume 4 bytes/token)
    if size_str.endswith("T"):
        # Accept token units like 10T, 500KT, 1MT, 2.5GT
        token_spec = size_str[:-1]
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMG]?)", token_spec)
        if not match:
            raise ValueError(f"Invalid token size: {size_str}")

        value = float(match.group(1))
        multiplier = {
            "": 1,
            "K": 1_000,
            "M": 1_000_000,
            "G": 1_000_000_000,
        }[match.group(2)]

        return int(value * multiplier * 4)  # 4 bytes per token estimate

    # Handle byte sizes
    multipliers = {
        "K": 1024,
        "M": 1024 * 1024,
        "G": 1024 * 1024 * 1024,
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            num = float(size_str[:-1])
            return int(num * multiplier)

    # No suffix, assume bytes
    return int(size_str)


def extract_subset(
    input_path: Path,
    output_path: Path,
    size_bytes: int,
    detector: NoBoundary | object | None = None,
    verbose: bool = False,
) -> None:
    """Extract a subset of text data, optionally respecting document boundaries.

    Args:
        input_path: Source text file
        output_path: Destination for subset
        size_bytes: Target size in bytes (will exceed to complete last document)
        detector: BoundaryDetector instance. NoBoundary (default) cuts at byte limit.
        verbose: Print progress
    """
    if detector is None:
        detector = NoBoundary()

    boundary_mode = not isinstance(detector, NoBoundary)

    console.print(f"[bold blue]Extracting subset from:[/bold blue] {input_path}")
    console.print(
        f"[bold blue]Target size:[/bold blue] {size_bytes:,} bytes (~{size_bytes // 4:,} tokens)"
    )
    if boundary_mode:
        console.print("[bold blue]Mode:[/bold blue] Document-aware (complete documents only)")
    else:
        console.print("[bold blue]Mode:[/bold blue] Arbitrary chunks")

    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        return

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    lines_written = 0
    docs_written = 0
    current_doc_lines: list[str] = []
    current_doc_bytes = 0
    target_reached = False

    with open(input_path, encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line_bytes = len(line.encode("utf-8"))

                if boundary_mode:
                    # Check if this is a document boundary
                    if detector.is_boundary(line):
                        # Write previous document if it exists
                        if current_doc_lines:
                            # If we've reached target, don't start new doc
                            if target_reached:
                                break

                            for doc_line in current_doc_lines:
                                f_out.write(doc_line)

                            bytes_written += current_doc_bytes
                            lines_written += len(current_doc_lines)
                            docs_written += 1

                            if verbose and docs_written % 10 == 0:
                                console.print(
                                    f"  Written {docs_written} docs, "
                                    f"{lines_written:,} lines, {bytes_written:,} bytes..."
                                )

                            # Check if we've reached target after writing this doc
                            if bytes_written >= size_bytes:
                                target_reached = True

                            # Reset for new document
                            current_doc_lines = []
                            current_doc_bytes = 0

                        # Start new document
                        current_doc_lines.append(line)
                        current_doc_bytes += line_bytes
                    else:
                        # Continue current document
                        current_doc_lines.append(line)
                        current_doc_bytes += line_bytes
                else:
                    # Simple mode: just stop at byte limit
                    if bytes_written >= size_bytes:
                        break

                    f_out.write(line)
                    bytes_written += line_bytes
                    lines_written += 1

                    if verbose and lines_written % 1000 == 0:
                        console.print(f"  Read {lines_written:,} lines, {bytes_written:,} bytes...")

            # Write final document if in boundary mode
            if boundary_mode and current_doc_lines and not target_reached:
                for doc_line in current_doc_lines:
                    f_out.write(doc_line)
                bytes_written += current_doc_bytes
                lines_written += len(current_doc_lines)
                docs_written += 1

    console.print(f"[green]✓[/green] Extracted {lines_written:,} lines ({bytes_written:,} bytes)")
    if boundary_mode:
        console.print(f"[green]✓[/green] Complete documents: {docs_written}")
    console.print(f"[green]✓[/green] Saved to: {output_path}")

    # Show stats
    console.print("\n[bold]Subset Statistics:[/bold]")
    if boundary_mode:
        console.print(f"  Documents: {docs_written:,}")
    console.print(f"  Lines: {lines_written:,}")
    console.print(f"  Bytes: {bytes_written:,}")
    console.print(f"  Estimated tokens: ~{bytes_written // 4:,}")
    console.print(f"  File size: {output_path.stat().st_size:,} bytes")

    if bytes_written > size_bytes:
        overage = bytes_written - size_bytes
        console.print(
            f"  [yellow]Note:[/yellow] Exceeded target by {overage:,} bytes "
            f"({overage * 100 / size_bytes:.1f}%) to complete final document"
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extract subset of text dataset with optional document boundary awareness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Size formats:
  10M   = 10 megabytes
  100K  = 100 kilobytes
  1G    = 1 gigabyte
  10MT  = 10 million tokens (~40MB)

Boundary detection (mutually exclusive, first match wins):
  --dataset NAME        Use a registered detector (wikitext, tinystories)
  --boundary-pattern P  Use a custom regex pattern
  (neither)             No boundary detection; cut at byte limit

Examples:
  # WikiText: extract complete articles
  python -m src.data.pipeline.extract_text \\
      --input /slow/wikitext/train_normalized.txt \\
      --output data/fast/wikitext_10mb.txt \\
      --size 10M --dataset wikitext

  # TinyStories: extract complete stories
  python -m src.data.pipeline.extract_text \\
      --input /slow/tinystories/train.txt \\
      --output data/fast/tinystories_10mb.txt \\
      --size 10M --dataset tinystories

  # Custom boundary pattern
  python -m src.data.pipeline.extract_text \\
      --input data.txt --output subset.txt --size 5M \\
      --boundary-pattern "^<\\|endoftext\\|>"

  # No boundaries (simple byte-limit cut)
  python -m src.data.pipeline.extract_text \\
      --input data.txt --output subset.txt --size 10M
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input text file (on slow storage)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output subset file (on fast storage)",
    )
    parser.add_argument(
        "--size",
        required=True,
        help="Target size (e.g., 10M, 100K, 1MT for 1M tokens)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name for registered boundary detector (e.g. wikitext, tinystories)",
    )
    parser.add_argument(
        "--boundary-pattern",
        type=str,
        help="Custom regex pattern for document boundaries",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress during extraction",
    )

    args = parser.parse_args()

    size_bytes = parse_size(args.size)
    detector = get_boundary_detector(dataset=args.dataset, pattern=args.boundary_pattern)
    extract_subset(
        input_path=args.input,
        output_path=args.output,
        size_bytes=size_bytes,
        detector=detector,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
