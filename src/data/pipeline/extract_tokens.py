#!/usr/bin/env python3
"""Extract a subset of tokenized data for fast training iteration.

Works with .npy token arrays cached on slow storage, extracts subsets to fast storage.

Usage:
    python -m src.data.pipeline.extract_tokens \\
        --input /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
        --output data/fast/wikitext_tokens_1m.npy \\
        --size 1M
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()


def parse_size(size_str: str) -> int:
    """Parse size string like '1M', '100K' into token count.

    For consistency with extract_subset, also accepts byte sizes but converts to tokens.
    """
    size_str = size_str.upper().strip()

    # Handle token counts (T suffix or plain number)
    if size_str.endswith("T"):
        num = float(size_str[:-1])
        return int(num)

    # Handle byte/token multipliers
    multipliers = {
        "K": 1_000,
        "M": 1_000_000,
        "G": 1_000_000_000,
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            num = float(size_str[:-1])
            return int(num * multiplier)

    # No suffix, assume token count
    return int(size_str)


def extract_token_subset(
    input_path: Path,
    output_path: Path,
    size_tokens: int,
    verbose: bool = False,
) -> None:
    """Extract a subset of tokens from cached .npy array.

    Args:
        input_path: Source .npy token array (on slow storage)
        output_path: Destination .npy array (on fast storage)
        size_tokens: Number of tokens to extract
        verbose: Print progress
    """
    console.print("[bold blue]Extracting token subset[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Target: {size_tokens:,} tokens")

    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        return

    # Load token array
    if verbose:
        console.print("\n[bold]Loading token array...[/bold]")

    try:
        tokens = np.load(input_path)
    except Exception as e:
        console.print(f"[bold red]Error loading tokens:[/bold red] {e}")
        return

    if verbose:
        console.print(f"[green]✓[/green] Loaded {len(tokens):,} tokens")

    # Check if we have enough tokens
    if len(tokens) < size_tokens:
        console.print(
            f"[yellow]⚠[/yellow] Requested {size_tokens:,} tokens but only "
            f"{len(tokens):,} available. Extracting all."
        )
        size_tokens = len(tokens)

    # Extract subset
    subset = tokens[:size_tokens]

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    if verbose:
        console.print("\n[bold]Saving subset...[/bold]")

    np.save(output_path, subset)

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓[/green] Extracted {len(subset):,} tokens")
    console.print(f"[green]✓[/green] Saved to: {output_path}")

    # Statistics
    console.print("\n[bold]Subset Statistics:[/bold]")
    console.print(f"  Tokens: {len(subset):,}")
    console.print(f"  Array dtype: {subset.dtype}")
    console.print(f"  Array shape: {subset.shape}")
    console.print(f"  File size: {output_size_mb:.1f} MB")
    console.print(f"  Percentage of source: {len(subset) * 100 / len(tokens):.2f}%")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extract token subset from cached .npy array",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Size formats:
  1M    = 1 million tokens
  100K  = 100 thousand tokens
  500   = 500 tokens (plain number)

Workflow:
  This tool assumes tokens are already cached on slow storage.
  It simply extracts the first N tokens and copies to fast storage.

  For document-aware extraction, use extract_subset on text first,
  then tokenize that subset directly.

Examples:
  # Extract 1M tokens for training
  python -m src.data.pipeline.extract_tokens \\
      --input /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
      --output data/fast/wikitext_1m_tokens.npy \\
      --size 1M

  # Extract 100K tokens for quick overfit test
  python -m src.data.pipeline.extract_tokens \\
      --input /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
      --output data/fast/wikitext_100k_tokens.npy \\
      --size 100K
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input .npy token array (on slow storage)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .npy token array (on fast storage)",
    )
    parser.add_argument(
        "--size",
        required=True,
        help="Number of tokens to extract (e.g., 1M, 100K, 500)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress during extraction",
    )

    args = parser.parse_args()

    size_tokens = parse_size(args.size)
    extract_token_subset(
        input_path=args.input,
        output_path=args.output,
        size_tokens=size_tokens,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
