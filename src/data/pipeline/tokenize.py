#!/usr/bin/env python3
"""Tokenize a text dataset and save as numpy array for caching.

This is part of the recommended workflow:
1. Normalize text on slow disk (one-time, cached)
2. Tokenize to .npy on slow disk (one-time, cached)
3. Extract token subsets to fast disk (instant, reusable)

Usage:
    python -m src.data.pipeline.tokenize \\
        --input /mnt/d/dev/data/wikitext-103-raw/train_normalized.txt \\
        --output /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
        --tokenizer char
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

from src.tokenizer import TokenizerFactory

console = Console()


def tokenize_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str,
    tokenizer_mode: str = "codepoint",
    vocab_size: int = 128,
    chunk_size: int = 100_000,  # Process in chunks to show progress
) -> None:
    """Tokenize text dataset and save as numpy array.

    Args:
        input_path: Source text file (normalized)
        output_path: Destination .npy file
        tokenizer_name: Tokenizer to use (e.g., 'char')
        tokenizer_mode: Tokenizer mode (codepoint, utf8, utf16, utf32)
        vocab_size: Vocabulary size for tokenizer (default: 128)
        chunk_size: Characters to process per chunk for progress display
    """
    console.print("[bold blue]Tokenizing dataset[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Tokenizer: {tokenizer_name}")
    console.print(f"  Mode: {tokenizer_mode}")
    if tokenizer_mode == "codepoint":
        console.print(f"  Vocab size: {vocab_size}")

    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    try:
        if tokenizer_mode == "codepoint":
            tokenizer = TokenizerFactory.create(
                tokenizer_name,
                mode=tokenizer_mode,
                vocab_size=vocab_size,
            )
        else:
            tokenizer = TokenizerFactory.create(tokenizer_name, mode=tokenizer_mode)
    except Exception as e:
        console.print(f"[bold red]Error loading tokenizer:[/bold red] {e}")
        sys.exit(1)

    # Read entire text
    console.print("\n[bold]Reading text file...[/bold]")
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    file_size_mb = len(text.encode("utf-8")) / (1024 * 1024)
    console.print(f"[green]✓[/green] Loaded {len(text):,} characters ({file_size_mb:.1f} MB)")

    # Tokenize
    console.print("\n[bold]Tokenizing...[/bold]")
    try:
        tokens = tokenizer.encode(text)
    except Exception as e:
        console.print(f"[bold red]Error tokenizing:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[green]✓[/green] Tokenized into {len(tokens):,} tokens")

    # Convert to numpy array
    tokens_array = np.array(tokens, dtype=np.int32)

    # Save
    console.print("\n[bold]Saving token array...[/bold]")
    np.save(output_path, tokens_array)

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓[/green] Saved to: {output_path}")
    console.print(f"[green]✓[/green] Output size: {output_size_mb:.1f} MB")

    # Statistics
    console.print("\n[bold]Tokenization Statistics:[/bold]")
    console.print(f"  Characters: {len(text):,}")
    console.print(f"  Tokens: {len(tokens):,}")
    console.print(f"  Chars per token: {len(text) / len(tokens):.2f}")
    console.print(f"  Vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
    console.print(f"  Array dtype: {tokens_array.dtype}")
    console.print(f"  Array shape: {tokens_array.shape}")
    console.print(f"  Bytes per token: {tokens_array.itemsize}")
    console.print(f"  Total size: {output_size_mb:.1f} MB")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Tokenize text dataset and cache as numpy array",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Normalize text on slow storage (one-time)
  2. Tokenize to .npy on slow storage (one-time, this tool)
  3. Extract token subsets to fast storage (instant, reusable)

Examples:
  # Tokenize normalized WikiText with character tokenizer
  python -m src.data.pipeline.tokenize \\
      --input /mnt/d/dev/data/wikitext-103-raw/train_normalized.txt \\
      --output /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
      --tokenizer char

  # Tokenize with different tokenizer (future)
  python -m src.data.pipeline.tokenize \\
      --input /mnt/d/dev/data/wikitext-103-raw/train_normalized.txt \\
      --output /mnt/d/dev/data/wikitext-103-raw/train_tokens_bpe.npy \\
      --tokenizer gpt2
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input text file (normalized)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .npy file for cached tokens",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name (e.g., 'char', 'gpt2')",
    )
    parser.add_argument(
        "--mode",
        default="codepoint",
        choices=["codepoint", "utf8", "utf16", "utf32"],
        help="Tokenizer mode (default: codepoint)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128,
        help="Vocabulary size for codepoint mode (default: 128)",
    )

    args = parser.parse_args()

    tokenize_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        tokenizer_mode=args.mode,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
