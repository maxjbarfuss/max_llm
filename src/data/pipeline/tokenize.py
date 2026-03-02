#!/usr/bin/env python3
"""Tokenize a text dataset and save as numpy array for caching.

This is part of the recommended workflow:
1. Normalize text on slow disk (one-time, cached)
2. Tokenize to .npy on slow disk (one-time, cached)
3. Extract token subsets to fast disk (instant, reusable)

Usage:
    python -m src.data.pipeline.tokenize \\
        --input data/slow/<dataset>_normalized.txt \\
        --output data/slow/<dataset>_tokens.npy \\
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


def _print_tokenizer_config(
    tokenizer_name: str,
    tokenizer_mode: str,
    vocab_size: int,
    encoding: str,
    model_path: str | None,
) -> None:
    """Print tokenizer configuration to console."""
    console.print(f"  Tokenizer: {tokenizer_name}")
    if tokenizer_name == "bpe":
        console.print(f"  Encoding: {encoding}")
    elif tokenizer_name == "unigram":
        console.print(f"  Model: {model_path}")
    else:
        console.print(f"  Mode: {tokenizer_mode}")
        if tokenizer_mode == "codepoint":
            console.print(f"  Vocab size: {vocab_size}")


def _create_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str,
    vocab_size: int,
    encoding: str,
    model_path: str | None,
):
    """Create tokenizer based on configuration."""
    if tokenizer_name == "bpe":
        return TokenizerFactory.create(tokenizer_name, encoding=encoding)
    elif tokenizer_name == "unigram":
        if not model_path:
            raise ValueError("--model-path is required when --tokenizer unigram")
        return TokenizerFactory.create(tokenizer_name, model_path=model_path)
    elif tokenizer_mode == "codepoint":
        return TokenizerFactory.create(
            tokenizer_name,
            mode=tokenizer_mode,
            vocab_size=vocab_size,
        )
    else:
        return TokenizerFactory.create(tokenizer_name, mode=tokenizer_mode)


def tokenize_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str,
    tokenizer_mode: str = "codepoint",
    vocab_size: int = 128,
    encoding: str = "gpt2",
    model_path: str | None = None,
    chunk_size: int = 100_000,  # Process in chunks to show progress
) -> None:
    """Tokenize text dataset and save as numpy array.

    Args:
        input_path: Source text file (normalized)
        output_path: Destination .npy file
        tokenizer_name: Tokenizer to use (e.g., 'char', 'bpe')
        tokenizer_mode: Tokenizer mode for char tokenizer (codepoint, utf8, utf16, utf32)
        vocab_size: Vocabulary size for codepoint mode (default: 128)
        encoding: tiktoken encoding name for BPE tokenizer (default: 'gpt2')
        model_path: sentencepiece model path for Unigram tokenizer
        chunk_size: Characters to process per chunk for progress display
    """
    console.print("[bold blue]Tokenizing dataset[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    _print_tokenizer_config(tokenizer_name, tokenizer_mode, vocab_size, encoding, model_path)

    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    try:
        tokenizer = _create_tokenizer(
            tokenizer_name, tokenizer_mode, vocab_size, encoding, model_path
        )
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
  # Tokenize normalized text with character tokenizer
  python -m src.data.pipeline.tokenize \\
      --input data/slow/<dataset>_normalized.txt \\
      --output data/slow/<dataset>_tokens.npy \\
      --tokenizer char

  # Tokenize with BPE tokenizer
  python -m src.data.pipeline.tokenize \\
      --input data/slow/<dataset>_normalized.txt \\
      --output data/slow/<dataset>_tokens_bpe.npy \\
      --tokenizer bpe --encoding gpt2
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
    parser.add_argument(
        "--encoding",
        default="gpt2",
        help="tiktoken encoding for BPE tokenizer (default: gpt2)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="SentencePiece .model path for Unigram tokenizer",
    )

    args = parser.parse_args()

    tokenize_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        tokenizer_mode=args.mode,
        vocab_size=args.vocab_size,
        encoding=args.encoding,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
