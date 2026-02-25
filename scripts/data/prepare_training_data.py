#!/usr/bin/env python3
"""Prepare training data: normalize, tokenize, and extract subsets.

This is the recommended workflow for Phase 2-4 training:
1. Normalize large dataset on slow storage (one-time, cached)
2. Tokenize to .npy on slow storage (one-time, cached)
3. Extract token subset to fast storage for active training
4. Update config to point at the subset

Usage:
    # Full workflow: normalize + tokenize + extract 500K token subset
    python -m scripts/data/prepare_training_data.py \
        --source /mnt/d/dev/data/wikitext-103-raw/train.txt \\
        --output data/fast/wikitext_500k_tokens.npy \\
        --size 500K \\
        --normalize \\
        --tokenize \\
        --tokenizer char

    # Extract token subset from already-cached tokens
    python -m scripts/data/prepare_training_data.py \
        --source /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
        --output data/fast/wikitext_1m_tokens.npy \\
        --size 1M
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def prepare_training_data(
    source_path: Path,
    output_path: Path,
    size: str,
    normalize: bool = False,
    tokenize: bool = False,
    tokenizer: str = "char",
    tokenizer_mode: str = "codepoint",
    vocab_size: int = 128,
    force: bool = False,
) -> None:
    """Prepare training data: optionally normalize, tokenize, then extract subset.

    Args:
        source_path: Source file (text or .npy tokens) on slow storage
        output_path: Destination on fast storage
        size: Target size (e.g., '500K' tokens, '10M' bytes)
        normalize: If True, normalize before tokenizing
        tokenize: If True, tokenize to .npy (requires text input)
        tokenizer: Tokenizer name (default: 'char')
        tokenizer_mode: Tokenizer mode (codepoint, utf8, utf16, utf32)
        vocab_size: Tokenizer vocabulary size (default: 128)
        force: If True, overwrite existing files
    """
    console.print("[bold blue]Preparing training data[/bold blue]")
    console.print(f"  Source: {source_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Size: {size}")
    console.print(f"  Pipeline: normalize={normalize}, tokenize={tokenize}")
    if tokenize:
        console.print(f"  Tokenizer: {tokenizer} (mode={tokenizer_mode}, vocab_size={vocab_size})")

    if not source_path.exists():
        console.print(f"[bold red]Error:[/bold red] Source file not found: {source_path}")
        sys.exit(1)

    # Determine workflow based on source and options
    current_path = source_path
    is_tokenized = source_path.suffix == ".npy"

    # Step 1: Normalize if requested (text only)
    if normalize:
        if is_tokenized:
            console.print("[yellow]⚠[/yellow] Skipping normalization (source is already tokenized)")
        else:
            console.print("\n[bold]Step 1: Normalizing text[/bold]")

            # Check if normalized version already exists
            normalized_path = source_path.with_name(source_path.stem + "_normalized.txt")

            if normalized_path.exists() and not force:
                console.print(
                    f"[yellow]⚠[/yellow] Normalized file already exists: {normalized_path}"
                )
                console.print("  Using existing normalized file. Use --force to re-normalize.")
            else:
                # Run normalization script
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "src/data/datasets/wikitext/normalize.py",
                            str(current_path),
                            str(normalized_path),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    console.print("[green]✓[/green] Normalization complete")
                except subprocess.CalledProcessError as e:
                    console.print(f"[bold red]Error normalizing:[/bold red] {e}")
                    console.print(e.stderr)
                    sys.exit(1)

            current_path = normalized_path

    # Step 2: Tokenize if requested (text only)
    if tokenize:
        if is_tokenized or current_path.suffix == ".npy":
            console.print("[yellow]⚠[/yellow] Skipping tokenization (already tokenized)")
        else:
            console.print("\n[bold]Step 2: Tokenizing to .npy[/bold]")

            # Create tokenized version on slow storage (for caching)
            tokenized_path = current_path.with_name(current_path.stem + "_tokens.npy")

            if tokenized_path.exists() and not force:
                console.print(f"[yellow]⚠[/yellow] Tokenized file already exists: {tokenized_path}")
                console.print("  Using existing tokenized file. Use --force to re-tokenize.")
            else:
                # Run tokenization script
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "src.data.pipeline.tokenize",
                            "--input",
                            str(current_path),
                            "--output",
                            str(tokenized_path),
                            "--tokenizer",
                            tokenizer,
                            "--mode",
                            tokenizer_mode,
                            "--vocab-size",
                            str(vocab_size),
                        ],
                        check=True,
                        capture_output=False,  # Show progress
                    )
                    console.print("[green]✓[/green] Tokenization complete")
                except subprocess.CalledProcessError as e:
                    console.print(f"[bold red]Error tokenizing:[/bold red] {e}")
                    sys.exit(1)

            current_path = tokenized_path
            is_tokenized = True

    # Step 3: Extract subset to fast storage
    console.print("\n[bold]Step 3: Extracting subset to fast storage[/bold]")

    if output_path.exists() and not force:
        console.print(f"[yellow]⚠[/yellow] Output file already exists: {output_path}")
        console.print("  Use --force to overwrite.")
        sys.exit(1)

    # Choose extraction tool based on data type
    if is_tokenized or current_path.suffix == ".npy":
        # Extract token subset
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.data.pipeline.extract_tokens",
                    "--input",
                    str(current_path),
                    "--output",
                    str(output_path),
                    "--size",
                    size,
                    "-v",
                ],
                check=True,
                capture_output=False,  # Show progress
            )
            console.print("[green]✓[/green] Token subset extraction complete")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error extracting tokens:[/bold red] {e}")
            sys.exit(1)
    else:
        # Extract text subset (legacy)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.data.pipeline.extract_text",
                    "--input",
                    str(current_path),
                    "--output",
                    str(output_path),
                    "--size",
                    size,
                ],
                check=True,
                capture_output=False,  # Show progress
            )
            console.print("[green]✓[/green] Text subset extraction complete")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error extracting subset:[/bold red] {e}")
            sys.exit(1)

    # Step 4: Instructions for config update
    console.print("\n[bold green]✓ Data preparation complete![/bold green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Update config/experiment.toml:")
    console.print(f'     dataset_path = "{output_path}"')
    console.print("  2. Run training:")
    console.print("     python -m src.training.train --config config/experiment.toml")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Prepare training data with normalization, tokenization, and subset extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Optional: Normalize text on slow storage (cached for reuse)
  2. Optional: Tokenize to .npy on slow storage (cached for reuse)
  3. Extract subset to fast storage for training
    4. Update config and run training

Size formats:
  500K  = 500,000 tokens
  1M    = 1,000,000 tokens
  10M   = 10 megabytes (for text extraction)

Examples:
  # Full workflow: normalize + tokenize + extract 500K tokens
    python -m scripts/data/prepare_training_data.py \
      --source /mnt/d/dev/data/wikitext-103-raw/train.txt \\
      --output data/fast/wikitext_500k_tokens.npy \\
      --size 500K \\
      --normalize \\
      --tokenize \\
      --tokenizer char

  # Extract from already-cached tokens
    python -m scripts/data/prepare_training_data.py \
      --source /mnt/d/dev/data/wikitext-103-raw/train_tokens.npy \\
      --output data/fast/wikitext_1m_tokens.npy \\
      --size 1M

  # Legacy text extraction (without tokenization)
    python -m scripts/data/prepare_training_data.py \
      --source /mnt/d/dev/data/wikitext-103-raw/train_normalized.txt \\
      --output data/fast/wikitext_10mb.txt \\
      --size 10M

  # Overwrite existing files
    python -m scripts/data/prepare_training_data.py \
      --source /mnt/d/dev/data/wikitext-103-raw/train.txt \\
      --output data/fast/wikitext_new_tokens.npy \\
      --size 1M \\
      --normalize \\
      --tokenize \\
      --force
        """,
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source file on slow storage (text or .npy tokens)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file on fast storage (e.g., data/fast/wikitext_500k_tokens.npy)",
    )
    parser.add_argument(
        "--size",
        required=True,
        help="Target size (e.g., 500K tokens, 1M tokens, 10M bytes)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize text before tokenizing (caches normalized version on slow disk)",
    )
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Tokenize to .npy format (caches tokens on slow disk)",
    )
    parser.add_argument(
        "--tokenizer",
        default="char",
        help="Tokenizer name (default: char)",
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
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    args = parser.parse_args()

    prepare_training_data(
        source_path=args.source,
        output_path=args.output,
        size=args.size,
        normalize=args.normalize,
        tokenize=args.tokenize,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.mode,
        vocab_size=args.vocab_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
