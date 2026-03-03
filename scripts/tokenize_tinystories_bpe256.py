#!/usr/bin/env python3
"""Tokenize TinyStories with custom BPE and create train/val/test splits."""
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def load_and_tokenize_tinystories(tokenizer_path, output_dir, max_tokens=10_000_000):
    """Load TinyStories, tokenize with BPE, and create splits."""

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    vocab_size = tokenizer.get_vocab_size()

    print("Loading TinyStories dataset...")
    # This loads from HuggingFace cache if available
    dataset = load_dataset(
        "karpathy/tinystories-gpt4-clean",
        split="train",
        cache_dir="/mnt/d/dev/data/tinystories-gpt4-clean/hf_cache",
    )

    print(f"Dataset size: {len(dataset)} stories")

    # Tokenize
    all_tokens = []
    total_chars = 0

    for i, example in enumerate(dataset):
        if i % 5000 == 0:
            print(f"  Processing story {i}: {len(all_tokens)} tokens accumulated")

        if len(all_tokens) >= max_tokens * 1.2:  # Get 20% extra for splits
            break

        text = example["text"]
        total_chars += len(text)
        encoding = tokenizer.encode(text)
        all_tokens.extend(encoding.ids)

    all_tokens = np.array(all_tokens[: int(max_tokens * 1.2)], dtype=np.int32)

    print("\nTokenization complete:")
    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  Total characters: {total_chars}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Avg chars/token: {total_chars / len(all_tokens):.2f}")

    # Create splits: 80% train, 10% val, 10% test
    train_size = int(len(all_tokens) * 0.8)
    val_size = int(len(all_tokens) * 0.1)

    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size : train_size + val_size]
    test_tokens = all_tokens[train_size + val_size :]

    # Save
    train_path = os.path.join(output_dir, f"tinystories_train_tokens_bpe_{vocab_size}.npy")
    val_path = os.path.join(output_dir, f"tinystories_val_tokens_bpe_{vocab_size}.npy")
    test_path = os.path.join(output_dir, f"tinystories_test_tokens_bpe_{vocab_size}.npy")

    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)
    np.save(test_path, test_tokens)

    print("\nSaved splits:")
    print(f"  Train: {len(train_tokens)} tokens -> {train_path}")
    print(f"  Val:   {len(val_tokens)} tokens -> {val_path}")
    print(f"  Test:  {len(test_tokens)} tokens -> {test_path}")

    return train_tokens, val_tokens, test_tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize TinyStories with custom BPE")
    parser.add_argument(
        "--tokenizer", default="data/fast/bpe_vocab_256.json", help="Tokenizer file"
    )
    parser.add_argument("--output-dir", default="data/fast", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=10_000_000, help="Max tokens")
    args = parser.parse_args()

    load_and_tokenize_tinystories(args.tokenizer, args.output_dir, args.max_tokens)
