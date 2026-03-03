#!/usr/bin/env python3
"""
Create and train a custom BPE tokenizer with exactly 256 vocab size.
"""

import numpy as np
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def create_custom_bpe_tokenizer(input_file, output_dir, vocab_size=256):
    """Train a BPE tokenizer with exact vocab size."""

    print(f"Creating custom BPE tokenizer with vocab_size={vocab_size}")

    # Create tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Set normalizer
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFC(),
        ]
    )

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        show_progress=True,
    )

    print(f"Training on {input_file}...")
    tokenizer.train([input_file], trainer=trainer)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Save tokenizer
    tokenizer_path = f"{output_dir}/bpe_vocab_{vocab_size}.json"
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    return tokenizer


def tokenize_dataset(input_file, tokenizer, output_file, max_tokens=10_000_000):
    """Tokenize a dataset file."""

    print(f"Tokenizing {input_file} -> {output_file}")
    print(f"Target: {max_tokens} tokens")

    all_tokens = []

    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  Line {i}: accumulated {len(all_tokens)} tokens")

            if len(all_tokens) >= max_tokens:
                print("Reached target token count")
                break

            if line.strip():
                encoding = tokenizer.encode(line.strip())
                all_tokens.extend(encoding.ids)

    # Trim to exact size
    all_tokens = all_tokens[:max_tokens]

    print(f"Final token count: {len(all_tokens)}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Max token ID: {max(all_tokens)}")

    # Save as numpy array
    tokens_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_file, tokens_array)
    print(f"Saved {len(all_tokens)} tokens to {output_file}")

    return tokens_array


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create custom BPE tokenizer")
    parser.add_argument(
        "--input",
        default="/mnt/d/dev/data/wikitext-103-raw/train_normalized.txt",
        help="Input file",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for tokenizer")
    parser.add_argument(
        "--output-tokens",
        default="wikitext_10m_tokens_custom_bpe_256.npy",
        help="Output token file",
    )
    parser.add_argument("--vocab-size", type=int, default=256, help="Target vocab size")
    parser.add_argument("--max-tokens", type=int, default=10_000_000, help="Max tokens to generate")
    args = parser.parse_args()

    # Create tokenizer
    tokenizer = create_custom_bpe_tokenizer(args.input, args.output_dir, args.vocab_size)

    # Tokenize dataset
    tokens = tokenize_dataset(args.input, tokenizer, args.output_tokens, args.max_tokens)

    print("\nDone!")
