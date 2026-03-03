#!/usr/bin/env python3
"""
Create a custom BPE vocabulary of exactly 256 tokens optimized for small models.
Uses byte-pair encoding starting from raw bytes, learning 256 total tokens.
"""
from collections import defaultdict


def get_bytes(text):
    """Convert text to list of bytes."""
    return list(text.encode("utf-8"))


def get_stats(tokens):
    """Count frequency of adjacent token pairs."""
    pairs = defaultdict(int)
    for token in tokens:
        for i in range(len(token) - 1):
            pairs[token[i], token[i + 1]] += 1
    return pairs


def merge_vocab(tokens, pair):
    """Merge the most frequent pair in all token sequences."""
    new_tokens = []
    for token in tokens:
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(("MERGE", token[i], token[i + 1]))
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_tokens.append(new_token)
    return new_tokens, pair


def learn_bpe(text, vocab_size=256, num_merges=None):
    """
    Learn BPE vocabulary of exact size.
    Start with 256 bytes, learn (vocab_size - 256) merges.
    """
    if num_merges is None:
        num_merges = vocab_size - 256

    print(f"Learning BPE: 256 initial bytes + {num_merges} merges = {vocab_size} vocab")

    # Initialize with bytes
    tokens = [tuple(get_bytes(word)) for word in text.split()]
    print(f"  Initialized with {len(tokens)} words")

    # Learn merges
    vocab = set()
    for i in range(num_merges):
        pairs = get_stats(tokens)
        if not pairs:
            print(f"  Stopped at merge {i}: no more pairs to merge")
            break

        best = max(pairs, key=pairs.get)
        tokens, pair = merge_vocab(tokens, best)
        vocab.add(pair)

        if (i + 1) % max(1, num_merges // 10) == 0:
            print(f"  Merge {i+1}/{num_merges}: learned pair {pair}")

    return vocab, tokens


def build_bpe_vocab(text, vocab_size=256):
    """Build final vocabulary mapping."""
    vocab, _ = learn_bpe(text, vocab_size)

    # Build token ID mapping
    token_to_id = {}
    # First 256 are raw bytes
    for i in range(256):
        token_to_id[bytes([i])] = i

    # Next are learned pairs
    idx = 256
    for pair in sorted(vocab):
        token_to_id[pair] = idx
        idx += 1

    print(f"Final vocab size: {len(token_to_id)}")
    return token_to_id


def tokenize_with_bpe(text, vocab):
    """Tokenize text using learned BPE vocabulary."""
    # This is simplified - full BPE would be more complex
    tokens = []
    for word in text.split():
        word_bytes = tuple(get_bytes(word))
        if word_bytes in vocab:
            tokens.append(vocab[word_bytes])
        else:
            # Fallback: tokenize as individual bytes
            for b in get_bytes(word):
                tokens.append(b)
    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create custom 256-token BPE vocab")
    parser.add_argument(
        "--input",
        default="/mnt/d/dev/data/wikitext-103-raw/train_normalized.txt",
        help="Input text file",
    )
    parser.add_argument("--vocab-size", type=int, default=256, help="Final vocab size")
    parser.add_argument(
        "--sample-size", type=int, default=100000, help="Sample size for learning (chars)"
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        # Sample first N characters for vocab learning
        text_sample = f.read(args.sample_size)

    print(f"Learning BPE from {len(text_sample)} characters...")
    vocab = build_bpe_vocab(text_sample, args.vocab_size)

    print("\nVocab created!")
    print(f"Total vocab size: {len(vocab)}")
    print(f"Expected final size: {args.vocab_size}")
