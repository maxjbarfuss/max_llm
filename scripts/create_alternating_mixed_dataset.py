#!/usr/bin/env python3
"""
Create 3:1 alternating mixed dataset (Wikitext:TinyStories) with train/val/test splits.
Uses all available wikitext (10M tokens) and tinystories (6M tokens).
Creates 3:1 alternating chunks, then splits 80/10/10.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("/home/max/dev/max_llm/data/fast")
OUTPUT_DIR = Path("/home/max/dev/max_llm/data/fast")


def load_data(path):
    """Load tokenized dataset."""
    data = np.load(path)
    print(f"  Loaded {path.name}: {len(data):,} tokens")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 3:1 alternating mixed dataset")
    parser.add_argument("--vocab-size", type=int, default=256, help="BPE vocab size suffix")
    parser.add_argument(
        "--wikitext-file",
        default=None,
        help="Override wikitext token file name (inside data/fast)",
    )
    args = parser.parse_args()

    vocab_size = args.vocab_size
    wikitext_file = args.wikitext_file or f"wikitext_10m_tokens_custom_bpe_{vocab_size}.npy"

    print("=" * 80)
    print("Creating 3:1 Alternating Mixed Dataset")
    print("=" * 80)

    # Load datasets
    print("\n[1/5] Loading datasets...")
    wikitext = load_data(DATA_DIR / wikitext_file)
    tinystories_train = load_data(DATA_DIR / f"tinystories_train_tokens_bpe_{vocab_size}.npy")
    tinystories_val = load_data(DATA_DIR / f"tinystories_val_tokens_bpe_{vocab_size}.npy")
    tinystories_test = load_data(DATA_DIR / f"tinystories_test_tokens_bpe_{vocab_size}.npy")

    # Combine tinystories
    print("\n[2/5] Combining tinystories splits...")
    tinystories = np.concatenate([tinystories_train, tinystories_val, tinystories_test])
    print(f"  Combined tinystories: {len(tinystories):,} tokens")
    print(f"  Wikitext: {len(wikitext):,} tokens")
    print(f"  Total: {len(wikitext) + len(tinystories):,} tokens")

    # Create 3:1 alternating chunks
    print("\n[3/5] Creating 3:1 alternating chunks...")
    chunk_size = 50000  # 50k tokens per chunk
    wikitext_chunks = []
    tinystories_chunks = []

    # Split into chunks
    for i in range(0, len(wikitext) - chunk_size, chunk_size):
        wikitext_chunks.append(wikitext[i : i + chunk_size])

    for i in range(0, len(tinystories) - chunk_size, chunk_size):
        tinystories_chunks.append(tinystories[i : i + chunk_size])

    print(f"  Wikitext chunks: {len(wikitext_chunks)}")
    print(f"  TinyStories chunks: {len(tinystories_chunks)}")

    # Create alternating sequence: 3 wikitext, 1 tinystories
    mixed_chunks = []
    wikitext_idx = 0
    tinystories_idx = 0

    while wikitext_idx < len(wikitext_chunks) and tinystories_idx < len(tinystories_chunks):
        # Add 3 wikitext chunks
        if wikitext_idx < len(wikitext_chunks):
            mixed_chunks.append(wikitext_chunks[wikitext_idx])
            wikitext_idx += 1
        if wikitext_idx < len(wikitext_chunks):
            mixed_chunks.append(wikitext_chunks[wikitext_idx])
            wikitext_idx += 1
        if wikitext_idx < len(wikitext_chunks):
            mixed_chunks.append(wikitext_chunks[wikitext_idx])
            wikitext_idx += 1

        # Add 1 tinystories chunk
        if tinystories_idx < len(tinystories_chunks):
            mixed_chunks.append(tinystories_chunks[tinystories_idx])
            tinystories_idx += 1

    # Append remaining chunks
    while wikitext_idx < len(wikitext_chunks):
        mixed_chunks.append(wikitext_chunks[wikitext_idx])
        wikitext_idx += 1

    while tinystories_idx < len(tinystories_chunks):
        mixed_chunks.append(tinystories_chunks[tinystories_idx])
        tinystories_idx += 1

    print(f"  Total alternating chunks: {len(mixed_chunks)}")
    print(
        f"  Chunk ratio: {sum(1 for i, c in enumerate(mixed_chunks) if i < 20 and (i+1) % 4 != 0)}/{sum(1 for i in range(20))//4} wikitext:tinystories (first 20)"
    )

    # Concatenate all chunks
    mixed = np.concatenate(mixed_chunks)
    print(f"  Final mixed dataset: {len(mixed):,} tokens")

    # Split into train/val/test (80/10/10)
    print("\n[4/5] Splitting into train/val/test (80/10/10)...")
    total = len(mixed)
    train_split = int(total * 0.8)
    val_split = int(total * 0.9)

    mixed_train = mixed[:train_split]
    mixed_val = mixed[train_split:val_split]
    mixed_test = mixed[val_split:]

    print(f"  Train: {len(mixed_train):,} tokens ({100*len(mixed_train)/total:.1f}%)")
    print(f"  Val:   {len(mixed_val):,} tokens ({100*len(mixed_val)/total:.1f}%)")
    print(f"  Test:  {len(mixed_test):,} tokens ({100*len(mixed_test)/total:.1f}%)")

    # Save
    print("\n[5/5] Saving datasets...")
    output_files = {
        f"mixed_wikitext_tinystories_3to1_train_bpe_{vocab_size}.npy": mixed_train,
        f"mixed_wikitext_tinystories_3to1_val_bpe_{vocab_size}.npy": mixed_val,
        f"mixed_wikitext_tinystories_3to1_test_bpe_{vocab_size}.npy": mixed_test,
    }

    for filename, data in output_files.items():
        path = OUTPUT_DIR / filename
        np.save(path, data)
        print(f"  {path.name}: {len(data):,} tokens ({len(data)*4/1024/1024:.1f}MB)")

    # Save metadata
    metadata = {
        "description": "3:1 alternating mixed dataset (Wikitext:TinyStories) with custom BPE tokenizer",
        "source": [
            wikitext_file,
            f"tinystories_train/val/test_tokens_bpe_{vocab_size}.npy",
        ],
        "total_tokens": len(mixed),
        "wikitext_tokens": len(wikitext),
        "tinystories_tokens": len(tinystories),
        "ratio": "3:1 (wikitext:tinystories)",
        "chunk_size": chunk_size,
        "splits": {
            "train": {"tokens": int(len(mixed_train)), "percent": 80},
            "val": {"tokens": int(len(mixed_val)), "percent": 10},
            "test": {"tokens": int(len(mixed_test)), "percent": 10},
        },
        "vocab_size": vocab_size,
    }

    meta_path = OUTPUT_DIR / f"mixed_wikitext_tinystories_3to1_bpe_{vocab_size}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  {meta_path.name}")

    print("\n" + "=" * 80)
    print("✓ Mixed dataset created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
