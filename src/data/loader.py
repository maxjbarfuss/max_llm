"""In-memory text data loader for Phase 2 LM training.

Converts raw text into non-overlapping (input, target) chunk pairs suitable
for next-token-prediction training. Target is input shifted right by one.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharTokenizer


class TextChunkDataset(Dataset):
    """LM dataset: splits a flat token sequence into (input, target) chunks.

    Each chunk is seq_len tokens long. Target is input shifted right by one::

        input  = tokens[i   : i + seq_len]
        target = tokens[i+1 : i + seq_len + 1]

    Chunks are non-overlapping (stride = seq_len). Requires at least
    seq_len + 1 tokens to produce one sample.
    """

    def __init__(self, tokens: list[int], seq_len: int) -> None:
        self._tokens = torch.tensor(tokens, dtype=torch.long)
        self._seq_len = seq_len
        self._n_samples = max(0, (len(tokens) - 1) // seq_len)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self._seq_len
        x = self._tokens[start : start + self._seq_len]
        y = self._tokens[start + 1 : start + self._seq_len + 1]
        return x, y


def make_data_loaders(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Encode text and return (train_loader, val_loader).

    Args:
        text: Raw text to tokenize.
        tokenizer: Tokenizer used to encode the text.
        seq_len: Chunk size (context window).
        batch_size: Number of chunks per batch.
        validation_split: Fraction of tokens reserved for validation.
        seed: RNG seed for training-set shuffle.

    Returns:
        (train_loader, val_loader) pair.
    """
    tokens = tokenizer.encode(text)
    split = int(len(tokens) * (1.0 - validation_split))

    train_ds = TextChunkDataset(tokens[:split], seq_len)
    val_ds = TextChunkDataset(tokens[split:], seq_len)

    g = torch.Generator().manual_seed(seed)
    train_loader: DataLoader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader: DataLoader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
