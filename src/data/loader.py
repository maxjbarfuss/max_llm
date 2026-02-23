"""In-memory token data loader for Phase 2 LM training.

Converts a pre-encoded token sequence into non-overlapping (input, target)
chunk pairs suitable for next-token-prediction training.
Target is input shifted right by one.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

_Batch = tuple[torch.Tensor, torch.Tensor]


class TextChunkDataset(Dataset[_Batch]):
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

    def __getitem__(self, idx: int) -> _Batch:
        start = idx * self._seq_len
        x = self._tokens[start : start + self._seq_len]
        y = self._tokens[start + 1 : start + self._seq_len + 1]
        return x, y


def load_corpus_text(dataset_path: str) -> str:
    """Load plain-text corpus content from a file or directory.

    For directories, all files are read recursively in deterministic sorted
    path order and concatenated with newlines.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")

    if path.is_file():
        return path.read_text(encoding="utf-8")

    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"dataset_path directory has no files: {dataset_path}")

    chunks = [p.read_text(encoding="utf-8", errors="ignore") for p in files]
    return "\n".join(chunks)


def _train_val_split_index(token_count: int, validation_split: float | int) -> int:
    """Compute split index into train tokens for float or absolute val split."""
    if isinstance(validation_split, float):
        return int(token_count * (1.0 - validation_split))

    val_token_count = max(0, validation_split)
    return max(0, token_count - val_token_count)


def make_data_loaders(
    tokens: list[int],
    seq_len: int,
    batch_size: int,
    validation_split: float | int = 0.1,
    seed: int = 42,
) -> tuple[DataLoader[_Batch], DataLoader[_Batch]]:
    """Split a pre-encoded token sequence and return (train_loader, val_loader).

    Args:
        tokens: Pre-encoded token IDs.
        seq_len: Chunk size (context window).
        batch_size: Number of chunks per batch.
        validation_split: Fraction of tokens or absolute token count reserved
            for validation.
        seed: RNG seed for training-set shuffle.

    Returns:
        (train_loader, val_loader) pair.
    """
    split = _train_val_split_index(len(tokens), validation_split)

    train_ds = TextChunkDataset(tokens[:split], seq_len)
    val_ds = TextChunkDataset(tokens[split:], seq_len)

    g = torch.Generator().manual_seed(seed)
    train_loader: DataLoader[_Batch] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader: DataLoader[_Batch] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
