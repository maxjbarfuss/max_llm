"""Sequence chunking helpers for feed-forward modules."""

from collections.abc import Callable

import torch


def validate_ffn_chunk_size(ffn_chunk_size: int | None) -> None:
    """Validate optional sequence chunk size for FFN execution."""
    if ffn_chunk_size is not None and ffn_chunk_size <= 0:
        raise ValueError("ffn_chunk_size must be positive when provided")


def chunk_sequence(
    x: torch.Tensor,
    ffn_chunk_size: int | None,
    chunk_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply ``chunk_fn`` to sequence chunks and concatenate along T."""
    if ffn_chunk_size is None or x.shape[1] <= ffn_chunk_size:
        return chunk_fn(x)
    return torch.cat(
        [
            chunk_fn(x[:, start : start + ffn_chunk_size, :])
            for start in range(0, x.shape[1], ffn_chunk_size)
        ],
        dim=1,
    )
