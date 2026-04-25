"""Learned positional embedding module (max_seq_len × d_model)."""

import torch
import torch.nn as nn


class LearnedPositionEmbedding(nn.Module):
    """Learnable absolute positional embedding table.

    Initialises weights from N(0, 0.02) following GPT-style practice.

    Args:
        max_seq_len: Maximum supported sequence length.
        d_model: Embedding dimensionality (must match token embedding).
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """Return positional embeddings (1, T, d_model); only sequence length is used.

        Args:
            x: Input token ids with shape (batch, seq_len).
            pos_offset: Starting absolute position for this chunk.
                Use 0 for training/prefill and cache.length during token-by-token generation.
        """
        assert (
            x.ndim == 2
        ), f"LearnedPositionEmbedding expects 2-D input (batch, seq_len), got shape {x.shape}"
        T = x.shape[1]
        start = pos_offset
        end = pos_offset + T
        assert end <= self.embedding.num_embeddings, (
            f"seq_len {T} exceeds max_seq_len {self.embedding.num_embeddings} "
            f"(pos_offset={pos_offset}, position range [{start}, {end}))"
        )
        positions = torch.arange(start, end, device=x.device)
        out = self.embedding(positions).unsqueeze(0)  # (1, T, d_model)
        assert out.shape == (
            1,
            T,
            self.embedding.embedding_dim,
        ), f"LearnedPositionEmbedding output shape mismatch: expected {(1, T, self.embedding.embedding_dim)}, got {out.shape}"
        return out
