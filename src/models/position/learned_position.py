"""Learned positional embedding module (max_seq_len × d_model)."""

from __future__ import annotations

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional embeddings for sequence length T.

        Args:
            x: Token tensor of shape (batch, T) — only T is used.

        Returns:
            Positional embeddings of shape (1, T, d_model), broadcastable
            over batch dimension.
        """
        T = x.shape[1]
        assert (
            T <= self.embedding.num_embeddings
        ), f"seq_len {T} exceeds max_seq_len {self.embedding.num_embeddings}"
        positions = torch.arange(T, device=x.device)
        return self.embedding(positions).unsqueeze(0)  # (1, T, d_model)
