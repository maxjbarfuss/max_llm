"""Token embedding module (vocab_size × d_model)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding table.

    Initialises weights from N(0, 0.02) following GPT-style practice.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding dimensionality.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token indices.

        Args:
            x: Token indices of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        assert (
            x.ndim == 2
        ), f"TokenEmbedding expects 2-D input (batch, seq_len), got shape {x.shape}"
        assert x.dtype == torch.long, f"TokenEmbedding expects dtype=torch.long, got {x.dtype}"
        out = self.embedding(x)
        assert out.shape == (
            *x.shape,
            self.embedding.embedding_dim,
        ), f"TokenEmbedding output shape mismatch: expected {(*x.shape, self.embedding.embedding_dim)}, got {out.shape}"
        return out
