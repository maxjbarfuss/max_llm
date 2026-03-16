"""Token embedding module (vocab_size × d_model)."""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding table initialised from N(0, 0.02)."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 2
        ), f"TokenEmbedding expects 2-D input (batch, seq_len), got shape {x.shape}"
        assert x.dtype == torch.long, f"TokenEmbedding expects dtype=torch.long, got {x.dtype}"
        return self.embedding(x)
