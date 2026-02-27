"""Transformer block module (pre-norm attention + feedforward with residuals)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.models.feedforward.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, causal attention, and feedforward.

    Architecture (pre-norm with residual connections):
    1. LayerNorm → Causal Multi-Head Attention → Residual Add
    2. LayerNorm → FeedForward (2 Linear + GELU) → Residual Add

    Args:
        d_model: Model dimension (embedding size).
        num_heads: Number of attention heads.
        dropout: Dropout probability (default: 0.0).
        ff_expansion_ratio: Expansion ratio for hidden dimension in FFN (default: 4).

    Attributes:
        norm1: Pre-norm for attention.
        attention: CausalMultiHeadAttention module.
        norm2: Pre-norm for feedforward.
        feedforward: FeedForward module.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_expansion_ratio: int = 4,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention and feedforward
        self.attention = CausalMultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, ff_expansion_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Pre-norm attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # Pre-norm feedforward with residual
        ff_out = self.feedforward(self.norm2(x))
        x = x + ff_out

        return x
