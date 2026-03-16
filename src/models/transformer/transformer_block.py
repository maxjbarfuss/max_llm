"""Transformer block module (pre-norm attention + feedforward with residuals)."""

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
        attention_backend: Attention backend to use (default: "flash").
            Options: "flash", "sage", "xformers", "standard"

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
        attention_backend: str = "flash",
        num_layers: int = 1,
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

        # Attention and feedforward (num_layers for scaled residual init)
        self.attention = CausalMultiHeadAttention(
            d_model, num_heads, dropout, attention_backend, num_layers=num_layers
        )
        self.feedforward = FeedForward(d_model, ff_expansion_ratio, dropout, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 3
        ), f"TransformerBlock expects 3-D input (batch, seq_len, d_model), got shape {x.shape}"
        assert (
            x.is_floating_point()
        ), f"TransformerBlock expects floating-point input, got {x.dtype}"
        assert (
            x.shape[-1] == self.d_model
        ), f"TransformerBlock input last dim {x.shape[-1]} != d_model {self.d_model}"
        in_shape = x.shape

        # Pre-norm attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # Pre-norm feedforward with residual
        ff_out = self.feedforward(self.norm2(x))
        x = x + ff_out

        assert (
            x.shape == in_shape
        ), f"TransformerBlock output shape {x.shape} != input shape {in_shape}"
        return x
