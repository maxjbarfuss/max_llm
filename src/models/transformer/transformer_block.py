"""Transformer block module (pre-norm attention + feedforward with residuals)."""

import torch
import torch.nn as nn

from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.models.feedforward import make_ffn
from src.models.norm import make_norm
from src.models.position.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, causal attention, and feedforward.

    Architecture (pre-norm with residual connections):
    1. norm1 → Causal Multi-Head Attention → Residual Add
    2. norm2 → FeedForward → Residual Add

    Args:
        d_model:           Model dimension (embedding size).
        num_heads:         Number of attention heads.
        dropout:           Dropout probability (default: 0.0).
        ff_expansion_ratio: Expansion ratio for FFN hidden dim (default: 4). Ignored when
                           intermediate_size is provided explicitly.
        attention_backend: One of "flash", "sage", "xformers", "standard" (default: "flash").
        norm_type:         "layer" (LayerNorm) or "rms" (RMSNorm, default: "layer").
        rope:              Optional RotaryEmbedding to apply to Q/K.
        ffn_type:          One of "gelu", "swiglu", "relu2", "xielu" (default: "gelu").
        intermediate_size: FFN hidden dimension. Overrides ff_expansion_ratio when set.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_expansion_ratio: int = 4,
        attention_backend: str = "flash",
        num_layers: int = 1,
        norm_type: str = "layer",
        rope: RotaryEmbedding | None = None,
        ffn_type: str = "gelu",
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert norm_type in {
            "layer",
            "rms",
        }, f"norm_type must be 'layer' or 'rms', got '{norm_type}'"

        self.d_model = d_model
        self.num_heads = num_heads

        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)

        self.attention = CausalMultiHeadAttention(
            d_model, num_heads, dropout, attention_backend, num_layers=num_layers, rope=rope
        )

        _intermediate = (
            intermediate_size if intermediate_size is not None else d_model * ff_expansion_ratio
        )
        self.feedforward = make_ffn(ffn_type, d_model, _intermediate, dropout, num_layers)

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

        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))

        assert (
            x.shape == in_shape
        ), f"TransformerBlock output shape {x.shape} != input shape {in_shape}"
        return x
