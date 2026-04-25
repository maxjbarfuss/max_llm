"""Transformer block module (pre-norm attention + feedforward with residuals)."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.models.kv_cache import LayerKVCache

from src.models.attention import make_attention
from src.models.feedforward import make_ffn
from src.models.norm import _VALID_NORM_TYPES, make_norm
from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, causal attention, and feedforward.

    Architecture (pre-norm with residual connections):
    1. norm1 → Attention → Residual Add
    2. norm2 → FeedForward → Residual Add

    Args:
        d_model:           Model dimension (embedding size).
        num_heads:         Number of attention heads.
        num_kv_heads:      K/V heads for GQA/MQA (None = MHA, 1 = MQA, N = GQA).
        dropout:           Dropout probability (default: 0.0).
        ff_expansion_ratio: Expansion ratio for FFN hidden dim (default: 4). Ignored when
                           intermediate_size is provided explicitly.
        attention_backend: One of "flash", "sage", "xformers", "standard" (default: "flash").
        norm_type:         One of "layer", "rms", "flash", "dyt", "crms" (default: "layer").
        rope:              Optional RotaryEmbedding to apply to Q/K (MHA/SWA only).
        ffn_type:          One of "gelu", "swiglu", "relu2", "xielu" (default: "gelu").
        intermediate_size: FFN hidden dimension. Overrides ff_expansion_ratio when set.
        mla_latent_dim:    Latent dimension used when attn_type="mla".
        attn_bias:         Optional ALiBi or RelativePositionBias to add to attention logits.
        attn_type:         One of "mha", "swa", "rla", "mla" (default: "mha").
        window_size:       Sliding-window width; used when attn_type="swa" (default: 256).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        ff_expansion_ratio: int = 4,
        attention_backend: str = "flash",
        num_layers: int = 1,
        norm_type: str = "layer",
        rope: RotaryEmbedding | AdditiveRoPE | None = None,
        ffn_type: str = "gelu",
        intermediate_size: int | None = None,
        mla_latent_dim: int | None = None,
        attn_bias: ALiBi | RelativePositionBias | None = None,
        attn_type: str = "mha",
        window_size: int = 256,
    ) -> None:
        super().__init__()
        self._validate_init(d_model, num_heads, norm_type)

        self.d_model = d_model
        self.num_heads = num_heads

        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)

        self.attention = make_attention(
            attn_type=attn_type,
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            attention_backend=attention_backend,
            num_layers=num_layers,
            rope=rope,
            attn_bias=attn_bias,
            mla_latent_dim=mla_latent_dim,
            window_size=window_size,
        )

        _intermediate = (
            intermediate_size if intermediate_size is not None else d_model * ff_expansion_ratio
        )
        self.feedforward = make_ffn(ffn_type, d_model, _intermediate, dropout, num_layers)

    @staticmethod
    def _validate_init(d_model: int, num_heads: int, norm_type: str) -> None:
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert (
            norm_type in _VALID_NORM_TYPES
        ), f"norm_type must be one of {_VALID_NORM_TYPES}, got '{norm_type}'"

    def _validate_input(self, x: torch.Tensor) -> None:
        assert (
            x.ndim == 3
        ), f"TransformerBlock expects 3-D input (batch, seq_len, d_model), got shape {x.shape}"
        assert (
            x.is_floating_point()
        ), f"TransformerBlock expects floating-point input, got {x.dtype}"
        assert (
            x.shape[-1] == self.d_model
        ), f"TransformerBlock input last dim {x.shape[-1]} != d_model {self.d_model}"

    def _apply_attention_residual(
        self, x: torch.Tensor, kv_cache: "LayerKVCache | None" = None
    ) -> torch.Tensor:
        return x + self.attention(self.norm1(x), kv_cache=kv_cache)

    def _apply_feedforward_residual(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.feedforward(self.norm2(x))

    def apply_attn_only(
        self, x: torch.Tensor, kv_cache: "LayerKVCache | None" = None
    ) -> torch.Tensor:
        """Return the attention sublayer output (no residual add).

        Used by AttnRes: the caller accumulates outputs externally.
        """
        return self.attention(self.norm1(x), kv_cache=kv_cache)

    def apply_ffn_only(self, x: torch.Tensor) -> torch.Tensor:
        """Return the FFN sublayer output (no residual add).

        Used by AttnRes: the caller accumulates outputs externally.
        """
        return self.feedforward(self.norm2(x))

    def forward(self, x: torch.Tensor, kv_cache: "LayerKVCache | None" = None) -> torch.Tensor:
        self._validate_input(x)
        in_shape = x.shape

        x = self._apply_attention_residual(x, kv_cache=kv_cache)
        x = self._apply_feedforward_residual(x)

        assert (
            x.shape == in_shape
        ), f"TransformerBlock output shape {x.shape} != input shape {in_shape}"
        return x
