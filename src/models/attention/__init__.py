"""Attention mechanisms: MHA, SWA, RLA, and MLA."""

import torch.nn as nn

from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.models.attention.multihead_latent_attention import MultiHeadLatentAttention
from src.models.attention.residual_linear_attention import ResidualLinearAttention
from src.models.attention.sliding_window_attention import SlidingWindowAttention
from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding

_VALID_ATTN_TYPES = {"mha", "swa", "rla", "mla"}


def make_attention(
    attn_type: str,
    d_model: int,
    num_heads: int,
    num_kv_heads: int | None,
    dropout: float,
    attention_backend: str,
    num_layers: int,
    rope: RotaryEmbedding | AdditiveRoPE | None,
    attn_bias: ALiBi | RelativePositionBias | None,
    mla_latent_dim: int | None = None,
    window_size: int = 256,
) -> nn.Module:
    """Factory for attention modules.

    Args:
        attn_type:         "mha" | "swa" | "rla" | "mla"
        d_model:           Model dimension.
        num_heads:         Number of query heads.
        num_kv_heads:      KV heads for GQA/MQA (None = MHA, 1 = MQA).
        dropout:           Dropout probability.
        attention_backend: Backend for MHA/SWA — "flash" | "sage" | "xformers" | "standard".
        num_layers:        Total transformer depth (for GPT-2 residual init scaling).
        rope:              Optional RoPE module (not supported by RLA).
        attn_bias:         Optional ALiBi or RelPosBias module.
        window_size:       Window width for SWA (ignored for MHA/RLA).
        mla_latent_dim:    Latent dimension for MLA content compression.

    Returns:
        An nn.Module implementing the requested attention variant.
    """
    if attn_type == "mha":
        return CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            attention_backend=attention_backend,
            num_layers=num_layers,
            rope=rope,
            attn_bias=attn_bias,
        )
    if attn_type == "swa":
        return SlidingWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            attention_backend=attention_backend,
            num_layers=num_layers,
            rope=rope,
            attn_bias=attn_bias,
        )
    if attn_type == "rla":
        return ResidualLinearAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            num_layers=num_layers,
            rope=rope,
        )
    if attn_type == "mla":
        return MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            latent_dim=mla_latent_dim,
            dropout=dropout,
            attention_backend=attention_backend,
            num_layers=num_layers,
            rope=rope,
            attn_bias=attn_bias,
        )
    raise ValueError(f"attn_type must be one of {_VALID_ATTN_TYPES}, got '{attn_type}'")


__all__ = [
    "CausalMultiHeadAttention",
    "SlidingWindowAttention",
    "ResidualLinearAttention",
    "MultiHeadLatentAttention",
    "make_attention",
]
