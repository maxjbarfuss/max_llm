"""Causal multi-head self-attention module with multiple backend support."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore[assignment, unused-ignore]
    FLASH_ATTN_AVAILABLE = False

# Try to import xformers attention
try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularMask

    XFORMERS_AVAILABLE = True
except ImportError:
    memory_efficient_attention = None  # type: ignore[assignment, unused-ignore]
    LowerTriangularMask = None  # type: ignore[assignment, unused-ignore]
    XFORMERS_AVAILABLE = False

# Try to import Sage Attention
try:
    from sageattention import sageattn as sage_attn_func

    SAGE_ATTN_AVAILABLE = True
except ImportError:
    sage_attn_func = None  # type: ignore[assignment, unused-ignore]
    SAGE_ATTN_AVAILABLE = False

# Type alias for cleaner annotations
AttnBiasModule = ALiBi | RelativePositionBias  # both have get_bias(T, device, dtype)


class CausalMultiHeadAttention(nn.Module):
    """Causal MHA with pluggable backends: "flash", "sage", "xformers", "standard"."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_backend: str = "flash",
        num_layers: int = 1,
        rope: RotaryEmbedding | AdditiveRoPE | None = None,
        attn_bias: ALiBi | RelativePositionBias | None = None,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.rope = rope
        self.attn_bias = attn_bias

        valid_backends = {"flash", "sage", "xformers", "standard"}
        assert (
            attention_backend in valid_backends
        ), f"attention_backend must be one of {valid_backends}, got {attention_backend}"

        self.attention_backend = self._select_attention_backend(attention_backend)

        # Fused QKV projection (no bias — modern practice for Q/K/V)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self._reset_parameters()

    def _select_attention_backend(self, requested: str) -> str:
        if requested == "flash":
            if FLASH_ATTN_AVAILABLE:
                return "flash"
            print(
                "Warning: Flash Attention requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            return "standard"
        if requested == "sage":
            if SAGE_ATTN_AVAILABLE:
                return "sage"
            print(
                "Warning: Sage Attention requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install sageattention"
            )
            return "standard"
        if requested == "xformers":
            if XFORMERS_AVAILABLE:
                return "xformers"
            print(
                "Warning: xFormers requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install xformers"
            )
            return "standard"
        return "standard"

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        # GPT-2 scaled residual init: scale down by 1/sqrt(2 * num_layers) to prevent
        # variance growth with depth.
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        assert (
            d_model == self.d_model
        ), f"Input d_model ({d_model}) does not match module d_model ({self.d_model})"

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)  # each (B, T, d_model)

        # Reshape to (B, T, num_heads, head_dim) — NHD layout used by flash/sage/xformers
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K before attention (NHD layout)
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Compute attention bias (None if not using ALiBi/RelPosBias)
        _attn_bias: torch.Tensor | None = None
        if self.attn_bias is not None:
            _attn_bias = self.attn_bias.get_bias(T, q.device, q.dtype)  # (num_heads, T, T)

        # For RelPosBias: fall back to standard attention (flash/sage/xformers don't easily
        # support arbitrary per-head bias matrices).
        effective_backend = self.attention_backend
        if isinstance(self.attn_bias, RelativePositionBias) and effective_backend != "standard":
            effective_backend = "standard"

        if effective_backend == "flash":
            assert flash_attn_func is not None
            alibi_slopes: torch.Tensor | None = None
            if isinstance(self.attn_bias, ALiBi):
                alibi_slopes = self.attn_bias.slopes.to(torch.float32)
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
                alibi_slopes=alibi_slopes,
            )
            assert attn_output is not None
            attn_output = attn_output.view(B, T, self.d_model)

        elif effective_backend == "sage":
            # tensor_layout="NHD": (B, T, num_heads, head_dim)
            assert sage_attn_func is not None
            attn_output = sage_attn_func(
                q,
                k,
                v,
                tensor_layout="NHD",
                is_causal=True,
                sm_scale=1.0 / math.sqrt(self.head_dim),
            )
            attn_output = attn_output.view(B, T, self.d_model)

        elif effective_backend == "xformers":
            # LowerTriangularMask avoids allocating a [B, num_heads, T, T] bias tensor
            assert LowerTriangularMask is not None and memory_efficient_attention is not None
            if _attn_bias is not None:
                causal_mask = torch.triu(
                    torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype), 1
                )
                combined = (
                    (causal_mask.unsqueeze(0) + _attn_bias).unsqueeze(0).expand(B, -1, -1, -1)
                )
                attn_output = memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=combined,
                    p=self.dropout_p if self.training else 0.0,
                )
            else:
                attn_output = memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=LowerTriangularMask(),
                    p=self.dropout_p if self.training else 0.0,
                )
            attn_output = attn_output.view(B, T, self.d_model)

        else:  # standard — F.scaled_dot_product_attention (PyTorch 2.0+)
            # Expects (B, num_heads, T, head_dim)
            q_std = q.transpose(1, 2)
            k_std = k.transpose(1, 2)
            v_std = v.transpose(1, 2)
            if _attn_bias is not None:
                causal_mask = torch.triu(
                    torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype), 1
                )
                combined_mask = causal_mask.unsqueeze(0) + _attn_bias  # (num_heads, T, T)
                attn_output = F.scaled_dot_product_attention(
                    q_std,
                    k_std,
                    v_std,
                    attn_mask=combined_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    q_std,
                    k_std,
                    v_std,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(attn_output)
