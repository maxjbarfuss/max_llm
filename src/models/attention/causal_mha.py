"""Causal multi-head self-attention module with multiple backend support."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Try to import xformers attention
try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularMask

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# Try to import Sage Attention
try:
    from sageattention import sageattn as sage_attn_func

    SAGE_ATTN_AVAILABLE = True
except ImportError:
    SAGE_ATTN_AVAILABLE = False


class CausalMultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with multiple backend support.

    Implements masked self-attention where each position can only attend to
    itself and earlier positions in the sequence. Supports multiple attention
    backends for performance optimization:
    - "flash": Flash Attention 2 (fastest, lower memory)
    - "sage": Sage Attention (alternative to Flash Attn)
    - "xformers": xFormers memory-efficient attention (torch.compile compatible)
    - "standard": Standard PyTorch scaled dot-product attention

    Args:
        d_model: Model dimension (embedding size).
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        dropout: Dropout probability applied to attention weights (default: 0.0).
        attention_backend: Attention backend to use (default: "flash").
            Options: "flash", "sage", "xformers", "standard"

    Attributes:
        head_dim: Dimension of each attention head (d_model // num_heads).
        q_proj: Query projection (Linear d_model -> d_model).
        k_proj: Key projection (Linear d_model -> d_model).
        v_proj: Value projection (Linear d_model -> d_model).
        out_proj: Output projection (Linear d_model -> d_model).
        attention_backend: Name of the selected attention backend.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_backend: str = "flash",
        num_layers: int = 1,
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

        # Validate and select attention backend
        valid_backends = {"flash", "sage", "xformers", "standard"}
        assert (
            attention_backend in valid_backends
        ), f"attention_backend must be one of {valid_backends}, got {attention_backend}"

        # Try to use requested backend, fall back to standard if unavailable
        self.attention_backend = self._select_attention_backend(attention_backend)

        # Fused QKV projection (no bias — modern practice for Q/K/V)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Dropout (applied to attention weights or in Flash Attention)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self._reset_parameters()

    def _select_attention_backend(self, requested: str) -> str:
        """Select attention backend, falling back to standard if unavailable.

        Args:
            requested: Requested attention backend name.

        Returns:
            Name of the selected backend.
        """
        if requested == "flash":
            if FLASH_ATTN_AVAILABLE:
                return "flash"
            else:
                print(
                    "Warning: Flash Attention requested but not available. "
                    "Falling back to standard attention. "
                    "Install with: pip install flash-attn --no-build-isolation"
                )
                return "standard"

        elif requested == "sage":
            if SAGE_ATTN_AVAILABLE:
                return "sage"
            else:
                print(
                    "Warning: Sage Attention requested but not available. "
                    "Falling back to standard attention. "
                    "Install with: pip install sageattention"
                )
                return "standard"

        elif requested == "xformers":
            if XFORMERS_AVAILABLE:
                return "xformers"
            else:
                print(
                    "Warning: xFormers requested but not available. "
                    "Falling back to standard attention. "
                    "Install with: pip install xformers"
                )
                return "standard"

        else:  # standard
            return "standard"

    def _reset_parameters(self) -> None:
        """Initialize weights: Xavier for fused QKV, GPT-2 scaled residual for out_proj."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)

        # GPT-2 scaled residual init: out_proj feeds directly into the residual stream.
        # Scale down by 1/sqrt(2 * num_layers) to prevent variance growth with depth.
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, d_model = x.shape
        assert (
            d_model == self.d_model
        ), f"Input d_model ({d_model}) does not match module d_model ({self.d_model})"

        # Fused QKV projection: (B, T, 3*d_model) -> split into Q, K, V
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)  # each (B, T, d_model)

        if self.attention_backend == "flash":
            # Flash Attention path: expects (B, T, num_heads, head_dim)
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)

            # Flash Attention handles causal masking internally
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=1.0 / (self.head_dim**0.5),
                causal=True,
            )
            # Output shape: (B, T, num_heads, head_dim)
            attn_output = attn_output.view(B, T, self.d_model)

        elif self.attention_backend == "sage":
            # Sage Attention path: expects (B, T, num_heads, head_dim)
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)

            # Sage Attention handles causal masking internally
            # tensor_layout="NHD": (B, T, num_heads, head_dim); sm_scale replaces softmax_scale
            attn_output = sage_attn_func(
                q,
                k,
                v,
                tensor_layout="NHD",
                is_causal=True,
                sm_scale=1.0 / (self.head_dim**0.5),
            )
            # Output shape: (B, T, num_heads, head_dim)
            attn_output = attn_output.view(B, T, self.d_model)

        elif self.attention_backend == "xformers":
            # xFormers memory-efficient attention path
            # Reshape for xformers: (B, T, num_heads, head_dim)
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)

            # Use xFormers' efficient LowerTriangularMask for causal masking
            # This avoids creating a large [B, num_heads, T, T] tensor
            attn_bias = LowerTriangularMask()

            attn_output = memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.dropout_p if self.training else 0.0,
            )
            # Output shape: (B, T, num_heads, head_dim)
            attn_output = attn_output.view(B, T, self.d_model)

        else:  # standard attention
            # Standard attention path: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention: (B, num_heads, T, T)
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

            # Apply causal mask (upper triangular, excluding diagonal)
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

            # Compute attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, T, T)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values: (B, num_heads, T, head_dim)
            attn_output = attn_weights @ v

            # Merge heads: (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, d_model)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Final output projection
        output = self.out_proj(attn_output)
        return output
