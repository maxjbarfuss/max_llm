"""Causal multi-head self-attention module."""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalMultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with scaled dot-product.

    Implements masked self-attention where each position can only attend to
    itself and earlier positions in the sequence.

    Args:
        d_model: Model dimension (embedding size).
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        dropout: Dropout probability applied to attention weights (default: 0.0).

    Attributes:
        head_dim: Dimension of each attention head (d_model // num_heads).
        q_proj: Query projection (Linear d_model -> d_model).
        k_proj: Key projection (Linear d_model -> d_model).
        v_proj: Value projection (Linear d_model -> d_model).
        out_proj: Output projection (Linear d_model -> d_model).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Dropout (applied to attention weights)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights with Xavier uniform (PyTorch default for Linear)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
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

        # Project to Q, K, V: (B, T, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, num_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask (upper triangular, excluding diagonal)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
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
