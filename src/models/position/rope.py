"""Rotary Position Embedding (RoPE).

Su et al. (2021) — https://arxiv.org/abs/2104.09864

Encodes relative position by rotating Q and K vectors before attention.
Key properties:
- No learnable parameters (pure positional encoding).
- Relative positions encoded via dot-product invariance: q_m · k_n depends only on (m-n).
- Naturally length-extrapolable beyond training context.

Formula (per head, per position m):
    theta_i = 1 / (base^(2i / head_dim))   for i in 0 .. head_dim//2 - 1
    [x'_2i, x'_{2i+1}] = R(m * theta_i) @ [x_2i, x_{2i+1}]

Implemented via the rotate-half trick (Llama-style):
    rotate_half(x) = cat([-x2, x1], dim=-1)   where x1=x[:.,:head_dim//2], x2=x[:,head_dim//2:]
    apply_rope(x, cos, sin) = x * cos + rotate_half(x) * sin
"""

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half, negated."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x using precomputed cos/sin.

    Args:
        x:   (B, T, num_heads, head_dim)
        cos: (1, T, 1, head_dim) — broadcast over batch and heads
        sin: (1, T, 1, head_dim)
    """
    return (x * cos + _rotate_half(x) * sin).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding with precomputed cos/sin cache.

    Args:
        head_dim:    Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute (default: 2048).
        base:        Frequency base θ (default: 10000, as in original paper).
    """

    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim//2)

        # Llama-style: concat freqs with itself so cos/sin shape = (max_seq_len, head_dim)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        # Non-persistent: recomputed on device transfer, not saved in state_dict
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to queries and keys.

        Args:
            q: (B, T, num_heads, head_dim)
            k: (B, T, num_heads, head_dim)

        Returns:
            Rotated (q, k) with same shape.
        """
        T = q.shape[1]
        cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)
        sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(2)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
