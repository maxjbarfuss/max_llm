"""Rotary Position Embedding (RoPE) with optional YaRN NTK-by-parts context scaling.

Su et al. (2021) — https://arxiv.org/abs/2104.09864
YaRN: Peng et al. (2023) — https://arxiv.org/abs/2309.00591

Base RoPE encodes relative position by rotating Q and K vectors before attention.
YaRN extends the effective context window at inference by applying frequency-dependent
scaling: high-frequency dimensions (fast-rotating) are left unchanged; low-frequency
dimensions (slow-rotating) are linearly interpolated; the transition is a smooth ramp.

Formula (per head, per position m):
    theta_i = 1 / (base^(2i / head_dim))   for i in 0 .. head_dim//2 - 1
    [x'_2i, x'_{2i+1}] = R(m * theta_i) @ [x_2i, x_{2i+1}]

Implemented via the rotate-half trick (Llama-style):
    rotate_half(x) = cat([-x2, x1], dim=-1)
    apply_rope(x, cos, sin) = x * cos + rotate_half(x) * sin
"""

import math

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
        cos: (1, T, 1, head_dim) — already cast to x.dtype, broadcast over batch and heads
        sin: (1, T, 1, head_dim)
    """
    return x * cos + _rotate_half(x) * sin


def _apply_yarn_scaling(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_seq_len: int,
) -> torch.Tensor:
    """NTK-by-parts frequency scaling for YaRN context extension.

    Applies frequency-dependent scaling so that:
    - High-frequency dims (wavelength < orig_len/high_freq_factor): unchanged
    - Low-frequency dims (wavelength > orig_len/low_freq_factor): scaled by 1/s
    - Transition region: smooth linear blend between the two

    Args:
        inv_freq:             (head_dim//2,) base inverse frequencies
        scaling_factor:       s = target_ctx / orig_ctx (must be >= 1)
        low_freq_factor:      threshold below which a dim is fully interpolated (α)
        high_freq_factor:     threshold above which a dim is fully unchanged (β)
        original_max_seq_len: training context length used to compute wavelength thresholds
    """
    low_wavelen = original_max_seq_len / low_freq_factor
    high_wavelen = original_max_seq_len / high_freq_factor

    wavelens = 2 * math.pi / inv_freq  # per-dim wavelength

    # smooth: 0 → fully interpolated (low-freq); 1 → unchanged (high-freq)
    smooth = ((original_max_seq_len / wavelens) - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth = smooth.clamp(0.0, 1.0)

    scaled = inv_freq / scaling_factor
    new_inv_freq = scaled + smooth * (inv_freq - scaled)

    # Exact override at hard boundaries to eliminate floating-point drift
    new_inv_freq = torch.where(wavelens > low_wavelen, scaled, new_inv_freq)
    new_inv_freq = torch.where(wavelens < high_wavelen, inv_freq, new_inv_freq)

    return new_inv_freq


class RotaryEmbedding(nn.Module):
    """Rotary position embedding with precomputed cos/sin cache.

    Optional YaRN NTK-by-parts scaling extends the effective context window
    beyond training length without fine-tuning.  Set scaling_factor > 1.0 to
    enable; attention modules must multiply their softmax_scale by attn_scale.

    Args:
        head_dim:              Dimension per attention head (must be even).
        max_seq_len:           Maximum sequence length to precompute (default: 2048).
        base:                  Frequency base θ (default: 10000).
        scaling_factor:        YaRN scale s = target_len / orig_len (1.0 = disabled).
        low_freq_factor:       Low-frequency threshold α (default: 1.0).
        high_freq_factor:      High-frequency threshold β (default: 4.0).
        original_max_seq_len:  Training context length for YaRN thresholds.
                               Defaults to max_seq_len when None.
    """

    cos_cache: torch.Tensor  # (1, max_seq_len, 1, head_dim)
    sin_cache: torch.Tensor  # (1, max_seq_len, 1, head_dim)

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        if scaling_factor > 1.0:
            orig_len = original_max_seq_len if original_max_seq_len is not None else max_seq_len
            inv_freq = _apply_yarn_scaling(
                inv_freq, scaling_factor, low_freq_factor, high_freq_factor, orig_len
            )
            self.attn_scale = math.sqrt(1.0 + 0.1 * math.log(scaling_factor))
        else:
            self.attn_scale = 1.0

        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim//2)

        # Llama-style: concat freqs with itself so shape = (1, max_seq_len, 1, head_dim)
        # Pre-shaped for broadcasting over (B, T, num_heads, head_dim) — no hot-path unsqueezes.
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0).unsqueeze(2)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0).unsqueeze(2)

        # Non-persistent: recomputed on device transfer, not saved in state_dict
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to queries and keys.

        Args:
            q: (B, T, num_heads, head_dim)
            k: (B, T, num_heads, head_dim)

        Returns:
            Rotated (q, k) with same shape and dtype.
        """
        T = q.shape[1]
        # Cast to input dtype here so _apply_rope stays in q/k dtype throughout (no upcast).
        cos = self.cos_cache[:, :T].to(q.dtype)  # (1, T, 1, head_dim)
        sin = self.sin_cache[:, :T].to(q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
