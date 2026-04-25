"""Additive RoPE: sinusoidal position encoding added to Q/K (no rotation).

Encodes absolute position by adding interleaved sin/cos to Q and K vectors
before attention. Uses the same frequency structure as RoPE (theta_i = 1/base^(2i/head_dim))
but is additive rather than rotational. Simpler than RoPE; does not preserve vector norms.
"""

import torch
import torch.nn as nn


class AdditiveRoPE(nn.Module):
    """Additive sinusoidal position encoding for Q and K.

    Args:
        head_dim:    Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base:        Frequency base (default: 10000).
    """

    enc_cache: torch.Tensor  # (1, max_seq_len, 1, head_dim)

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim//2)

        # Interleaved sin/cos: [sin(pos*theta_0), cos(pos*theta_0), sin(pos*theta_1), ...]
        enc = torch.stack([freqs.sin(), freqs.cos()], dim=-1).flatten(-2)  # (max_seq_len, head_dim)
        enc = enc.unsqueeze(0).unsqueeze(2)  # (1, max_seq_len, 1, head_dim)

        self.register_buffer("enc_cache", enc, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding to Q and K.

        Args:
            q:          (B, T, num_heads, head_dim)
            k:          (B, T, num_heads, head_dim)
            pos_offset: Index of the first token in the full sequence (for KV-cache generation).

        Returns:
            (q + enc, k + enc) with same shape and dtype.
        """
        T = q.shape[1]
        enc = self.enc_cache[:, pos_offset : pos_offset + T].to(q.dtype)
        return q + enc, k + enc
