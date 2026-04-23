"""ALiBi: Attention with Linear Biases.

Press et al. (2021) — "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation" — https://arxiv.org/abs/2108.12409

Adds a fixed linear bias to attention logits: bias[h, i, j] = -slope_h * (i - j)
for positions i >= j (causal). No learnable parameters.

Flash attention supports ALiBi natively via alibi_slopes argument.
"""

import torch
import torch.nn as nn


class ALiBi(nn.Module):
    """ALiBi attention bias (no parameters).

    Args:
        num_heads: Number of attention heads.
    """

    slopes: torch.Tensor  # (num_heads,) float32

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        # slopes[h] = 2^(-8*(h+1)/num_heads) for h in 0..num_heads-1
        h = torch.arange(1, num_heads + 1, dtype=torch.float32)
        slopes = 2.0 ** (-8.0 * h / num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    def get_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute ALiBi bias matrix.

        Returns:
            (num_heads, T, T) — lower-triangular region contains -slope * distance;
            upper-triangular is 0 (future positions handled by causal mask separately).
        """
        positions = torch.arange(T, device=device)
        # distance[i, j] = max(0, i - j) — non-negative, 0 for upper triangle
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).clamp(min=0)  # (T, T)
        bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0).to(
            self.slopes.dtype
        )  # (num_heads, T, T)
        return bias.to(dtype)
