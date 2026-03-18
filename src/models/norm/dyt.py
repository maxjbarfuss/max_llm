"""DyT: Dynamic Tanh normalisation.

Replaces layer normalisation with a bounded learned nonlinearity.  No
running statistics, no mean/variance computation — just a scalar gate and
per-channel scale.

Paper: "Transformers without Normalisation" — Zhai et al., 2025.
Formula:
    y = γ ⊙ tanh(α · x)
    α: learnable scalar     (init: 0.5)
    γ: learnable per-channel scale (init: 1.0), analogous to RMSNorm weight
"""

import torch
import torch.nn as nn


class DyT(nn.Module):
    """Dynamic Tanh normalisation layer.

    Args:
        d_model: Feature dimension.
        alpha_init: Initial value for the scalar gate α (default: 0.5).
    """

    def __init__(self, d_model: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.to(x.dtype) * torch.tanh(self.alpha.to(x.dtype) * x)
