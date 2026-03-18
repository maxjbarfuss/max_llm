"""CRMSNorm: Centered Root Mean Square Normalisation.

Extends RMSNorm with mean subtraction before normalisation — bridging the gap
between RMSNorm (no centering) and LayerNorm (centering + additive bias).
CRMSNorm centres but omits the bias term.

Formula:
    x_c = x − mean(x)
    y   = γ ⊙ x_c / RMS(x_c)
    RMS(x_c) = sqrt(mean(x_c²) + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRMSNorm(nn.Module):
    """Centered RMSNorm.

    Args:
        d_model: Feature dimension.
        eps: Numerical stability constant (default: 1e-6).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - x.mean(-1, keepdim=True)
        return F.rms_norm(x_c, [self.d_model], self.weight.to(x.dtype), self.eps)
