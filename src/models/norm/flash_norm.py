"""FlashNorm: parameter-free RMSNorm.

The learnable scale (γ) is absorbed into the weights of the adjacent linear
layer, so this module performs only the normalisation step with no stored
parameters.  This lets the norm be fused cheaply with the following matmul.

Reference: "FlashNorm: No-Weight RMS Normalisation for Faster Transformers"
Formula:
    y = x / RMS(x)
    RMS(x) = sqrt(mean(x²) + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashNorm(nn.Module):
    """Parameter-free RMSNorm.

    Identical to RMSNorm but with no learnable weight.  To preserve
    representational capacity, the scale should be absorbed into the
    weight matrix of the linear layer that follows this norm.

    Args:
        d_model: Feature dimension (used for shape checking).
        eps: Numerical stability constant (default: 1e-6).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight=None: F.rms_norm skips the scale multiply entirely.
        return F.rms_norm(x, [self.d_model], weight=None, eps=self.eps)
