"""RMSNorm: Root Mean Square Layer Normalization.

Zhang & Sennrich (2019) — https://arxiv.org/abs/1910.07467

Replaces LayerNorm in Llama-style architectures.  Key differences:
- No mean subtraction (only RMS scaling).
- No bias parameter.
- Learnable per-element gain (weight) initialized to 1.0.

Formula:
    y = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x²) + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        d_model: Number of features (last dimension of input tensor).
        eps: Small constant for numerical stability (default: 1e-6).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.d_model
        ), f"RMSNorm: input last dim {x.shape[-1]} != d_model {self.d_model}"
        # Cast weight to input dtype so F.rms_norm dispatches to the fused CUDA kernel.
        return F.rms_norm(x, [self.d_model], self.weight.to(x.dtype), self.eps)
