"""xIELU feed-forward network.

"Deriving Activation Functions Using Integration" (2024)
https://arxiv.org/abs/2411.13010
Reference impl: https://github.com/rubber-duck-debug/xielu

xIELU is derived by integrating trainable affine transformations of the ELU gradient,
yielding a piecewise activation with quadratic positive branch and exponential negative branch.

Formula:
    x > 0:  α_p · x² + β · x         (quadratic; gradient linearly increases)
    x ≤ 0:  α_n · expm1(clamp(x)) - α_n · x + β · x   (exponential; continuous at 0)

Parameters:
    α_p = softplus(alpha_p_raw)              > 0
    α_n = β + softplus(alpha_n_raw)          > β
    β   = 0.5  (fixed; ensures gradient continuity at x = 0)

Trainable per module: 2 scalars (shared across all channels/positions).
Performance: 10.207 ppl vs ReLU² 10.352 / SwiGLU 10.517 on 1.1B model / 126B tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_BETA = 0.5
_EPS = -1e-6  # clamp_max for expm1 stability
_ALPHA_P_INIT = 0.8
_ALPHA_N_INIT = 0.8


def _inv_softplus(y: float) -> float:
    """softplus⁻¹(y) = log(exp(y) - 1)  — used to init parameters."""
    import math

    return math.log(math.exp(y) - 1.0)


class xIELU(nn.Module):
    """xIELU activation function (arXiv:2411.13010).

    2 shared trainable scalars per module; applied identically across all features/positions.
    """

    def __init__(
        self,
        alpha_p_init: float = _ALPHA_P_INIT,
        alpha_n_init: float = _ALPHA_N_INIT,
        beta: float = _BETA,
        eps: float = _EPS,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps
        # Store pre-softplus values so softplus(stored) == init at construction
        self.alpha_p = nn.Parameter(torch.tensor([_inv_softplus(alpha_p_init)]))
        self.alpha_n = nn.Parameter(torch.tensor([_inv_softplus(alpha_n_init - beta)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        pos = alpha_p * x * x + self.beta * x
        neg = alpha_n * torch.expm1(torch.clamp_max(x, self.eps)) - alpha_n * x + self.beta * x
        return torch.where(x > 0, pos, neg)


class xIELUFFN(nn.Module):
    """Feed-forward network with xIELU activation.

    Same structure as GELU FFN (linear → activation → dropout → linear) but
    activation is replaced by xIELU.

    Args:
        d_model:           Model dimension.
        intermediate_size: Hidden dimension (typically 4 × d_model).
        dropout:           Dropout probability (default: 0.0).
        num_layers:        Total transformer layers — used for GPT-2 scaled residual init.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.linear1 = nn.Linear(d_model, intermediate_size, bias=True)
        self.activation = xIELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(intermediate_size, d_model, bias=True)

        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=residual_std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"xIELUFFN expects (B, T, d_model), got {x.shape}"
        assert x.shape[-1] == self.d_model, f"xIELUFFN input dim {x.shape[-1]} != {self.d_model}"
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
