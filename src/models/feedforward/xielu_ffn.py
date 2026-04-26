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
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feedforward.chunking import chunk_sequence, validate_ffn_chunk_size

_BETA = 0.5
_EPS = -1e-6  # clamp_max for expm1 stability
_ALPHA_P_INIT = 0.8
_ALPHA_N_INIT = 0.8
_ApplyFn = Callable[..., torch.Tensor]


def _inv_softplus(y: float) -> float:
    """softplus⁻¹(y) = log(exp(y) - 1)  — used to init parameters."""
    return math.log(math.exp(y) - 1.0)


class _XIELUFunction(torch.autograd.Function):
    """Memory-efficient xIELU forward/backward.

    Forward saves only the input tensor and the two raw α parameters; the
    pos/neg/expm1/mask intermediates over (B, T, H) are recomputed in
    backward. This roughly halves the activation memory carried between
    forward and backward for the FFN's xIELU stage, raising the OOM ceiling.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        alpha_p_raw: torch.Tensor,
        alpha_n_raw: torch.Tensor,
        beta: float,
        eps: float,
    ) -> torch.Tensor:
        alpha_p = F.softplus(alpha_p_raw)
        alpha_n = beta + F.softplus(alpha_n_raw)
        pos = alpha_p * x * x + beta * x
        neg_x = torch.clamp_max(x, eps)
        neg = alpha_n * torch.expm1(neg_x) - alpha_n * x + beta * x
        out = torch.where(x > 0, pos, neg)
        ctx.save_for_backward(x, alpha_p_raw, alpha_n_raw)
        ctx.beta = beta
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_out,) = grad_outputs
        x, alpha_p_raw, alpha_n_raw = ctx.saved_tensors
        beta: float = ctx.beta
        eps: float = ctx.eps
        needs_x, needs_ap, needs_an = ctx.needs_input_grad[:3]

        alpha_p = F.softplus(alpha_p_raw)
        alpha_n = beta + F.softplus(alpha_n_raw)
        mask = x > 0  # bool, (B, T, H)

        grad_x: torch.Tensor | None = None
        grad_alpha_p_raw: torch.Tensor | None = None
        grad_alpha_n_raw: torch.Tensor | None = None

        # Recompute negative-branch primitives once; reuse for x and α_n grads.
        neg_x = torch.clamp_max(x, eps)
        # exp(neg_x) is bounded above by exp(eps) ≈ 1, and below by 0.
        exp_neg = torch.exp(neg_x)
        # clamp gates the gradient when x > eps (a thin sliver of (eps, 0]).
        clamp_active = (x <= eps).to(x.dtype)

        if needs_x:
            d_pos_dx = 2.0 * alpha_p * x + beta
            d_neg_dx = alpha_n * exp_neg * clamp_active - alpha_n + beta
            grad_x = grad_out * torch.where(mask, d_pos_dx, d_neg_dx)

        if needs_ap:
            # d out / d alpha_p = x*x on positive branch, 0 on negative branch.
            # alpha_p = softplus(alpha_p_raw) ⇒ chain by sigmoid(alpha_p_raw).
            contrib_p = grad_out * x * x
            contrib_p = torch.where(
                mask, contrib_p, torch.zeros((), dtype=x.dtype, device=x.device)
            )
            grad_alpha_p_raw = contrib_p.sum().reshape(alpha_p_raw.shape) * torch.sigmoid(
                alpha_p_raw
            )

        if needs_an:
            # d out / d alpha_n = expm1(neg_x) - x on negative branch, 0 on positive.
            # alpha_n = beta + softplus(alpha_n_raw) ⇒ chain by sigmoid(alpha_n_raw).
            expm1_neg = torch.expm1(neg_x)
            contrib_n = grad_out * (expm1_neg - x)
            contrib_n = torch.where(
                mask, torch.zeros((), dtype=x.dtype, device=x.device), contrib_n
            )
            grad_alpha_n_raw = contrib_n.sum().reshape(alpha_n_raw.shape) * torch.sigmoid(
                alpha_n_raw
            )

        return grad_x, grad_alpha_p_raw, grad_alpha_n_raw, None, None


class xIELU(nn.Module):
    """xIELU activation function (arXiv:2411.13010).

    2 shared trainable scalars per module; applied identically across all features/positions.

    Uses a custom autograd Function that saves only the input tensor and the
    two scalar α parameters between forward and backward. The pos/neg/expm1/
    where intermediates over (B, T, H) are recomputed in backward, which is
    a strict memory win at modest extra FLOPs.
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
        apply = cast(_ApplyFn, _XIELUFunction.apply)  # pyright: ignore
        return apply(x, self.alpha_p, self.alpha_n, self.beta, self.eps)


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
        ffn_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        validate_ffn_chunk_size(ffn_chunk_size)
        self.d_model = d_model
        self.ffn_chunk_size = ffn_chunk_size

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
        return chunk_sequence(x, self.ffn_chunk_size, self._forward_chunk)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
