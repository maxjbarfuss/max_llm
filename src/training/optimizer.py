"""Optimizer utilities for training."""

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

_ADAMW_PARAM_PATTERNS = ("token_embedding", "position_embedding", "lm_head")


def zeropower_via_newtonschulz5(
    gradient: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate nearest semi-orthogonal matrices with Newton-Schulz iterations.

    Accepts a single 2-D matrix or a batch whose last two dimensions are matrices.
    """
    if gradient.ndim < 2:
        raise ValueError("zeropower_via_newtonschulz5 expects matrix-shaped input")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    original_dtype = gradient.dtype
    matrix = gradient.detach().to(dtype=torch.float32)
    transposed = False

    if matrix.shape[-2] > matrix.shape[-1]:
        matrix = matrix.transpose(-2, -1)
        transposed = True

    matrix_norm = torch.linalg.vector_norm(matrix, dim=(-2, -1), keepdim=True)
    matrix = matrix / (matrix_norm + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.transpose(-2, -1)
        gram_sq = gram @ gram
        matrix = a * matrix + b * (gram @ matrix) + c * (gram_sq @ matrix)

    if transposed:
        matrix = matrix.transpose(-2, -1)

    return matrix.to(dtype=original_dtype)


class MuonOptimizer(torch.optim.Optimizer):
    """Muon optimizer for 2-D weight matrices."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "is_muon": True,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Apply one Muon optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buckets: dict[
                tuple[torch.device, tuple[int, int]],
                list[tuple[torch.nn.Parameter, torch.Tensor, float]],
            ] = {}

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("MuonOptimizer only supports 2-D parameters")

                grad_fp32 = parameter.grad.detach().to(dtype=torch.float32)
                state = self.state[parameter]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(grad_fp32)
                    state["momentum_buffer"] = momentum_buffer

                momentum_buffer.mul_(momentum).add_(grad_fp32)
                update = grad_fp32 + momentum * momentum_buffer if nesterov else momentum_buffer
                rows, cols = parameter.shape
                scale = max(1.0, rows / cols) ** 0.5
                key = (parameter.device, (rows, cols))
                update_buckets.setdefault(key, []).append((parameter, update, scale))

            for bucket in update_buckets.values():
                if len(bucket) == 1:
                    parameter, update, scale = bucket[0]
                    orthogonal_update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    parameter.add_(orthogonal_update.to(dtype=parameter.dtype), alpha=-lr * scale)
                    continue

                stacked_updates = torch.stack([update for _, update, _ in bucket])
                orthogonal_updates = zeropower_via_newtonschulz5(stacked_updates, steps=ns_steps)
                for orthogonal_update, (parameter, _, scale) in zip(
                    orthogonal_updates, bucket, strict=True
                ):
                    parameter.add_(orthogonal_update.to(dtype=parameter.dtype), alpha=-lr * scale)

        return loss


class MuonAdamWOptimizer(torch.optim.Optimizer):
    """Composite optimizer that delegates to Muon and AdamW."""

    def __init__(self, muon: MuonOptimizer, adamw: torch.optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw
        self.defaults: dict[str, Any] = {}
        self.state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [*self.muon.param_groups, *self.adamw.param_groups]

    @param_groups.setter
    def param_groups(self, _: Any) -> None:
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def configure_muon_optimizer(
    model: nn.Module,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    adamw_fused: bool = False,
) -> MuonAdamWOptimizer:
    """Route model parameters between Muon and AdamW."""
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))

        if parameter.ndim == 2 and not any(pattern in name for pattern in _ADAMW_PARAM_PATTERNS):
            muon_params.append(parameter)
        elif parameter.ndim >= 2:
            adamw_decay_params.append(parameter)
        else:
            adamw_no_decay_params.append(parameter)

    muon = MuonOptimizer(
        [
            {
                "params": muon_params,
                "lr": muon_lr,
                "momentum": muon_momentum,
                "nesterov": True,
                "ns_steps": muon_ns_steps,
                "is_muon": True,
            }
        ],
        lr=muon_lr,
        momentum=muon_momentum,
        ns_steps=muon_ns_steps,
    )
    adamw = torch.optim.AdamW(
        [
            {
                "params": adamw_decay_params,
                "weight_decay": weight_decay,
                "lr": adamw_lr,
                "betas": betas,
                "eps": eps,
                "is_muon": False,
            },
            {
                "params": adamw_no_decay_params,
                "weight_decay": 0.0,
                "lr": adamw_lr,
                "betas": betas,
                "eps": eps,
                "is_muon": False,
            },
        ],
        fused=adamw_fused,
    )
    return MuonAdamWOptimizer(muon=muon, adamw=adamw)


def configure_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """Configure parameter groups with selective weight decay.

    Excludes weight decay from all 1-D parameters (biases, LayerNorm weight/bias,
    and any other scalar/vector parameters).  2-D+ parameters (weight matrices,
    embeddings) receive weight decay.

    This is standard practice for transformer training to avoid
    degrading learned biases and normalization parameters.

    Args:
        model: Model to configure parameter groups for.
        weight_decay: Weight decay coefficient.
        learning_rate: Learning rate.
        betas: Adam beta parameters.
        eps: Adam epsilon.

    Returns:
        List of parameter group dicts suitable for torch.optim.AdamW.

    Example:
        >>> param_groups = configure_optimizer_param_groups(model, weight_decay=0.1, lr=1e-3)
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": learning_rate,
            "betas": betas,
            "eps": eps,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": learning_rate,
            "betas": betas,
            "eps": eps,
        },
    ]

    return param_groups
