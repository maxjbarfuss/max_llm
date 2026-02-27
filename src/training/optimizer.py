"""Optimizer utilities for training."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def configure_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """Configure parameter groups with selective weight decay.

    Excludes weight decay from:
        - All bias parameters
        - All LayerNorm parameters (weight and bias)
        - All parameters with ndim < 2 (scalars, 1D vectors)

    This is standard practice for transformer training to avoid
    degrading learned biases and normalization parameters.

    Args:
        model: Model to configure parameter groups for.
        weight_decay: Weight decay coefficient.
        learning_rate: Learning rate.
        betas: Adam beta parameters.
        eps: Adam epsilon.

    Returns:
        List of parameter group dicts suitable for torch.optim.Adam.

    Example:
        >>> param_groups = configure_optimizer_param_groups(model, weight_decay=0.1, lr=1e-3)
        >>> optimizer = torch.optim.Adam(param_groups)
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude from weight decay if:
        # - bias parameter (name ends with .bias)
        # - LayerNorm parameter (name contains layernorm or layer_norm, case insensitive)
        # - scalar or 1D parameter (ndim < 2)
        if (
            name.endswith(".bias")
            or "layernorm" in name.lower()
            or "layer_norm" in name.lower()
            or param.ndim < 2
        ):
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
