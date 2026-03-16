"""Normalization modules for transformer architectures."""

import torch.nn as nn

from src.models.norm.rms_norm import RMSNorm


def make_norm(norm_type: str, d_model: int) -> nn.Module:
    """Instantiate a norm layer by type name.

    Args:
        norm_type: "layer" for LayerNorm, "rms" for RMSNorm.
        d_model: Feature dimension to normalize over.
    """
    if norm_type == "rms":
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)


__all__ = ["RMSNorm", "make_norm"]
