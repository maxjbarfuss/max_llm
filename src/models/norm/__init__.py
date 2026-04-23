"""Normalization modules for transformer architectures."""

import torch.nn as nn

from src.models.norm.crms_norm import CRMSNorm
from src.models.norm.dyt import DyT
from src.models.norm.flash_norm import FlashNorm
from src.models.norm.rms_norm import RMSNorm

_VALID_NORM_TYPES = {"layer", "rms", "flash", "dyt", "crms"}


def make_norm(norm_type: str, d_model: int) -> nn.Module:
    """Instantiate a norm layer by type name.

    Args:
        norm_type: One of "layer", "rms", "flash", "dyt", "crms".
        d_model: Feature dimension to normalize over.
    """
    if norm_type == "rms":
        return RMSNorm(d_model)
    if norm_type == "flash":
        return FlashNorm(d_model)
    if norm_type == "dyt":
        return DyT(d_model)
    if norm_type == "crms":
        return CRMSNorm(d_model)
    return nn.LayerNorm(d_model)


__all__ = ["RMSNorm", "FlashNorm", "DyT", "CRMSNorm", "make_norm", "_VALID_NORM_TYPES"]
