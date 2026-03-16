"""Feed-forward network module."""

import torch.nn as nn

from src.models.feedforward.feedforward import FeedForward
from src.models.feedforward.relu2_ffn import ReLU2FFN
from src.models.feedforward.swiglu import SwiGLU, swiglu_intermediate_size
from src.models.feedforward.xielu_ffn import xIELUFFN

_FFN_TYPES = {"gelu", "swiglu", "relu2", "xielu"}


def make_ffn(
    ffn_type: str,
    d_model: int,
    intermediate_size: int,
    dropout: float = 0.0,
    num_layers: int = 1,
) -> nn.Module:
    """Factory for feed-forward network variants.

    Args:
        ffn_type:          One of "gelu", "swiglu", "relu2", "xielu".
        d_model:           Model dimension.
        intermediate_size: Hidden dimension. For SwiGLU, use swiglu_intermediate_size(d_model).
        dropout:           Dropout probability.
        num_layers:        Total transformer layers (for scaled residual init).
    """
    assert ffn_type in _FFN_TYPES, f"ffn_type must be one of {_FFN_TYPES}, got '{ffn_type}'"
    if ffn_type == "swiglu":
        return SwiGLU(d_model, intermediate_size, dropout, num_layers)
    if ffn_type == "relu2":
        return ReLU2FFN(d_model, intermediate_size, dropout, num_layers)
    if ffn_type == "xielu":
        return xIELUFFN(d_model, intermediate_size, dropout, num_layers)
    return FeedForward(d_model, intermediate_size // d_model, dropout, num_layers)


__all__ = ["FeedForward", "ReLU2FFN", "SwiGLU", "xIELUFFN", "make_ffn", "swiglu_intermediate_size"]
