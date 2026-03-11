"""Feed-forward network module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Standard transformer feed-forward block: Linear(d_model -> 4*d_model)
    -> GELU -> Linear(4*d_model -> d_model).

    Args:
        d_model: Model dimension (embedding size).
        expansion_ratio: Ratio for hidden dimension (default: 4).
        dropout: Dropout probability (default: 0.0).

    Attributes:
        linear1: First linear layer (d_model -> 4*d_model).
        activation: GELU activation function.
        dropout: Dropout layer.
        linear2: Second linear layer (4*d_model -> d_model).
    """

    def __init__(
        self, d_model: int, expansion_ratio: int = 4, dropout: float = 0.0, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        hidden_dim = d_model * expansion_ratio

        self.linear1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=True)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights: Xavier for linear1, GPT-2 scaled residual for linear2."""
        nn.init.xavier_uniform_(self.linear1.weight)

        # GPT-2 scaled residual init: linear2 feeds directly into the residual stream.
        # Scale down by 1/sqrt(2 * num_layers) to prevent variance growth with depth.
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=residual_std)

        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        assert (
            x.ndim == 3
        ), f"FeedForward expects 3-D input (batch, seq_len, d_model), got shape {x.shape}"
        assert x.is_floating_point(), f"FeedForward expects floating-point input, got {x.dtype}"
        assert (
            x.shape[-1] == self.d_model
        ), f"FeedForward input last dim {x.shape[-1]} != d_model {self.d_model}"
        in_shape = x.shape
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        assert x.shape == in_shape, f"FeedForward output shape {x.shape} != input shape {in_shape}"
        return x
