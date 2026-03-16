"""Feed-forward network module."""

import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network: Linear(d_model -> 4*d_model) -> GELU(tanh) -> Linear(4*d_model -> d_model)."""

    def __init__(
        self, d_model: int, expansion_ratio: int = 4, dropout: float = 0.0, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        hidden_dim = d_model * expansion_ratio

        self.linear1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        # GPT-2 scaled residual init: scale down by 1/sqrt(2 * num_layers) to prevent
        # variance growth with depth.
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=residual_std)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 3
        ), f"FeedForward expects 3-D input (batch, seq_len, d_model), got shape {x.shape}"
        assert x.is_floating_point(), f"FeedForward expects floating-point input, got {x.dtype}"
        assert (
            x.shape[-1] == self.d_model
        ), f"FeedForward input last dim {x.shape[-1]} != d_model {self.d_model}"
        in_shape = x.shape
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        assert x.shape == in_shape, f"FeedForward output shape {x.shape} != input shape {in_shape}"
        return x
