"""ReLU² (Squared ReLU) feed-forward network.

So et al. (2021) — "Primer: Searching for Efficient Transformers for Language Modeling"
https://arxiv.org/abs/2109.08668

Also: "ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs"
https://arxiv.org/abs/2402.03804

Architecture: identical to GELU FFN but with activation f(x) = relu(x)².

Key properties vs SwiGLU:
- No gate — single projection path, fewer FLOPs.
- Naturally sparse activations (dead neurons for x ≤ 0) — better for MoE / sparse inference.
- Competitive perplexity: 10.352 vs SwiGLU 10.517 on 1.1B/126B token run.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feedforward.chunking import chunk_sequence, validate_ffn_chunk_size


class ReLU2FFN(nn.Module):
    """Feed-forward network with ReLU² activation.

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
        self.linear2 = nn.Linear(intermediate_size, d_model, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=residual_std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"ReLU2FFN expects (B, T, d_model), got {x.shape}"
        assert x.shape[-1] == self.d_model, f"ReLU2FFN input dim {x.shape[-1]} != {self.d_model}"
        return chunk_sequence(x, self.ffn_chunk_size, self._forward_chunk)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x)).pow(2)))
