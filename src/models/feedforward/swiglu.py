"""SwiGLU feed-forward network.

Shazeer (2020) — https://arxiv.org/abs/2002.05202
Used in LLaMA, Mistral, Gemma, PaLM.

Architecture:
    down_proj(silu(gate_proj(x)) * up_proj(x))

Key differences from GELU FFN:
- Gated: gate and up projections are multiplied element-wise before projecting down.
- No bias on any projection (Llama convention).
- intermediate_size ≈ d_model * 8/3 (2/3 of standard 4× expansion) — use swiglu_intermediate_size().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feedforward.chunking import chunk_sequence, validate_ffn_chunk_size


def swiglu_intermediate_size(d_model: int) -> int:
    """Compute SwiGLU intermediate size: 4 × d_model × 2/3, rounded to next multiple of 256."""
    raw = int(d_model * 4 * 2 / 3)
    return ((raw + 255) // 256) * 256


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Args:
        d_model:           Model dimension.
        intermediate_size: Hidden dimension for gate/up projections.
                           Use swiglu_intermediate_size(d_model) for the standard Llama sizing.
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

        # No bias — Llama convention for gated FFNs
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int) -> None:
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=residual_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"SwiGLU expects (B, T, d_model), got {x.shape}"
        assert x.shape[-1] == self.d_model, f"SwiGLU input dim {x.shape[-1]} != {self.d_model}"
        return chunk_sequence(x, self.ffn_chunk_size, self._forward_chunk)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))
