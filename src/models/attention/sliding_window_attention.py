"""Sliding window causal attention — each token attends only to the most recent `window_size` tokens."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore[assignment, unused-ignore]
    FLASH_ATTN_AVAILABLE = False


class SlidingWindowAttention(nn.Module):
    """Causal multi-head attention with a sliding window of size `window_size`.

    Each token attends to at most `window_size` preceding tokens (including itself),
    reducing the O(T²) quadratic cost to O(T · window_size).

    Backend support:
    - flash:    Flash Attention 2 native window_size=(window_size-1, 0) — most efficient.
    - standard: Band-diagonal causal mask via F.scaled_dot_product_attention.
    - sage/xformers: Fall back to standard.

    GQA/MQA are supported (same `num_kv_heads` interface as CausalMultiHeadAttention).
    RoPE and ALiBi are supported. RelativePositionBias falls back to standard backend.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        attention_backend: str = "flash",
        num_layers: int = 1,
        rope: RotaryEmbedding | AdditiveRoPE | None = None,
        attn_bias: ALiBi | RelativePositionBias | None = None,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert window_size >= 1, f"window_size must be >= 1, got {window_size}"

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert (
            num_heads % num_kv_heads == 0
        ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.groups = num_heads // num_kv_heads
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if isinstance(rope, RotaryEmbedding) and rope.attn_scale != 1.0:
            self.softmax_scale *= rope.attn_scale
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.window_size = window_size
        self.rope = rope
        self.attn_bias = attn_bias

        self.attention_backend = self._select_backend(attention_backend)

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self._reset_parameters()

    def _select_backend(self, requested: str) -> str:
        """Flash Attention 2 has native sliding-window support; everything else uses standard."""
        if requested == "flash" and FLASH_ATTN_AVAILABLE:
            return "flash"
        if requested not in {"flash", "sage", "xformers", "standard"}:
            raise ValueError(
                f"attention_backend must be one of {{'flash','sage','xformers','standard'}}, "
                f"got '{requested}'"
            )
        return "standard"

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _maybe_expand_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.groups == 1:
            return k, v
        return k.repeat_interleave(self.groups, dim=2), v.repeat_interleave(self.groups, dim=2)

    def _sliding_window_causal_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Band-diagonal causal mask: -inf where j > i (future) or j < i - window_size + 1 (outside window)."""
        idx = torch.arange(T, device=device)
        row = idx.unsqueeze(1)  # (T, 1) — query positions
        col = idx.unsqueeze(0)  # (1, T) — key positions
        outside = (col > row) | (col < row - self.window_size + 1)
        mask = torch.zeros(T, T, device=device, dtype=dtype)
        mask[outside] = float("-inf")
        return mask

    def _runtime_dropout(self) -> float:
        return self.dropout_p if self.training else 0.0

    def forward(
        self,
        x: torch.Tensor,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if document_ids is not None:
            raise ValueError("Packed document masks are not supported for sliding-window attention")
        B, T, d_model = x.shape
        assert (
            d_model == self.d_model
        ), f"Input d_model ({d_model}) does not match module d_model ({self.d_model})"

        q = self.q_proj(x)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn_bias_tensor = (
            self.attn_bias.get_bias(T, q.device, q.dtype) if self.attn_bias is not None else None
        )
        dropout_p = self._runtime_dropout()

        if self.attention_backend == "flash":
            assert flash_attn_func is not None
            alibi_slopes: torch.Tensor | None = None
            if isinstance(self.attn_bias, ALiBi):
                alibi_slopes = self.attn_bias.slopes.to(torch.float32)
            out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=(self.window_size - 1, 0),
                alibi_slopes=alibi_slopes,
            )
            assert out is not None
        else:
            k, v = self._maybe_expand_kv(k, v)
            window_mask = self._sliding_window_causal_mask(T, q.device, q.dtype)
            if attn_bias_tensor is not None:
                window_mask = window_mask + attn_bias_tensor
            q_std = q.transpose(1, 2)
            k_std = k.transpose(1, 2)
            v_std = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q_std,
                k_std,
                v_std,
                attn_mask=window_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.softmax_scale,
            ).transpose(1, 2)

        out = out.contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
