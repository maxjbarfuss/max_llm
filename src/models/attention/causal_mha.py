"""Causal multi-head self-attention module with multiple backend support."""

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention.masks import document_causal_bias
from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding

if TYPE_CHECKING:
    from src.models.kv_cache import LayerKVCache

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore[assignment, unused-ignore]
    FLASH_ATTN_AVAILABLE = False

# Try to import xformers attention
try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularMask

    XFORMERS_AVAILABLE = True
except ImportError:
    memory_efficient_attention = None  # type: ignore[assignment, unused-ignore]
    LowerTriangularMask = None  # type: ignore[assignment, unused-ignore]
    XFORMERS_AVAILABLE = False

# Try to import Sage Attention
try:
    from sageattention import sageattn as sage_attn_func

    SAGE_ATTN_AVAILABLE = True
except ImportError:
    sage_attn_func = None  # type: ignore[assignment, unused-ignore]
    SAGE_ATTN_AVAILABLE = False

# Type alias for cleaner annotations
AttnBiasModule = ALiBi | RelativePositionBias  # both have get_bias(T, device, dtype)


class CausalMultiHeadAttention(nn.Module):
    """Causal attention with pluggable backends supporting MHA, GQA, and MQA.

    Set num_kv_heads to control the attention variant:
    - num_kv_heads == num_heads (default): standard MHA
    - 1 < num_kv_heads < num_heads: GQA (num_heads must be divisible by num_kv_heads)
    - num_kv_heads == 1: MQA

    Backends: "flash", "sage", "xformers", "standard".
    Flash Attention handles GQA natively; other backends expand K/V via repeat_interleave.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
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

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert (
            num_heads % num_kv_heads == 0
        ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.groups = num_heads // num_kv_heads  # repeat factor for K/V in non-flash backends
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if isinstance(rope, RotaryEmbedding) and rope.attn_scale != 1.0:
            self.softmax_scale *= rope.attn_scale
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.rope = rope
        self.attn_bias = attn_bias

        valid_backends = {"flash", "sage", "xformers", "standard"}
        assert (
            attention_backend in valid_backends
        ), f"attention_backend must be one of {valid_backends}, got {attention_backend}"

        self.attention_backend = self._select_attention_backend(attention_backend)
        self._causal_bias_cache: dict[tuple[int, int, str, int, torch.dtype], torch.Tensor] = {}

        # Separate Q and KV projections to support GQA/MQA (no bias — modern practice)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self._reset_parameters()

    def _select_attention_backend(self, requested: str) -> str:
        if requested == "flash":
            if FLASH_ATTN_AVAILABLE:
                return "flash"
            print(
                "Warning: Flash Attention requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            return "standard"
        if requested == "sage":
            if SAGE_ATTN_AVAILABLE:
                return "sage"
            print(
                "Warning: Sage Attention requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install sageattention"
            )
            return "standard"
        if requested == "xformers":
            if XFORMERS_AVAILABLE:
                return "xformers"
            print(
                "Warning: xFormers requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install xformers"
            )
            return "standard"
        return "standard"

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        # GPT-2 scaled residual init: scale down by 1/sqrt(2 * num_layers) to prevent
        # variance growth with depth.
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _maybe_expand_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand K/V from num_kv_heads to num_heads for GQA/MQA (NHD layout)."""
        if self.groups == 1:
            return k, v
        return k.repeat_interleave(self.groups, dim=2), v.repeat_interleave(self.groups, dim=2)

    def _prepare_attn_context(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, str]:
        """Return (attn_bias_tensor, effective_backend) for this forward pass."""
        attn_bias = (
            self.attn_bias.get_bias(T, device, dtype) if self.attn_bias is not None else None
        )
        backend = self.attention_backend
        if isinstance(self.attn_bias, RelativePositionBias) and backend != "standard":
            backend = "standard"
        return attn_bias, backend

    def _runtime_dropout(self) -> float:
        """Use dropout only in training mode."""
        return self.dropout_p if self.training else 0.0

    def _causal_bias(
        self, T_q: int, T_k: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Causal bias for (T_q, T_k) attention with bottom-right alignment.

        For T_q == T_k: standard upper-triangular mask.
        For T_q < T_k: query i attends to keys 0..i+(T_k-T_q) (generation with KV-cache).
        PyTorch's is_causal=True uses upper-left convention and is wrong when T_q < T_k.
        """
        key = (T_q, T_k, device.type, device.index or -1, dtype)
        cached = self._causal_bias_cache.get(key)
        if cached is None:
            cached = torch.triu(
                torch.full((T_q, T_k), float("-inf"), device=device, dtype=dtype),
                T_k - T_q + 1,
            )
            self._causal_bias_cache[key] = cached
        return cached

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
    ) -> torch.Tensor:
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
            alibi_slopes=alibi_slopes,
        )
        assert out is not None
        return out

    def _sage_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # tensor_layout="NHD": (B, T, num_heads, head_dim)
        assert sage_attn_func is not None
        return sage_attn_func(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=True,
            sm_scale=self.softmax_scale,
        )

    def _xformers_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: torch.Tensor | None,
        dropout_p: float,
    ) -> torch.Tensor:
        assert LowerTriangularMask is not None and memory_efficient_attention is not None
        if attn_bias is None:
            return memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask(), p=dropout_p)

        T_q, T_k = q.shape[1], k.shape[1]
        causal = self._causal_bias(T_q, T_k, q.device, q.dtype)
        combined = (causal.unsqueeze(0) + attn_bias).unsqueeze(0).expand(q.shape[0], -1, -1, -1)
        return memory_efficient_attention(q, k, v, attn_bias=combined, p=dropout_p)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: torch.Tensor | None,
        dropout_p: float,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # F.scaled_dot_product_attention expects (B, num_heads, T, head_dim)
        q_std = q.transpose(1, 2)
        k_std = k.transpose(1, 2)
        v_std = v.transpose(1, 2)

        T_q, T_k = q_std.shape[2], k_std.shape[2]

        if attn_bias is None and document_ids is None and T_q == T_k:
            # Square case: PyTorch's is_causal=True is correct and efficient.
            out = F.scaled_dot_product_attention(
                q_std,
                k_std,
                v_std,
                dropout_p=dropout_p,
                is_causal=True,
                scale=self.softmax_scale,
            )
        else:
            # Asymmetric (KV-cache generation) or bias present: explicit bottom-right mask.
            # PyTorch's is_causal=True uses upper-left convention and is wrong when T_q < T_k.
            causal = self._causal_bias(T_q, T_k, q_std.device, q_std.dtype)
            mask = causal if attn_bias is None else causal + attn_bias
            if document_ids is not None:
                if T_q != T_k:
                    raise ValueError("document_ids attention mask requires square self-attention")
                mask = mask + document_causal_bias(document_ids, q_std.dtype)
            out = F.scaled_dot_product_attention(
                q_std,
                k_std,
                v_std,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.softmax_scale,
            )

        return out.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: "LayerKVCache | None" = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, T, d_model = x.shape
        assert (
            d_model == self.d_model
        ), f"Input d_model ({d_model}) does not match module d_model ({self.d_model})"

        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        k, v = self.kv_proj(x).chunk(2, dim=-1)  # each (B, T, num_kv_heads * head_dim)

        # Reshape: Q→(B,T,num_heads,head_dim), KV→(B,T,num_kv_heads,head_dim) [NHD layout]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K; pos_offset from cache so rotations are correct during generation
        pos_offset = kv_cache.length if kv_cache is not None else 0
        if self.rope is not None:
            q, k = self.rope(q, k, pos_offset=pos_offset)

        # Append new K/V to cache and retrieve the full sequence K/V
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Compute attention bias (None if not using ALiBi/RelPosBias)
        _attn_bias, effective_backend = self._prepare_attn_context(T, q.device, q.dtype)
        if document_ids is not None:
            effective_backend = "standard"
        runtime_dropout = self._runtime_dropout()

        # Expand K/V from num_kv_heads to num_heads for backends without native GQA support.
        # Flash Attention 2 handles GQA natively; sage/xformers/standard require expansion.
        if effective_backend != "flash":
            k, v = self._maybe_expand_kv(k, v)

        if effective_backend == "flash":
            attn_output = self._flash_attention(q, k, v, runtime_dropout)

        elif effective_backend == "sage":
            attn_output = self._sage_attention(q, k, v)

        elif effective_backend == "xformers":
            attn_output = self._xformers_attention(q, k, v, _attn_bias, runtime_dropout)

        else:  # standard — F.scaled_dot_product_attention (PyTorch 2.0+)
            attn_output = self._standard_attention(
                q, k, v, _attn_bias, runtime_dropout, document_ids=document_ids
            )

        attn_output = attn_output.contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_output)
