"""Multi-Head Latent Attention (MLA) with decoupled RoPE.

MLA compresses token states into a lower-dimensional latent space for content
Q/K/V projections, while keeping a separate RoPE branch for positional signal.
The positional RoPE branch is decoupled from latent compression so we retain
strong relative-position encoding without paying full-rank Q/K costs.

Projection layout (two fused GEMMs replace four separate ones):
  Content path:  x → latent_proj → z
                 q_content_proj(z)          → q_content  [num_heads   × content_head_dim]
                 kv_proj(z)     → split     → k_content  [num_kv_heads × content_head_dim]
                                             → v          [num_kv_heads × head_dim]
  RoPE path:    qk_rope_proj(x) → split    → q_rope     [num_heads   × rope_head_dim]
                                             → k_rope     [num_kv_heads × rope_head_dim]
  Combine:       Q = cat(q_content, q_rope), K = cat(k_content, k_rope)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class MultiHeadLatentAttention(nn.Module):
    """MLA with decoupled RoPE branch.

    Each Q/K head has two subspaces:
    - content  (from latent-compressed path)
    - rope     (from raw token states, rotated by RoPE or AddRoPE)

    V is produced from the latent content path only.
    See module docstring for the fused-projection layout.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        latent_dim: int | None = None,
        rope_head_dim: int | None = None,
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

        if latent_dim is None:
            latent_dim = d_model
        assert latent_dim > 0, "latent_dim must be positive"
        assert (
            latent_dim % num_heads == 0
        ), f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"

        if rope is None:
            raise ValueError("MLA requires RoPE/AddRoPE for decoupled positional branch")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.rope = rope
        self.attn_bias = attn_bias
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if isinstance(rope, RotaryEmbedding) and rope.attn_scale != 1.0:
            self.softmax_scale *= rope.attn_scale

        default_rope_dim = min(32, max(2, self.head_dim // 2))
        if default_rope_dim % 2 != 0:
            default_rope_dim -= 1
        rope_dim = rope_head_dim if rope_head_dim is not None else default_rope_dim
        if rope_dim <= 0 or rope_dim >= self.head_dim or rope_dim % 2 != 0:
            raise ValueError(
                "rope_head_dim must be even, positive, and smaller than head_dim "
                f"(got rope_head_dim={rope_dim}, head_dim={self.head_dim})"
            )
        self.rope_head_dim = rope_dim
        self.content_head_dim = self.head_dim - self.rope_head_dim

        valid_backends = {"flash", "sage", "xformers", "standard"}
        assert (
            attention_backend in valid_backends
        ), f"attention_backend must be one of {valid_backends}, got {attention_backend}"
        self.attention_backend = self._select_attention_backend(attention_backend)
        self._causal_bias_cache: dict[tuple[int, str, int, torch.dtype], torch.Tensor] = {}

        # Content path: shared latent, then Q and fused KV
        self.latent_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.q_content_proj = nn.Linear(latent_dim, num_heads * self.content_head_dim, bias=False)
        # Fused k_content + v from latent in one GEMM
        self.kv_proj = nn.Linear(
            latent_dim,
            num_kv_heads * self.content_head_dim + num_kv_heads * self.head_dim,
            bias=False,
        )

        # Decoupled RoPE path: fused q_rope + k_rope from raw x in one GEMM
        self.qk_rope_proj = nn.Linear(
            d_model,
            (num_heads + num_kv_heads) * self.rope_head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self._reset_parameters()

    def _select_attention_backend(self, requested: str) -> str:
        if requested == "flash" and FLASH_ATTN_AVAILABLE:
            return "flash"
        if requested == "sage" and SAGE_ATTN_AVAILABLE:
            return "sage"
        if requested == "xformers" and XFORMERS_AVAILABLE:
            return "xformers"
        return "standard"

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.xavier_uniform_(self.q_content_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.qk_rope_proj.weight)
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        nn.init.zeros_(self.out_proj.bias)

    def _apply_rope_decoupled(
        self, q_rope: torch.Tensor, k_rope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply positional encoding to the rope subspace of Q and K.

        The shared rope module is built for the full head_dim; we slice its
        cache to rope_head_dim so the narrower rope subspace gets the correct
        (lower-frequency) rotations.
        """
        T = q_rope.shape[1]
        if isinstance(self.rope, RotaryEmbedding):
            cos = self.rope.cos_cache[:, :T, :, : self.rope_head_dim].to(q_rope.dtype)
            sin = self.rope.sin_cache[:, :T, :, : self.rope_head_dim].to(q_rope.dtype)
            return (
                q_rope * cos + _rotate_half(q_rope) * sin,
                k_rope * cos + _rotate_half(k_rope) * sin,
            )
        # AdditiveRoPE: add positional signal
        enc = self.rope.enc_cache[:, :T, :, : self.rope_head_dim].to(q_rope.dtype)
        return q_rope + enc, k_rope + enc

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

    def _sage_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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
        T: int,
        attn_bias: torch.Tensor | None,
        dropout_p: float,
    ) -> torch.Tensor:
        assert LowerTriangularMask is not None and memory_efficient_attention is not None
        if attn_bias is None:
            return memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask(), p=dropout_p)
        causal = self._causal_bias(T, q.device, q.dtype)
        combined = (causal.unsqueeze(0) + attn_bias).unsqueeze(0).expand(q.shape[0], -1, -1, -1)
        return memory_efficient_attention(q, k, v, attn_bias=combined, p=dropout_p)

    def _causal_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return cached upper-triangular causal bias with -inf above diagonal."""
        key = (T, device.type, device.index or -1, dtype)
        cached = self._causal_bias_cache.get(key)
        if cached is None:
            cached = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), 1)
            self._causal_bias_cache[key] = cached
        return cached

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
        attn_bias: torch.Tensor | None,
        dropout_p: float,
    ) -> torch.Tensor:
        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attn_bias is None:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, dropout_p=dropout_p, is_causal=True, scale=self.softmax_scale
            )
        else:
            # SDPA forbids is_causal=True when attn_mask is set;
            # combine explicit causal bias with the additive position bias.
            causal = self._causal_bias(T, q_t.device, q_t.dtype)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=causal + attn_bias,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.softmax_scale,
            )
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Content path: compress to latent, project Q and fused KV
        z = self.latent_proj(x)
        q_content = self.q_content_proj(z).view(B, T, self.num_heads, self.content_head_dim)
        k_content, v = self.kv_proj(z).split(
            [self.num_kv_heads * self.content_head_dim, self.num_kv_heads * self.head_dim], dim=-1
        )
        k_content = k_content.view(B, T, self.num_kv_heads, self.content_head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # Decoupled RoPE path: project from raw x, apply positional encoding
        q_rope, k_rope = self.qk_rope_proj(x).split(
            [self.num_heads * self.rope_head_dim, self.num_kv_heads * self.rope_head_dim], dim=-1
        )
        q_rope = q_rope.view(B, T, self.num_heads, self.rope_head_dim)
        k_rope = k_rope.view(B, T, self.num_kv_heads, self.rope_head_dim)
        q_rope, k_rope = self._apply_rope_decoupled(q_rope, k_rope)

        # Assemble full Q/K heads
        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)

        attn_bias = (
            self.attn_bias.get_bias(T, q.device, q.dtype) if self.attn_bias is not None else None
        )
        dropout_p = self.dropout_p if self.training else 0.0
        backend = self.attention_backend
        if isinstance(self.attn_bias, RelativePositionBias):
            backend = "standard"

        # GQA expansion: flash handles internally; others need explicit expand
        if backend != "flash" and self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)

        if backend == "flash":
            out = self._flash_attention(q, k, v, dropout_p)
        elif backend == "sage":
            out = self._sage_attention(q, k, v)
        elif backend == "xformers":
            out = self._xformers_attention(q, k, v, T, attn_bias, dropout_p)
        else:
            out = self._standard_attention(q, k, v, T, attn_bias, dropout_p)

        return self.out_proj(out.contiguous().view(B, T, self.d_model))
