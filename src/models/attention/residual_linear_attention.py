"""Causal linear attention (ELU+1 kernel) with a learned input residual path.

Linear attention (Katharopoulos et al., 2020) replaces the softmax with a kernel:
    Attn(Q,K,V)_i ≈ (Σ_{j≤i} φ(q_i)·φ(k_j) · v_j) / (Σ_{j≤i} φ(q_i)·φ(k_j))
where φ(x) = ELU(x)+1.  This reduces the complexity from O(T²·D) to O(T·D²).

The residual path  `res_proj(x)`  is added to the attention output to counteract
rank collapse — a known failure mode of pure linear attention — and to improve
gradient flow through the module.

    output = out_proj(linear_attn(x)) + res_proj(x)

Note: RoPE is incompatible with this module because the ELU+1 kernel breaks the
rotation equivariance that RoPE depends on.  Use pos_type='learned' or 'alibi'.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.position.add_rope import AdditiveRoPE
from src.models.position.rope import RotaryEmbedding


class ResidualLinearAttention(nn.Module):
    """Causal linear attention (ELU+1 kernel) with a learned input residual.

    Architecture per forward pass:
        phi_q = ELU(Q) + 1                                          # (B, T, H, D)
        phi_k = ELU(K) + 1                                          # (B, T, H, D)
        KV_cum[i] = Σ_{j≤i} outer(phi_k[j], v[j])                  # (B, T, H, D, D) cumsum
        K_cum[i]  = Σ_{j≤i} phi_k[j]                               # (B, T, H, D) cumsum
        linear_out[i] = KV_cum[i] @ phi_q[i] / (K_cum[i]·phi_q[i] + ε)
        output = out_proj(linear_out) + res_proj(x)

    Memory: O(B·T·H·D²) for the outer-product table.  For typical settings
    (head_dim=64, T=2048, B=4, H=8, bf16) this is ~512 MB — acceptable.

    GQA/MQA are supported via the same `num_kv_heads` interface as CausalMultiHeadAttention.
    RoPE (pos_type='rope' / 'add_rope') is NOT supported — pass rope=None and use
    pos_type='learned' or 'alibi' instead.
    """

    _EPS: float = 1e-6

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        rope: RotaryEmbedding | AdditiveRoPE | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        if rope is not None:
            raise ValueError(
                "ResidualLinearAttention does not support RoPE — the ELU+1 kernel "
                "breaks rotation equivariance. Use pos_type='learned' or 'alibi'."
            )

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.groups = num_heads // num_kv_heads
        self.dropout_p = dropout
        self.num_layers = num_layers

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        # Residual path: prevents rank collapse; starts near-zero and is learned.
        self.res_proj = nn.Linear(d_model, d_model, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        # Small init so the residual path starts close to zero
        nn.init.normal_(self.res_proj.weight, mean=0.0, std=residual_std)

    def _maybe_expand_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand KV heads from num_kv_heads to num_heads for GQA/MQA."""
        if self.groups == 1:
            return k, v
        return k.repeat_interleave(self.groups, dim=2), v.repeat_interleave(self.groups, dim=2)

    def _causal_linear_attn(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """O(T·D²) causal linear attention via cumulative outer products.

        Args:
            phi_q: (B, T, H, D) — ELU+1 feature-mapped queries
            phi_k: (B, T, H, D) — ELU+1 feature-mapped keys
            v:     (B, T, H, D) — values

        Returns:
            out:   (B, T, H, D)
        """
        # Outer product phi_k ⊗ v: KV[b,t,h,d_k,d_v] = phi_k[b,t,h,d_k] * v[b,t,h,d_v]
        kv = phi_k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, T, H, D, D)
        kv_cum = kv.cumsum(dim=1)  # (B, T, H, D, D)  — Σ_{j≤t} outer(phi_k_j, v_j)
        k_cum = phi_k.cumsum(dim=1)  # (B, T, H, D)   — Σ_{j≤t} phi_k_j

        # Numerator: for each (b,t,h): KV_cum[t] @ phi_q[t]
        #   (B, T, H, D, D) * (B, T, H, D, 1) → sum over d_k dim → (B, T, H, D)
        num = (kv_cum * phi_q.unsqueeze(-1)).sum(dim=-2)  # (B, T, H, D)

        # Denominator: K_cum[t] · phi_q[t]  → (B, T, H, 1)
        den = (k_cum * phi_q).sum(dim=-1, keepdim=True)  # (B, T, H, 1)

        return num / (den + self._EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        assert d_model == self.d_model, (
            f"Input d_model ({d_model}) does not match module d_model ({self.d_model})"
        )

        q = self.q_proj(x)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        k, v = self._maybe_expand_kv(k, v)

        phi_q = F.elu(q) + 1  # (B, T, H, D)
        phi_k = F.elu(k) + 1  # (B, T, H, D)

        attn_out = self._causal_linear_attn(phi_q, phi_k, v)  # (B, T, H, D)
        attn_out = attn_out.contiguous().view(B, T, self.d_model)

        if self.training and self.dropout_p > 0.0:
            attn_out = F.dropout(attn_out, p=self.dropout_p)

        return self.out_proj(attn_out) + self.res_proj(x)
