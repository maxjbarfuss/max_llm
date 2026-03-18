"""Attention Residuals (AttnRes) — depth-wise attention over layer outputs.

Replaces fixed residual accumulation (h_l = h_{l-1} + f(h_{l-1})) with learned
softmax attention over all preceding layer outputs:

    h_l = sum_i  alpha_{i->l} * v_i
    alpha_{i->l} = softmax(w_l^T · RMSNorm(k_i))   (over i < l)
    k_i = v_i = f_i(h_i)   (layer output as both key and value)

Two variants:
  - FullAttnRes: attends over every individual preceding sublayer output O(L^2).
  - BlockAttnRes: attends over N block-level summaries O(N^2), practical at scale.

Reference: "Attention Residuals", Kimi Team, MoonshotAI (2025).
           https://github.com/MoonshotAI/Attention-Residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnResidual(nn.Module):
    """Depth-wise attention residual mixer.

    Holds one learnable query vector w_l per sublayer (and one extra for the
    final output aggregation). At each call, computes softmax over the keys
    derived from the provided source tensors and returns the weighted sum.

    All query vectors are zero-initialised so that at the start of training the
    softmax weights are uniform, reducing AttnRes to an equal-weight average —
    equivalent in effect to standard residuals at init (prevents training shock).

    Args:
        num_sublayers: Total number of sublayers (2 × num_transformer_blocks).
                       An extra query is created for the final output aggregation.
        d_model: Hidden dimension.
    """

    def __init__(self, num_sublayers: int, d_model: int) -> None:
        super().__init__()
        # +1 query for the final output aggregation step
        num_queries = num_sublayers + 1
        self.queries = nn.Parameter(torch.zeros(num_queries, d_model))
        # RMSNorm on keys prevents high-magnitude layers from dominating weights.
        # No learnable scale here — we just want normalised directions.
        self.key_norm = nn.RMSNorm(d_model, elementwise_affine=False)

    def forward(self, query_idx: int, sources: list[torch.Tensor]) -> torch.Tensor:
        """Compute depth-wise attention output for one layer.

        Args:
            query_idx: Index into self.queries selecting w_l for this layer.
            sources: List of (B, T, d_model) tensors (v_0, v_1, …, v_{l-1}).
                     Must contain at least one tensor.

        Returns:
            Tensor of shape (B, T, d_model): weighted combination of sources.
        """
        # V: (B, T, n_src, d)
        V = torch.stack(sources, dim=-2)
        # Normalise keys along the d dimension for stable attention scores
        K = self.key_norm(V)  # (B, T, n_src, d)
        q = self.queries[query_idx].to(V.dtype)  # (d,)
        # Dot-product scores: q · k_i  →  (B, T, n_src)
        scores = (K * q).sum(-1)
        weights = F.softmax(scores, dim=-1)  # (B, T, n_src)
        # Weighted sum over sources
        return (weights.unsqueeze(-1) * V).sum(-2)  # (B, T, d)
