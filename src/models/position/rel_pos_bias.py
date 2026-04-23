"""Relative Position Bias (T5-style).

Shaw et al. (2018) / Raffel et al. (2019) — learned bias per relative position bucket.

Adds a learned (num_heads, T, T) bias to attention logits based on key–query distance.
Relative positions are bucketed and clamped to num_buckets - 1 for large distances.
"""

import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """Learned relative position bias.

    Args:
        num_heads:   Number of attention heads.
        num_buckets: Number of distance buckets (default: 32).
    """

    def __init__(self, num_heads: int, num_buckets: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        # bias_table[bucket, head] → scalar bias
        self.bias_table = nn.Embedding(num_buckets, num_heads)
        nn.init.zeros_(self.bias_table.weight)

    def get_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute relative position bias matrix.

        Returns:
            (num_heads, T, T) bias tensor.
        """
        q_pos = torch.arange(T, device=device)
        k_pos = torch.arange(T, device=device)
        # rel[i, j] = i - j: positive = key is in the past (causal direction)
        rel = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).clamp(0, self.num_buckets - 1)  # (T, T)
        bias = self.bias_table(rel)  # (T, T, num_heads)
        return bias.permute(2, 0, 1).to(dtype)  # (num_heads, T, T)
