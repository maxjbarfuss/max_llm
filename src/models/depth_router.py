"""Mixture-of-Depths token router utilities."""

from collections.abc import Callable
from math import ceil

import torch
import torch.nn as nn


def _is_torch_compiling() -> bool:
    return bool(torch.compiler.is_compiling())


class TokenDepthRouter(nn.Module):
    """Select a per-layer token subset for expensive FFN compute.

    Attention still runs on the full sequence so RoPE positions, packed-document masks,
    and KV-cache semantics remain intact. The routed FFN path compacts selected tokens,
    applies the supplied norm + FFN only to those tokens, and scatters the delta back.
    """

    def __init__(
        self,
        d_model: int,
        capacity_fraction: float = 0.5,
        min_tokens: int = 1,
        use_soft_gate: bool = True,
        inference_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < capacity_fraction <= 1.0):
            raise ValueError("capacity_fraction must be in (0, 1]")
        if min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if inference_threshold is not None and not (0.0 <= inference_threshold <= 1.0):
            raise ValueError("inference_threshold must be in [0, 1]")

        self.capacity_fraction = capacity_fraction
        self.min_tokens = min_tokens
        self.use_soft_gate = use_soft_gate
        self.inference_threshold = inference_threshold
        self.score = nn.Linear(d_model, 1)
        self.last_selected_fraction = 1.0

    def _capacity(self, seq_len: int) -> int:
        capacity = ceil(seq_len * self.capacity_fraction)
        capacity = max(self.min_tokens, capacity)
        return min(seq_len, capacity)

    def _selection_mask(self, scores: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = scores.shape
        capacity = self._capacity(seq_len)
        mask = torch.zeros_like(scores, dtype=torch.bool)

        if capacity > 0:
            topk_idx = torch.topk(scores, k=capacity, dim=-1).indices
            mask.scatter_(dim=-1, index=topk_idx, value=True)

        if not self.training and self.inference_threshold is not None:
            threshold_mask = scores >= self.inference_threshold
            if self.min_tokens == 0:
                mask = threshold_mask
            else:
                mask = threshold_mask | mask
            if capacity < seq_len:
                capped = torch.zeros_like(mask)
                capped_scores = scores.masked_fill(~mask, -torch.inf)
                capped_idx = torch.topk(capped_scores, k=capacity, dim=-1).indices
                capped.scatter_(dim=-1, index=capped_idx, value=True)
                mask = capped & mask

        if not _is_torch_compiling():
            self.last_selected_fraction = float(mask.sum().item()) / float(batch_size * seq_len)
        return mask

    def _scores(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.score(x.to(dtype=self.score.weight.dtype))).squeeze(-1)

    def apply_ffn_delta(
        self,
        x: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
        feedforward: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return FFN deltas for selected tokens and zeros for skipped tokens."""
        scores = self._scores(x)
        mask = self._selection_mask(scores)
        if not bool(mask.any()):
            return torch.zeros_like(x)

        selected = x[mask]
        selected_3d = selected.unsqueeze(0)
        delta = feedforward(norm(selected_3d)).squeeze(0)

        if self.use_soft_gate:
            selected_scores = scores[mask].unsqueeze(-1)
            delta = delta * (selected_scores / selected_scores.detach().clamp_min(1e-6))
        delta = delta.to(dtype=x.dtype)

        routed = torch.zeros_like(x)
        routed[mask] = delta
        return routed

    def apply_ffn_residual(
        self,
        x: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
        feedforward: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply residual FFN only to selected tokens."""
        return x + self.apply_ffn_delta(x, norm, feedforward)
