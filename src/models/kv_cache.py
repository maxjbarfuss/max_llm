"""KV-cache for autoregressive generation.

LayerKVCache: pre-allocated K/V buffers for a single attention layer.
ModelKVCache: container holding one LayerKVCache per transformer layer.

Usage (inference loop):
    cache = model.make_kv_cache(max_seq_len=512, batch_size=1)

    # Prefill (process full prompt at once):
    model(prompt_ids, kv_caches=cache)

    # Generate token-by-token:
    for _ in range(max_new_tokens):
        logits = model(next_token_ids, kv_caches=cache)
        next_token = sample(logits[0, -1])
        tokens.append(next_token)

    cache.reset()  # reuse for next sequence
"""

from __future__ import annotations

import torch


class LayerKVCache:
    """Pre-allocated K/V cache for a single attention layer.

    Buffers are allocated once at max_seq_len and filled in-place via
    slice-copy to avoid per-step allocation.  `length` tracks how many
    tokens have been written.

    Raises RuntimeError if an update would exceed max_seq_len.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self._k = torch.zeros(
            batch_size, max_seq_len, num_kv_heads, head_dim, device=dev, dtype=dtype
        )
        self._v = torch.zeros(
            batch_size, max_seq_len, num_kv_heads, head_dim, device=dev, dtype=dtype
        )
        self._length = 0
        self._max = max_seq_len

    @property
    def length(self) -> int:
        """Number of tokens currently cached."""
        return self._length

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tokens and return the full cached (K, V).

        Args:
            new_k: (batch, T_new, num_kv_heads, head_dim)
            new_v: (batch, T_new, num_kv_heads, head_dim)

        Returns:
            k_all: (batch, length+T_new, num_kv_heads, head_dim)
            v_all: same shape
        """
        T = new_k.shape[1]
        end = self._length + T
        if end > self._max:
            raise RuntimeError(
                f"KV-cache overflow: tried to write to position {end} but max_seq_len={self._max}. "
                "Increase max_seq_len or call cache.reset()."
            )
        self._k[:, self._length : end].copy_(new_k)
        self._v[:, self._length : end].copy_(new_v)
        self._length = end
        return self._k[:, :end], self._v[:, :end]

    def reset(self) -> None:
        """Clear the cache (does not zero the buffers; length reset suffices)."""
        self._length = 0


class ModelKVCache:
    """Container holding one LayerKVCache per transformer layer.

    Indexing: cache[i] returns the LayerKVCache for layer i, or None for
    layers that do not support KV-caching (e.g. RLA).
    """

    def __init__(self, layers: list[LayerKVCache | None]) -> None:
        self._layers = layers

    def __getitem__(self, i: int) -> LayerKVCache | None:
        return self._layers[i]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    @property
    def length(self) -> int:
        """Tokens cached so far (requires all non-None layers to agree)."""
        expected: int | None = None
        for i, c in enumerate(self._layers):
            if c is None:
                continue
            if expected is None:
                expected = c.length
                continue
            if c.length != expected:
                raise RuntimeError(
                    "Inconsistent KV-cache lengths across layers: "
                    f"layer 0 has length {expected}, layer {i} has length {c.length}."
                )
        if expected is not None:
            return expected
        return 0

    def reset(self) -> None:
        """Reset all layer caches."""
        for c in self._layers:
            if c is not None:
                c.reset()
