"""MinHash LSH near-deduplication across spilled token arrays."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Warn when the total document count may stress RAM
_LARGE_CORPUS_WARN_THRESHOLD = 2_000_000


class _FilteredSpilledDocs:
    """Thin view over a _SpilledDocs restricted to a subset of document indices.

    Exposes the same interface as _SpilledDocs so the rest of the pipeline is
    transparent to whether dedup has been applied.
    """

    def __init__(self, spilled: Any, indices: np.ndarray) -> None:
        self._spilled = spilled
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._spilled[int(self._indices[idx])]

    @property
    def total_tokens(self) -> int:
        if len(self._indices) == 0:
            return 0
        return int(sum(len(self._spilled[int(i)]) for i in self._indices))

    @property
    def doc_lengths(self) -> np.ndarray:
        return np.array([len(self._spilled[int(i)]) for i in self._indices], dtype=np.int64)

    @property
    def dtype(self) -> Any:
        return self._spilled.dtype

    @property
    def offsets_bytes(self) -> int:
        return self._spilled.offsets_bytes

    @property
    def token_bytes(self) -> int:
        return self._spilled.token_bytes


def _minhash_for_doc(tokens: np.ndarray, shingle_size: int, num_perm: int) -> Any:
    """Compute a MinHash signature from contiguous token n-gram shingles."""
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    if len(tokens) < shingle_size:
        # Document too short: treat entire token sequence as one shingle
        m.update(tokens.tobytes())
    else:
        raw = tokens.tobytes()
        stride = tokens.itemsize * shingle_size
        for i in range(len(tokens) - shingle_size + 1):
            start = i * tokens.itemsize
            m.update(raw[start : start + stride])
    return m


def run_minhash_dedup(
    all_docs: dict[str, Any],
    jaccard_threshold: float = 0.8,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Near-deduplicate documents across all sources using MinHash LSH.

    Documents are processed in source order. The first occurrence within a
    near-duplicate cluster is kept; subsequent near-duplicates are dropped.

    Args:
        all_docs: mapping of source name → _SpilledDocs (or any Sequence[np.ndarray]).
        jaccard_threshold: minimum Jaccard similarity to consider two docs duplicates.
        num_perm: number of MinHash permutations (higher = more accurate but slower).
        shingle_size: number of consecutive tokens per shingle.

    Returns:
        A tuple of:
          - kept_indices_per_source: dict source_name → uint32 array of kept doc indices.
          - stats: summary dict with total/kept/dropped counts and dedup rate.
    """
    from datasketch import MinHashLSH

    total_docs = sum(len(spilled) for spilled in all_docs.values())
    if total_docs > _LARGE_CORPUS_WARN_THRESHOLD:
        logger.warning(
            "MinHash dedup: %d total docs exceeds %d — this may require significant RAM. "
            "Consider running dedup on a smaller slice first.",
            total_docs,
            _LARGE_CORPUS_WARN_THRESHOLD,
        )

    print(
        f"  [dedup] {total_docs:,} total docs across {len(all_docs)} source(s)"
        f"  threshold={jaccard_threshold}  num_perm={num_perm}  shingle_size={shingle_size}",
        flush=True,
    )

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    kept: dict[str, list[int]] = {name: [] for name in all_docs}

    for source_name, spilled in all_docs.items():
        n = len(spilled)
        for i in range(n):
            tokens = spilled[i]
            m = _minhash_for_doc(tokens, shingle_size, num_perm)
            candidates = lsh.query(m)
            if candidates:
                # Near-duplicate of an already-kept document — skip
                pass
            else:
                key = f"{source_name}:{i}"
                lsh.insert(key, m)
                kept[source_name].append(i)

        n_kept = len(kept[source_name])
        n_dropped = n - n_kept
        logger.info(
            "dedup: source=%s  in=%d  kept=%d  dropped=%d",
            source_name,
            n,
            n_kept,
            n_dropped,
        )

    total_kept = sum(len(v) for v in kept.values())
    total_dropped = total_docs - total_kept
    dedup_rate = total_dropped / max(1, total_docs)
    drop_counts = {name: len(all_docs[name]) - len(kept[name]) for name in all_docs}

    print(
        f"  [dedup] kept {total_kept:,}  dropped {total_dropped:,}"
        f"  dedup_rate={dedup_rate:.4f}",
        flush=True,
    )

    stats: dict[str, Any] = {
        "total_docs_in": total_docs,
        "total_docs_kept": total_kept,
        "total_docs_dropped": total_dropped,
        "dedup_rate": dedup_rate,
        "drop_counts_per_source": drop_counts,
    }
    kept_indices: dict[str, np.ndarray] = {
        name: np.array(kept[name], dtype=np.uint32) if kept[name] else np.array([], dtype=np.uint32)
        for name in all_docs
    }
    return kept_indices, stats
