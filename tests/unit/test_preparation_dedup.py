"""Unit tests for MinHash LSH near-deduplication."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preparation.pipeline._dedup import (
    _FilteredSpilledDocs,
    _minhash_for_doc,
    run_minhash_dedup,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_spilled(token_lists: list[list[int]]) -> _FakeSpilled:
    """Minimal _SpilledDocs-compatible stub backed by plain Python lists."""
    return _FakeSpilled(token_lists)


class _FakeSpilled:
    def __init__(self, token_lists: list[list[int]]) -> None:
        self._docs = [np.array(t, dtype=np.uint16) for t in token_lists]

    def __len__(self) -> int:
        return len(self._docs)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._docs[idx]

    @property
    def total_tokens(self) -> int:
        return sum(len(d) for d in self._docs)

    @property
    def doc_lengths(self) -> np.ndarray:
        return np.array([len(d) for d in self._docs], dtype=np.int64)

    @property
    def offsets_bytes(self) -> int:
        return 0

    @property
    def token_bytes(self) -> int:
        return 0

    @property
    def dtype(self):  # type: ignore[override]
        return np.dtype(np.uint16)


# ── _minhash_for_doc ─────────────────────────────────────────────────────────


def test_minhash_same_doc_equal():
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    m1 = _minhash_for_doc(tokens, shingle_size=3, num_perm=64)
    m2 = _minhash_for_doc(tokens, shingle_size=3, num_perm=64)
    jaccard = m1.jaccard(m2)
    assert jaccard == pytest.approx(1.0, abs=0.01)


def test_minhash_different_docs_low_jaccard():
    t1 = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    t2 = np.array([100, 200, 300, 400, 500], dtype=np.uint16)
    m1 = _minhash_for_doc(t1, shingle_size=3, num_perm=128)
    m2 = _minhash_for_doc(t2, shingle_size=3, num_perm=128)
    assert m1.jaccard(m2) < 0.3


def test_minhash_short_doc_falls_back():
    """Documents shorter than shingle_size should not raise and produce a valid signature."""
    tokens = np.array([1, 2], dtype=np.uint16)
    m = _minhash_for_doc(tokens, shingle_size=5, num_perm=64)
    assert m is not None


# ── run_minhash_dedup ─────────────────────────────────────────────────────────


def test_dedup_keeps_unique_docs():
    spilled = _make_spilled([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    kept, stats = run_minhash_dedup({"src": spilled}, jaccard_threshold=0.8, num_perm=64)
    assert len(kept["src"]) == 3
    assert stats["total_docs_kept"] == 3
    assert stats["total_docs_dropped"] == 0
    assert stats["dedup_rate"] == pytest.approx(0.0)


def test_dedup_removes_identical_duplicate():
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    spilled = _make_spilled([tokens, tokens, [1, 2, 3, 4, 5, 6, 7, 8]])
    kept, stats = run_minhash_dedup({"src": spilled}, jaccard_threshold=0.8, num_perm=128)
    # The first occurrence is kept; the second identical doc is dropped
    assert stats["total_docs_dropped"] >= 1
    assert stats["total_docs_kept"] <= 2


def test_dedup_cross_source():
    """Near-duplicate appearing in a second source should be dropped."""
    tokens = list(range(1, 21))
    src_a = _make_spilled([tokens])
    src_b = _make_spilled([tokens])  # exact duplicate
    kept, stats = run_minhash_dedup(
        {"a": src_a, "b": src_b},
        jaccard_threshold=0.8,
        num_perm=128,
    )
    total_kept = len(kept["a"]) + len(kept["b"])
    # One of the two identical docs should be dropped
    assert total_kept == 1
    assert stats["total_docs_dropped"] == 1


def test_dedup_empty_source():
    spilled = _make_spilled([])
    kept, stats = run_minhash_dedup({"empty": spilled}, jaccard_threshold=0.8, num_perm=64)
    assert len(kept["empty"]) == 0
    assert stats["total_docs_kept"] == 0


def test_dedup_returns_sorted_uint32_indices():
    spilled = _make_spilled([[1, 2], [3, 4], [5, 6]])
    kept, _ = run_minhash_dedup({"s": spilled}, jaccard_threshold=0.8, num_perm=64)
    arr = kept["s"]
    assert arr.dtype == np.uint32
    assert list(arr) == sorted(arr.tolist())


# ── _FilteredSpilledDocs ──────────────────────────────────────────────────────


def test_filtered_spilled_docs_len_and_getitem():
    spilled = _make_spilled([[10, 20], [30, 40], [50, 60], [70, 80]])
    indices = np.array([0, 2], dtype=np.uint32)
    filtered = _FilteredSpilledDocs(spilled, indices)
    assert len(filtered) == 2
    np.testing.assert_array_equal(filtered[0], [10, 20])
    np.testing.assert_array_equal(filtered[1], [50, 60])


def test_filtered_spilled_docs_total_tokens():
    spilled = _make_spilled([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    indices = np.array([0, 2], dtype=np.uint32)  # skip doc at index 1
    filtered = _FilteredSpilledDocs(spilled, indices)
    assert filtered.total_tokens == 7  # 3 + 4


def test_filtered_spilled_docs_empty_indices():
    spilled = _make_spilled([[1, 2], [3, 4]])
    filtered = _FilteredSpilledDocs(spilled, np.array([], dtype=np.uint32))
    assert len(filtered) == 0
    assert filtered.total_tokens == 0
