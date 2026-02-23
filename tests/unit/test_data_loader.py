"""Unit tests for the in-memory token data loader."""

from pathlib import Path

import pytest
import torch

from src.data.loader import TextChunkDataset, load_corpus_text, make_data_loaders


def _tokens(n: int) -> list[int]:
    """Produce n sequential token IDs (mod 128 to stay in vocab)."""
    return [i % 128 for i in range(n)]


class TestTextChunkDataset:
    """TextChunkDataset chunk-splitting behaviour."""

    def test_length(self):
        """Number of chunks is (len(tokens) - 1) // seq_len."""
        assert len(TextChunkDataset(_tokens(100), seq_len=10)) == 9
        assert len(TextChunkDataset(_tokens(50), seq_len=5)) == 9
        assert len(TextChunkDataset(_tokens(11), seq_len=10)) == 1

    def test_item_shapes(self):
        """Each item is a pair of (seq_len,) tensors."""
        ds = TextChunkDataset(_tokens(50), seq_len=8)
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_item_dtype(self):
        """Token tensors are long (int64)."""
        ds = TextChunkDataset(_tokens(20), seq_len=5)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_target_is_shifted(self):
        """y is x shifted right by one position."""
        tokens = list(range(20))
        ds = TextChunkDataset(tokens, seq_len=5)
        x, y = ds[0]
        # All but the last target token overlap with input
        assert torch.equal(y[:-1], x[1:])
        # The final target token is the first token of the next chunk
        assert y[-1].item() == tokens[5]

    def test_non_overlapping_chunks(self):
        """Consecutive chunks are non-overlapping (stride = seq_len)."""
        tokens = list(range(30))
        ds = TextChunkDataset(tokens, seq_len=5)
        x0, _ = ds[0]
        x1, _ = ds[1]
        assert x0[-1].item() + 1 == x1[0].item()

    def test_too_short_returns_empty(self):
        """Text shorter than seq_len + 1 yields an empty dataset."""
        assert len(TextChunkDataset(_tokens(5), seq_len=10)) == 0
        assert len(TextChunkDataset(_tokens(10), seq_len=10)) == 0

    def test_empty_tokens_returns_empty(self):
        """Empty token list yields an empty dataset."""
        assert len(TextChunkDataset([], seq_len=8)) == 0


class TestMakeDataLoaders:
    """make_data_loaders split and batch behaviour."""

    def test_returns_two_loaders(self):
        """Returns a (train_loader, val_loader) pair."""
        train_loader, val_loader = make_data_loaders(_tokens(200), seq_len=10, batch_size=4)
        assert train_loader is not None
        assert val_loader is not None

    def test_train_larger_than_val(self):
        """Train split holds at least 85% of total chunks with a 0.1 validation split."""
        train_loader, val_loader = make_data_loaders(
            _tokens(1000), seq_len=10, batch_size=4, validation_split=0.1
        )
        n_train = len(train_loader.dataset)  # type: ignore[arg-type]
        n_val = len(val_loader.dataset)  # type: ignore[arg-type]
        assert n_train > n_val
        assert n_train / (n_train + n_val) > 0.85

    def test_batch_shapes(self):
        """Batches from train_loader have shape (batch_size, seq_len)."""
        train_loader, _ = make_data_loaders(_tokens(500), seq_len=16, batch_size=4)
        x, y = next(iter(train_loader))
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_val_batch_shapes(self):
        """Batches from val_loader have shape (batch_size, seq_len)."""
        _, val_loader = make_data_loaders(_tokens(500), seq_len=8, batch_size=2)
        x, y = next(iter(val_loader))
        assert x.shape[1] == 8
        assert y.shape[1] == 8

    def test_same_seed_produces_same_order(self):
        """Identical seeds yield identical first-batch order."""
        train1, _ = make_data_loaders(_tokens(200), seq_len=10, batch_size=4, seed=42)
        train2, _ = make_data_loaders(_tokens(200), seq_len=10, batch_size=4, seed=42)
        x1, _ = next(iter(train1))
        x2, _ = next(iter(train2))
        assert torch.equal(x1, x2)

    def test_different_seeds_produce_different_order(self):
        """Different seeds yield different first-batch orders."""
        train1, _ = make_data_loaders(_tokens(200), seq_len=10, batch_size=4, seed=42)
        train2, _ = make_data_loaders(_tokens(200), seq_len=10, batch_size=4, seed=99)
        x1, _ = next(iter(train1))
        x2, _ = next(iter(train2))
        assert not torch.equal(x1, x2)

    def test_validation_split_zero(self):
        """validation_split=0.0 yields a full train set and an empty val dataset."""
        train_loader, val_loader = make_data_loaders(
            _tokens(100), seq_len=10, batch_size=4, validation_split=0.0
        )
        assert len(train_loader.dataset) > 0  # type: ignore[arg-type]
        assert len(val_loader.dataset) == 0  # type: ignore[arg-type]

    def test_val_loader_does_not_shuffle(self):
        """Iterating val_loader twice yields identical batch order."""
        _, val_loader = make_data_loaders(
            _tokens(200), seq_len=10, batch_size=4, validation_split=0.3
        )
        first_pass = [x.clone() for x, _ in val_loader]
        second_pass = [x.clone() for x, _ in val_loader]
        assert len(first_pass) > 0, "val_loader should not be empty"
        for b1, b2 in zip(first_pass, second_pass, strict=True):
            assert torch.equal(b1, b2)

    def test_validation_split_absolute_token_count(self):
        """Integer validation_split reserves that many tokens for validation."""
        tokens = _tokens(100)
        train_loader, val_loader = make_data_loaders(
            tokens, seq_len=10, batch_size=4, validation_split=20
        )
        expected_train = (80 - 1) // 10
        expected_val = (20 - 1) // 10
        assert len(train_loader.dataset) == expected_train  # type: ignore[arg-type]
        assert len(val_loader.dataset) == expected_val  # type: ignore[arg-type]


class TestLoadCorpusText:
    """load_corpus_text file and directory loading behavior."""

    def test_loads_single_file(self, tmp_path: Path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello\nworld", encoding="utf-8")
        assert load_corpus_text(str(corpus)) == "hello\nworld"

    def test_loads_directory_recursively(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "b.txt").write_text("beta", encoding="utf-8")

        text = load_corpus_text(str(tmp_path))
        assert "alpha" in text
        assert "beta" in text

    def test_missing_path_raises(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_corpus_text(str(missing))
