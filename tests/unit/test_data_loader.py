"""Unit tests for the in-memory text data loader."""

import torch

from src.data.loader import TextChunkDataset, make_data_loaders
from src.tokenizer import CharTokenizer


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

    def _make_text(self, n_chars: int) -> str:
        """Produce n_chars of ASCII text (cycling through a–z)."""
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        return (alphabet * ((n_chars // len(alphabet)) + 1))[:n_chars]

    def test_returns_two_loaders(self):
        """Returns a (train_loader, val_loader) pair."""
        tok = CharTokenizer()
        text = self._make_text(200)
        train_loader, val_loader = make_data_loaders(
            text, tok, seq_len=10, batch_size=4
        )
        assert train_loader is not None
        assert val_loader is not None

    def test_train_larger_than_val(self):
        """Train split is larger than validation split."""
        tok = CharTokenizer()
        text = self._make_text(1000)
        train_loader, val_loader = make_data_loaders(
            text, tok, seq_len=10, batch_size=4, validation_split=0.1
        )
        assert len(train_loader.dataset) > len(val_loader.dataset)  # type: ignore[arg-type]

    def test_batch_shapes(self):
        """Batches from train_loader have shape (batch_size, seq_len)."""
        tok = CharTokenizer()
        text = self._make_text(500)
        train_loader, _ = make_data_loaders(
            text, tok, seq_len=16, batch_size=4
        )
        x, y = next(iter(train_loader))
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_val_batch_shapes(self):
        """Batches from val_loader have shape (batch_size, seq_len)."""
        tok = CharTokenizer()
        text = self._make_text(500)
        _, val_loader = make_data_loaders(
            text, tok, seq_len=8, batch_size=2
        )
        x, y = next(iter(val_loader))
        assert x.shape[1] == 8
        assert y.shape[1] == 8
