"""Tests for token and positional embedding modules."""

from __future__ import annotations

import pytest
import torch

from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.position.learned_position import LearnedPositionEmbedding


class TestTokenEmbedding:
    def test_output_shape(self) -> None:
        emb = TokenEmbedding(vocab_size=256, d_model=64)
        x = torch.randint(0, 256, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 64)

    def test_output_dtype_float32(self) -> None:
        emb = TokenEmbedding(vocab_size=256, d_model=64)
        x = torch.randint(0, 256, (1, 8))
        out = emb(x)
        assert out.dtype == torch.float32

    def test_embedding_dim_matches_d_model(self) -> None:
        emb = TokenEmbedding(vocab_size=512, d_model=128)
        assert emb.embedding.embedding_dim == 128
        assert emb.embedding.num_embeddings == 512

    def test_weight_init_std(self) -> None:
        """Embedding weights should be initialised from N(0, 0.02)."""
        torch.manual_seed(0)
        emb = TokenEmbedding(vocab_size=1024, d_model=256)
        std = emb.embedding.weight.std().item()
        # Expect std close to 0.02 (within 50% for stochastic init on large tensor)
        assert 0.01 <= std <= 0.04

    def test_different_tokens_different_embeddings(self) -> None:
        emb = TokenEmbedding(vocab_size=256, d_model=64)
        x0 = torch.tensor([[0]])
        x1 = torch.tensor([[1]])
        assert not torch.allclose(emb(x0), emb(x1))

    def test_batch_independence(self) -> None:
        """Two identical sequences in a batch should yield identical embeddings."""
        emb = TokenEmbedding(vocab_size=256, d_model=64)
        seq = torch.randint(0, 256, (1, 12))
        batch = seq.repeat(3, 1)
        out = emb(batch)
        assert torch.allclose(out[0], out[1])
        assert torch.allclose(out[0], out[2])

    def test_single_token(self) -> None:
        emb = TokenEmbedding(vocab_size=128, d_model=32)
        x = torch.tensor([[42]])
        out = emb(x)
        assert out.shape == (1, 1, 32)


class TestLearnedPositionEmbedding:
    def test_output_shape(self) -> None:
        pos = LearnedPositionEmbedding(max_seq_len=512, d_model=64)
        x = torch.randint(0, 256, (2, 16))
        out = pos(x)
        assert out.shape == (1, 16, 64)  # (1, T, d_model) — broadcast-ready

    def test_seq_len_exceeds_max_raises(self) -> None:
        pos = LearnedPositionEmbedding(max_seq_len=8, d_model=32)
        x = torch.zeros(1, 9, dtype=torch.long)
        with pytest.raises(AssertionError):
            pos(x)

    def test_different_positions_different_embeddings(self) -> None:
        pos = LearnedPositionEmbedding(max_seq_len=16, d_model=32)
        x = torch.zeros(1, 4, dtype=torch.long)
        out = pos(x)  # shape (1, 4, 32)
        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_weight_init_std(self) -> None:
        """Position embeddings should be initialised from N(0, 0.02)."""
        torch.manual_seed(0)
        pos = LearnedPositionEmbedding(max_seq_len=512, d_model=256)
        std = pos.embedding.weight.std().item()
        assert 0.01 <= std <= 0.04

    def test_output_dtype_float32(self) -> None:
        pos = LearnedPositionEmbedding(max_seq_len=64, d_model=32)
        x = torch.zeros(2, 8, dtype=torch.long)
        out = pos(x)
        assert out.dtype == torch.float32

    def test_position_indices_from_zero(self) -> None:
        """The embedding should use positions 0..T-1."""
        pos = LearnedPositionEmbedding(max_seq_len=16, d_model=8)
        x = torch.zeros(1, 4, dtype=torch.long)
        out = pos(x)  # (1, 4, 8)
        # Manually look up position 0 and compare
        pos0 = pos.embedding(torch.tensor([0]))  # (1, 8)
        assert torch.allclose(out[0, 0], pos0[0])
