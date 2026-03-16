"""Tests for feed-forward network module."""

from __future__ import annotations

import torch

from src.models.feedforward.feedforward import FeedForward


class TestFeedForward:
    def test_output_shape(self) -> None:
        """Output should be (batch, seq_len, d_model)."""
        ff = FeedForward(d_model=64)
        x = torch.randn(2, 8, 64)
        out = ff(x)
        assert out.shape == (2, 8, 64)

    def test_output_dtype_float32(self) -> None:
        """Output should be float32 by default."""
        ff = FeedForward(d_model=128)
        x = torch.randn(1, 16, 128)
        out = ff(x)
        assert out.dtype == torch.float32

    def test_hidden_dimension_expansion(self) -> None:
        """Hidden dimension should be 4x the model dimension by default."""
        d_model = 64
        ff = FeedForward(d_model=d_model)
        # Check that the first linear layer has output size 4*d_model
        assert ff.linear1.out_features == 4 * d_model

    def test_hidden_dimension_custom(self) -> None:
        """Should support custom hidden dimension expansion."""
        d_model = 64
        ff = FeedForward(d_model=d_model, expansion_ratio=2)
        assert ff.linear1.out_features == 2 * d_model

    def test_gelu_activation(self) -> None:
        """Should use GELU activation function."""
        ff = FeedForward(d_model=64)
        assert isinstance(ff.activation, torch.nn.GELU)

    def test_single_token_sequence(self) -> None:
        """Should handle single-token sequences (T=1)."""
        ff = FeedForward(d_model=32)
        x = torch.randn(1, 1, 32)
        out = ff(x)
        assert out.shape == (1, 1, 32)

    def test_large_batch_size(self) -> None:
        """Should handle large batch sizes."""
        ff = FeedForward(d_model=32)
        x = torch.randn(16, 8, 32)
        out = ff(x)
        assert out.shape == (16, 8, 32)

    def test_different_d_models(self) -> None:
        """Should work with different model dimensions."""
        for d_model in [32, 64, 128, 256]:
            ff = FeedForward(d_model=d_model)
            x = torch.randn(1, 4, d_model)
            out = ff(x)
            assert out.shape == (1, 4, d_model)

    def test_output_changes_with_different_inputs(self) -> None:
        """Different inputs should produce different outputs."""
        ff = FeedForward(d_model=64)
        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        out1 = ff(x1)
        out2 = ff(x2)
        assert not torch.allclose(out1, out2)

    def test_linear_layers_exist(self) -> None:
        """Must have linear1 and linear2 layers."""
        ff = FeedForward(d_model=64)
        assert hasattr(ff, "linear1")
        assert hasattr(ff, "linear2")
        assert isinstance(ff.linear1, torch.nn.Linear)
        assert isinstance(ff.linear2, torch.nn.Linear)

    def test_dropout_applied_when_specified(self) -> None:
        """Dropout should be applied when dropout > 0."""
        ff = FeedForward(d_model=64, dropout=0.1)
        assert hasattr(ff, "dropout")
        assert isinstance(ff.dropout, torch.nn.Dropout)

    def test_dropout_not_applied_when_zero(self) -> None:
        """Dropout should not be applied when dropout=0."""
        ff = FeedForward(d_model=64, dropout=0.0)
        # Either should be Identity or not present
        if hasattr(ff, "dropout"):
            assert isinstance(ff.dropout, torch.nn.Identity)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the module."""
        ff = FeedForward(d_model=32)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_weight_initialization(self) -> None:
        """Linear layer weights should be initialized (not all zeros or ones)."""
        ff = FeedForward(d_model=64)
        # Check that weights are not all the same value
        assert not torch.allclose(ff.linear1.weight, ff.linear1.weight[0])
        assert not torch.allclose(ff.linear2.weight, ff.linear2.weight[0])
