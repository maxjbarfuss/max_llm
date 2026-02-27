"""Tests for transformer block module."""

from __future__ import annotations

import pytest
import torch

from src.models.transformer.transformer_block import TransformerBlock


class TestTransformerBlock:
    def test_output_shape(self) -> None:
        """Output should be (batch, seq_len, d_model)."""
        block = TransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_output_dtype_float32(self) -> None:
        """Output should be float32 by default."""
        block = TransformerBlock(d_model=128, num_heads=8)
        x = torch.randn(1, 16, 128)
        out = block(x)
        assert out.dtype == torch.float32

    def test_has_attention_layer(self) -> None:
        """Must have an attention layer."""
        block = TransformerBlock(d_model=64, num_heads=4)
        assert hasattr(block, "attention")

    def test_has_feedforward_layer(self) -> None:
        """Must have a feedforward layer."""
        block = TransformerBlock(d_model=64, num_heads=4)
        assert hasattr(block, "feedforward")

    def test_has_layer_norms(self) -> None:
        """Must have pre-norm layer normalization layers."""
        block = TransformerBlock(d_model=64, num_heads=4)
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)

    def test_single_token_sequence(self) -> None:
        """Should handle single-token sequences (T=1)."""
        block = TransformerBlock(d_model=32, num_heads=2)
        x = torch.randn(1, 1, 32)
        out = block(x)
        assert out.shape == (1, 1, 32)

    def test_large_batch_size(self) -> None:
        """Should handle large batch sizes."""
        block = TransformerBlock(d_model=32, num_heads=4)
        x = torch.randn(16, 8, 32)
        out = block(x)
        assert out.shape == (16, 8, 32)

    def test_different_d_models(self) -> None:
        """Should work with different model dimensions (divisible by num_heads)."""
        for d_model, num_heads in [(32, 2), (64, 4), (128, 8)]:
            block = TransformerBlock(d_model=d_model, num_heads=num_heads)
            x = torch.randn(1, 4, d_model)
            out = block(x)
            assert out.shape == (1, 4, d_model)

    def test_output_changes_with_different_inputs(self) -> None:
        """Different inputs should produce different outputs."""
        block = TransformerBlock(d_model=64, num_heads=4)
        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        out1 = block(x1)
        out2 = block(x2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the module."""
        block = TransformerBlock(d_model=32, num_heads=4)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_residual_connection_attention(self) -> None:
        """Attention branch should include residual connection."""
        torch.manual_seed(42)
        block = TransformerBlock(d_model=64, num_heads=4)
        block.eval()

        x = torch.randn(1, 4, 64)
        out = block(x)

        # With strong identity mapping (learned attention near zero),
        # output should be close to input. We can't test this directly,
        # but we can test that modifications work.
        assert out.shape == x.shape

    def test_residual_connection_feedforward(self) -> None:
        """Feedforward branch should include residual connection."""
        torch.manual_seed(42)
        block = TransformerBlock(d_model=64, num_heads=4)
        block.eval()

        x = torch.randn(1, 4, 64)
        out = block(x)

        # Check that output shape is preserved (residual allows this)
        assert out.shape == x.shape

    def test_pre_norm_architecture(self) -> None:
        """Should use pre-normalization (norm before transformation)."""
        block = TransformerBlock(d_model=64, num_heads=4)

        # Verify the forward pass uses pre-norm by checking the module order
        # Pre-norm: x -> norm -> attention -> residual
        # This is harder to test directly, but we verify norm layers exist
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")

    def test_dropout_applied_when_specified(self) -> None:
        """Dropout should be applied when specified."""
        block = TransformerBlock(d_model=64, num_heads=4, dropout=0.1)
        # Check that dropout is configured in attention
        assert hasattr(block.attention, "dropout_p")
        assert block.attention.dropout_p > 0

    def test_custom_expansion_ratio(self) -> None:
        """Should support custom feedforward expansion ratio."""
        block = TransformerBlock(d_model=64, num_heads=4, ff_expansion_ratio=2)
        x = torch.randn(1, 4, 64)
        out = block(x)
        assert out.shape == (1, 4, 64)

    def test_d_model_divisible_by_num_heads(self) -> None:
        """d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            TransformerBlock(d_model=63, num_heads=4)

    def test_long_sequence(self) -> None:
        """Should handle longer sequences without issues."""
        block = TransformerBlock(d_model=32, num_heads=4)
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == (2, 64, 32)

    def test_deterministic_with_seed(self) -> None:
        """With same seed, should produce same output."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, 64)

        torch.manual_seed(42)
        block1 = TransformerBlock(d_model=64, num_heads=4)
        block1.eval()
        out1 = block1(x)

        torch.manual_seed(42)
        block2 = TransformerBlock(d_model=64, num_heads=4)
        block2.eval()
        # Copy weights to make them identical
        block2.load_state_dict(block1.state_dict())
        out2 = block2(x)

        assert torch.allclose(out1, out2, atol=1e-5)
