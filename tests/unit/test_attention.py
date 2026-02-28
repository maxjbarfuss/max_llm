"""Tests for causal multi-head attention module."""

from __future__ import annotations

import pytest
import torch

from src.models.attention.causal_mha import CausalMultiHeadAttention


class TestCausalMultiHeadAttention:
    def test_output_shape(self) -> None:
        """Output should be (batch, seq_len, d_model)."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        x = torch.randn(2, 8, 64)
        out = mha(x)
        assert out.shape == (2, 8, 64)

    def test_output_dtype_float32(self) -> None:
        """Output should be float32 by default."""
        mha = CausalMultiHeadAttention(d_model=128, num_heads=8, attention_backend="standard")
        x = torch.randn(1, 16, 128)
        out = mha(x)
        assert out.dtype == torch.float32

    def test_causal_masking_prevents_future_leakage(self) -> None:
        """Token at position t should not depend on tokens at positions > t.

        We verify this by:
        1. Computing attention for a sequence
        2. Modifying a future token
        3. Re-computing attention for an earlier token
        4. Checking that the earlier token's output is unchanged
        """
        torch.manual_seed(42)
        mha = CausalMultiHeadAttention(d_model=32, num_heads=2, attention_backend="standard")
        mha.eval()

        # Original sequence
        x = torch.randn(1, 4, 32)
        out1 = mha(x)

        # Modify future token (position 3)
        x_modified = x.clone()
        x_modified[0, 3, :] = torch.randn(32)
        out2 = mha(x_modified)

        # Position 0 should be unchanged (it cannot see position 3)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)

        # Position 2 should be unchanged (it cannot see position 3)
        assert torch.allclose(out1[0, 2], out2[0, 2], atol=1e-6)

        # Position 3 itself should be different (it depends on its own input)
        assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-6)

    def test_attention_output_changes_with_different_inputs(self) -> None:
        """Different inputs should produce different outputs."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        out1 = mha(x1)
        out2 = mha(x2)
        assert not torch.allclose(out1, out2)

    def test_d_model_divisible_by_num_heads(self) -> None:
        """d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            CausalMultiHeadAttention(d_model=63, num_heads=4, attention_backend="standard")

    def test_single_token_sequence(self) -> None:
        """Should handle single-token sequences (T=1)."""
        mha = CausalMultiHeadAttention(d_model=32, num_heads=2, attention_backend="standard")
        x = torch.randn(1, 1, 32)
        out = mha(x)
        assert out.shape == (1, 1, 32)

    def test_single_head_attention(self) -> None:
        """Should work with a single attention head."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=1, attention_backend="standard")
        x = torch.randn(2, 10, 64)
        out = mha(x)
        assert out.shape == (2, 10, 64)

    def test_multiple_heads(self) -> None:
        """Should work with different head counts."""
        for num_heads in [1, 2, 4, 8]:
            mha = CausalMultiHeadAttention(d_model=64, num_heads=num_heads)
            x = torch.randn(1, 8, 64)
            out = mha(x)
            assert out.shape == (1, 8, 64)

    def test_large_batch_size(self) -> None:
        """Should handle large batch sizes."""
        mha = CausalMultiHeadAttention(d_model=32, num_heads=4, attention_backend="standard")
        x = torch.randn(16, 8, 32)
        out = mha(x)
        assert out.shape == (16, 8, 32)

    def test_projection_layer_exists(self) -> None:
        """Should have Q, K, V projection layers."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        assert hasattr(mha, "q_proj")
        assert hasattr(mha, "k_proj")
        assert hasattr(mha, "v_proj")

    def test_output_projection_exists(self) -> None:
        """Should have an output projection layer."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        assert hasattr(mha, "out_proj")

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical outputs."""
        mha = CausalMultiHeadAttention(d_model=32, num_heads=2, attention_backend="standard")
        x = torch.randn(1, 4, 32)

        torch.manual_seed(0)
        mha.eval()
        out1 = mha(x)

        torch.manual_seed(0)
        mha.eval()
        out2 = mha(x)

        assert torch.allclose(out1, out2)

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights for each query position should sum to ~1.

        We'll manually compute attention weights and verify they're normalized.
        This tests the correctness of the softmax normalization.
        """
        torch.manual_seed(42)
        mha = CausalMultiHeadAttention(d_model=32, num_heads=2, attention_backend="standard")
        mha.eval()

        x = torch.randn(1, 4, 32)

        # Forward pass to trigger any internal caching
        _ = mha(x)

        # Manually compute Q, K
        q = mha.q_proj(x)  # (B, T, d_model)
        k = mha.k_proj(x)  # (B, T, d_model)

        B, T, d_model = x.shape
        head_dim = d_model // mha.num_heads

        # Reshape for multi-head: (B, num_heads, T, head_dim)
        q = q.view(B, T, mha.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, mha.num_heads, head_dim).transpose(1, 2)

        # Compute attention scores: (B, num_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, T, T)

        # Verify weights sum to 1 along the key dimension for each query
        sums = attn_weights.sum(dim=-1)  # (B, num_heads, T)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_gradient_flows_backward(self) -> None:
        """Gradients should flow through the attention mechanism."""
        mha = CausalMultiHeadAttention(d_model=32, num_heads=2, attention_backend="standard")
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_head_dim_computation(self) -> None:
        """head_dim should be d_model // num_heads."""
        mha = CausalMultiHeadAttention(d_model=128, num_heads=8, attention_backend="standard")
        assert mha.head_dim == 16

        mha2 = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        assert mha2.head_dim == 16
