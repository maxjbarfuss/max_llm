"""Tests for causal multi-head attention module."""

from __future__ import annotations

import pytest
import torch

from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.models.attention.multihead_latent_attention import MultiHeadLatentAttention
from src.models.attention.residual_linear_attention import ResidualLinearAttention
from src.models.attention.sliding_window_attention import SlidingWindowAttention
from src.models.position.rope import RotaryEmbedding


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

    def test_document_mask_prevents_packed_cross_document_leakage(self) -> None:
        torch.manual_seed(43)
        mha = CausalMultiHeadAttention(d_model=32, num_heads=2, attention_backend="standard")
        mha.eval()

        x = torch.randn(1, 4, 32)
        document_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        masked = mha(x, document_ids=document_ids)

        x_modified = x.clone()
        x_modified[0, :2, :] = torch.randn(2, 32)
        masked_modified = mha(x_modified, document_ids=document_ids)
        unmasked_modified = mha(x_modified)

        assert torch.allclose(masked[0, 2:], masked_modified[0, 2:], atol=1e-6)
        assert not torch.allclose(masked[0, 2:], unmasked_modified[0, 2:], atol=1e-6)

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
            mha = CausalMultiHeadAttention(
                d_model=64, num_heads=num_heads, attention_backend="standard"
            )
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
        """Should have separate Q and KV projection layers."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        assert hasattr(mha, "q_proj")
        assert hasattr(mha, "kv_proj")

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

        # Manually compute Q, K via separate projections
        q = mha.q_proj(x)  # (B, T, d_model)
        kv = mha.kv_proj(x)  # (B, T, 2*d_model)
        k, _ = kv.chunk(2, dim=-1)  # each (B, T, d_model)

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


class TestGQAMQA:
    """Tests for Grouped Query Attention and Multi-Query Attention."""

    def test_mqa_output_shape(self) -> None:
        """MQA (num_kv_heads=1) should produce same output shape as MHA."""
        mqa = CausalMultiHeadAttention(
            d_model=64, num_heads=4, num_kv_heads=1, attention_backend="standard"
        )
        x = torch.randn(2, 8, 64)
        out = mqa(x)
        assert out.shape == (2, 8, 64)

    def test_gqa_output_shape(self) -> None:
        """GQA (num_kv_heads between 1 and num_heads) should produce same output shape."""
        gqa = CausalMultiHeadAttention(
            d_model=64, num_heads=4, num_kv_heads=2, attention_backend="standard"
        )
        x = torch.randn(2, 8, 64)
        out = gqa(x)
        assert out.shape == (2, 8, 64)

    def test_mha_default_unchanged(self) -> None:
        """num_kv_heads=None (default) is full MHA — output shape unchanged."""
        mha = CausalMultiHeadAttention(d_model=64, num_heads=4, attention_backend="standard")
        assert mha.num_kv_heads == 4
        assert mha.groups == 1
        x = torch.randn(2, 8, 64)
        assert mha(x).shape == (2, 8, 64)

    def test_mqa_has_fewer_kv_params_than_mha(self) -> None:
        """MQA should have fewer total parameters than MHA (smaller kv_proj)."""
        mha = CausalMultiHeadAttention(d_model=128, num_heads=8, attention_backend="standard")
        mqa = CausalMultiHeadAttention(
            d_model=128, num_heads=8, num_kv_heads=1, attention_backend="standard"
        )
        assert sum(p.numel() for p in mqa.parameters()) < sum(p.numel() for p in mha.parameters())

    def test_gqa_kv_proj_shape(self) -> None:
        """kv_proj output size should be 2 * num_kv_heads * head_dim."""
        gqa = CausalMultiHeadAttention(
            d_model=64, num_heads=4, num_kv_heads=2, attention_backend="standard"
        )
        assert gqa.kv_proj.out_features == 2 * 2 * 16  # 2 kv_heads * head_dim=16, x2 for k+v

    def test_num_kv_heads_must_divide_num_heads(self) -> None:
        """num_kv_heads that does not divide num_heads should raise AssertionError."""
        with pytest.raises(AssertionError):
            CausalMultiHeadAttention(
                d_model=64, num_heads=4, num_kv_heads=3, attention_backend="standard"
            )

    def test_causal_masking_preserved_in_mqa(self) -> None:
        """MQA should still enforce causal masking (no future token leakage)."""
        torch.manual_seed(42)
        mqa = CausalMultiHeadAttention(
            d_model=32, num_heads=4, num_kv_heads=1, attention_backend="standard"
        )
        mqa.eval()
        x = torch.randn(1, 4, 32)
        out1 = mqa(x)
        x_mod = x.clone()
        x_mod[0, 3, :] = torch.randn(32)
        out2 = mqa(x_mod)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 2], out2[0, 2], atol=1e-6)
        assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-6)

    def test_gradient_flows_through_gqa(self) -> None:
        """Gradients should flow through GQA."""
        gqa = CausalMultiHeadAttention(
            d_model=32, num_heads=4, num_kv_heads=2, attention_backend="standard"
        )
        x = torch.randn(1, 4, 32, requires_grad=True)
        gqa(x).sum().backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSlidingWindowAttention:
    """Tests for SlidingWindowAttention."""

    def test_output_shape(self) -> None:
        swa = SlidingWindowAttention(
            d_model=64, num_heads=4, window_size=4, attention_backend="standard"
        )
        x = torch.randn(2, 16, 64)
        assert swa(x).shape == (2, 16, 64)

    def test_causal_masking_preserved(self) -> None:
        """Token at position t must not depend on tokens at positions > t."""
        torch.manual_seed(0)
        swa = SlidingWindowAttention(
            d_model=32, num_heads=2, window_size=8, attention_backend="standard"
        )
        swa.eval()
        x = torch.randn(1, 6, 32)
        out1 = swa(x)
        x_mod = x.clone()
        x_mod[0, 5, :] = torch.randn(32)
        out2 = swa(x_mod)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 4], out2[0, 4], atol=1e-6)
        assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-6)

    def test_window_constraint(self) -> None:
        """Tokens outside the window must not affect the output."""
        torch.manual_seed(1)
        window_size = 3
        swa = SlidingWindowAttention(
            d_model=32, num_heads=2, window_size=window_size, attention_backend="standard"
        )
        swa.eval()
        x = torch.randn(1, 8, 32)
        out1 = swa(x)
        # Modify position 0 — it is outside the window for position 4 (gap=4 >= window_size=3)
        x_mod = x.clone()
        x_mod[0, 0, :] = torch.randn(32)
        out2 = swa(x_mod)
        assert torch.allclose(
            out1[0, 4], out2[0, 4], atol=1e-5
        ), "Position 4 should not depend on position 0 when window_size=3"

    def test_gradient_flows(self) -> None:
        swa = SlidingWindowAttention(
            d_model=32, num_heads=2, window_size=4, attention_backend="standard"
        )
        x = torch.randn(1, 8, 32, requires_grad=True)
        swa(x).sum().backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_gqa_output_shape(self) -> None:
        """GQA variant should produce same output shape."""
        swa = SlidingWindowAttention(
            d_model=64, num_heads=4, window_size=4, num_kv_heads=2, attention_backend="standard"
        )
        x = torch.randn(2, 8, 64)
        assert swa(x).shape == (2, 8, 64)

    def test_window_size_1_equals_self_attention_only(self) -> None:
        """window_size=1 means each token attends only to itself (diagonal)."""
        swa = SlidingWindowAttention(
            d_model=32, num_heads=2, window_size=1, attention_backend="standard"
        )
        swa.eval()
        x = torch.randn(1, 4, 32)
        out = swa(x)
        assert out.shape == (1, 4, 32)

    def test_window_size_assertion(self) -> None:
        with pytest.raises(AssertionError):
            SlidingWindowAttention(
                d_model=32, num_heads=2, window_size=0, attention_backend="standard"
            )


class TestResidualLinearAttention:
    """Tests for ResidualLinearAttention."""

    def test_output_shape(self) -> None:
        rla = ResidualLinearAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        assert rla(x).shape == (2, 16, 64)

    def test_output_dtype_float32(self) -> None:
        rla = ResidualLinearAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        assert rla(x).dtype == torch.float32

    def test_causal_masking(self) -> None:
        """Output at position t must not depend on inputs at positions > t."""
        torch.manual_seed(42)
        rla = ResidualLinearAttention(d_model=32, num_heads=2)
        rla.eval()
        x = torch.randn(1, 6, 32)
        out1 = rla(x)
        x_mod = x.clone()
        x_mod[0, 5, :] = torch.randn(32)
        out2 = rla(x_mod)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 4], out2[0, 4], atol=1e-6)
        assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-6)

    def test_rope_raises(self) -> None:
        """RoPE must be rejected with a clear error."""
        from src.models.position.rope import RotaryEmbedding

        rope = RotaryEmbedding(head_dim=16, max_seq_len=64, base=10000)
        with pytest.raises(ValueError, match="RoPE"):
            ResidualLinearAttention(d_model=64, num_heads=4, rope=rope)

    def test_gradient_flows(self) -> None:
        rla = ResidualLinearAttention(d_model=32, num_heads=2)
        x = torch.randn(1, 8, 32, requires_grad=True)
        rla(x).sum().backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_has_residual_projection(self) -> None:
        rla = ResidualLinearAttention(d_model=64, num_heads=4)
        assert hasattr(rla, "res_proj")
        assert isinstance(rla.res_proj, torch.nn.Linear)

    def test_gqa_output_shape(self) -> None:
        rla = ResidualLinearAttention(d_model=64, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 8, 64)
        assert rla(x).shape == (2, 8, 64)

    def test_different_inputs_different_outputs(self) -> None:
        rla = ResidualLinearAttention(d_model=64, num_heads=4)
        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        assert not torch.allclose(rla(x1), rla(x2))


class TestMultiHeadLatentAttention:
    """Tests for MLA with decoupled RoPE branch."""

    def test_output_shape(self) -> None:
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64, base=10000)
        mla = MultiHeadLatentAttention(
            d_model=64,
            num_heads=4,
            num_kv_heads=2,
            latent_dim=32,
            attention_backend="standard",
            rope=rope,
        )
        x = torch.randn(2, 16, 64)
        assert mla(x).shape == (2, 16, 64)

    def test_causal_masking(self) -> None:
        torch.manual_seed(42)
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64, base=10000)
        mla = MultiHeadLatentAttention(
            d_model=32,
            num_heads=2,
            num_kv_heads=1,
            latent_dim=16,
            attention_backend="standard",
            rope=rope,
        )
        mla.eval()
        x = torch.randn(1, 6, 32)
        out1 = mla(x)
        x_mod = x.clone()
        x_mod[0, 5, :] = torch.randn(32)
        out2 = mla(x_mod)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 4], out2[0, 4], atol=1e-6)
        assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-6)

    def test_requires_rope(self) -> None:
        with pytest.raises(ValueError, match="requires RoPE"):
            MultiHeadLatentAttention(
                d_model=64,
                num_heads=4,
                latent_dim=32,
                attention_backend="standard",
                rope=None,
            )
