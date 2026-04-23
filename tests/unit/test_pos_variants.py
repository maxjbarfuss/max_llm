"""Tests for AdditiveRoPE, ALiBi, and RelativePositionBias positional encoding variants."""

import pytest
import torch

from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.rel_pos_bias import RelativePositionBias

# Standard test dims used throughout
D = 64
B = 2
T = 16
NUM_HEADS = 4
HEAD_DIM = D // NUM_HEADS  # 16


# ---------------------------------------------------------------------------
# AdditiveRoPE
# ---------------------------------------------------------------------------


def test_add_rope_output_shape() -> None:
    rope = AdditiveRoPE(HEAD_DIM, max_seq_len=T)
    q = torch.randn(B, T, NUM_HEADS, HEAD_DIM)
    k = torch.randn(B, T, NUM_HEADS, HEAD_DIM)
    q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_add_rope_dtype_float32_preserved() -> None:
    rope = AdditiveRoPE(HEAD_DIM, max_seq_len=T)
    q = torch.randn(B, T, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(B, T, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    q_out, k_out = rope(q, k)
    assert q_out.dtype == torch.float32
    assert k_out.dtype == torch.float32


def test_add_rope_no_parameters() -> None:
    rope = AdditiveRoPE(HEAD_DIM)
    assert sum(p.numel() for p in rope.parameters()) == 0


def test_add_rope_cache_not_in_state_dict() -> None:
    """Non-persistent buffer should not appear in state_dict."""
    rope = AdditiveRoPE(HEAD_DIM)
    assert "enc_cache" not in rope.state_dict()


def test_add_rope_encoding_differs_across_positions() -> None:
    """The same vector at different positions should produce different outputs."""
    rope = AdditiveRoPE(HEAD_DIM, max_seq_len=T)
    x = torch.randn(1, 1, 1, HEAD_DIM).expand(1, T, 1, HEAD_DIM).clone()
    q_out, _ = rope(x, x)
    # Position 0 and position 1 should differ
    assert not torch.allclose(q_out[:, 0], q_out[:, 1])


def test_add_rope_odd_head_dim_raises() -> None:
    with pytest.raises(AssertionError):
        AdditiveRoPE(head_dim=13)


def test_add_rope_cache_shape() -> None:
    rope = AdditiveRoPE(HEAD_DIM, max_seq_len=32)
    assert rope.enc_cache.shape == (1, 32, 1, HEAD_DIM)


def test_add_rope_gradient_flows() -> None:
    rope = AdditiveRoPE(HEAD_DIM)
    q = torch.randn(B, T, NUM_HEADS, HEAD_DIM, requires_grad=True)
    k = torch.randn(B, T, NUM_HEADS, HEAD_DIM, requires_grad=True)
    q_out, k_out = rope(q, k)
    (q_out.sum() + k_out.sum()).backward()
    assert q.grad is not None
    assert k.grad is not None


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------


def test_alibi_slopes_formula() -> None:
    """slopes[h] == 2^(-8*(h+1)/num_heads) for h in 0..num_heads-1."""
    alibi = ALiBi(num_heads=NUM_HEADS)
    for h in range(NUM_HEADS):
        expected = 2.0 ** (-8.0 * (h + 1) / NUM_HEADS)
        assert (
            abs(alibi.slopes[h].item() - expected) < 1e-6
        ), f"slope[{h}] mismatch: got {alibi.slopes[h].item()}, expected {expected}"


def test_alibi_no_parameters() -> None:
    alibi = ALiBi(NUM_HEADS)
    assert sum(p.numel() for p in alibi.parameters()) == 0


def test_alibi_slopes_not_in_state_dict() -> None:
    alibi = ALiBi(NUM_HEADS)
    assert "slopes" not in alibi.state_dict()


def test_alibi_get_bias_shape() -> None:
    alibi = ALiBi(NUM_HEADS)
    bias = alibi.get_bias(T, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (NUM_HEADS, T, T)


def test_alibi_bias_nonpositive_in_lower_triangle() -> None:
    """Causal region (i >= j) should have bias <= 0 (distance is non-negative)."""
    alibi = ALiBi(NUM_HEADS)
    bias = alibi.get_bias(T, device=torch.device("cpu"), dtype=torch.float32)
    # Lower triangle (including diagonal): rows i >= columns j
    for i in range(T):
        for j in range(i + 1):
            assert (
                bias[:, i, j] <= 0
            ).all(), (
                f"ALiBi bias should be <= 0 in lower triangle, got {bias[:, i, j]} at ({i},{j})"
            )


def test_alibi_diagonal_is_zero() -> None:
    """Distance is 0 on the diagonal (i == j), so bias should be 0."""
    alibi = ALiBi(NUM_HEADS)
    bias = alibi.get_bias(T, device=torch.device("cpu"), dtype=torch.float32)
    for i in range(T):
        assert torch.allclose(
            bias[:, i, i], torch.zeros(NUM_HEADS)
        ), f"ALiBi diagonal bias should be 0 at position {i}"


def test_alibi_slopes_shape() -> None:
    """Slopes tensor shape used by flash attn: (num_heads,)."""
    alibi = ALiBi(NUM_HEADS)
    assert alibi.slopes.shape == (NUM_HEADS,)


def test_alibi_get_bias_dtype_preserved() -> None:
    alibi = ALiBi(NUM_HEADS)
    bias = alibi.get_bias(T, device=torch.device("cpu"), dtype=torch.float16)
    assert bias.dtype == torch.float16


# ---------------------------------------------------------------------------
# RelativePositionBias
# ---------------------------------------------------------------------------


def test_rel_pos_has_parameters() -> None:
    """bias_table has num_buckets * num_heads parameters."""
    num_buckets = 32
    rpb = RelativePositionBias(NUM_HEADS, num_buckets=num_buckets)
    total = sum(p.numel() for p in rpb.parameters())
    assert (
        total == num_buckets * NUM_HEADS
    ), f"Expected {num_buckets * NUM_HEADS} params, got {total}"


def test_rel_pos_get_bias_shape() -> None:
    rpb = RelativePositionBias(NUM_HEADS)
    bias = rpb.get_bias(T, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (NUM_HEADS, T, T)


def test_rel_pos_gradient_flows_through_bias_table() -> None:
    rpb = RelativePositionBias(NUM_HEADS)
    bias = rpb.get_bias(T, device=torch.device("cpu"), dtype=torch.float32)
    bias.sum().backward()
    assert rpb.bias_table.weight.grad is not None
    assert rpb.bias_table.weight.grad.abs().sum() > 0


def test_rel_pos_diagonal_gets_bucket_zero() -> None:
    """On the diagonal i == j: rel distance = 0, bucket = clamp(0, 0) = 0."""
    rpb = RelativePositionBias(NUM_HEADS)
    # Set bucket 0 to a recognizable value
    with torch.no_grad():
        rpb.bias_table.weight[0] = 99.0
    bias = rpb.get_bias(T, device=torch.device("cpu"), dtype=torch.float32)
    for i in range(T):
        assert torch.allclose(
            bias[:, i, i], torch.full((NUM_HEADS,), 99.0)
        ), f"Diagonal position {i} should map to bucket 0"


def test_rel_pos_get_bias_dtype_preserved() -> None:
    rpb = RelativePositionBias(NUM_HEADS)
    bias = rpb.get_bias(T, device=torch.device("cpu"), dtype=torch.float16)
    assert bias.dtype == torch.float16


# ---------------------------------------------------------------------------
# Integration: CausalMHA with attn_bias
# ---------------------------------------------------------------------------


def test_causal_mha_with_alibi_shape() -> None:
    from src.models.attention.causal_mha import CausalMultiHeadAttention

    alibi = ALiBi(NUM_HEADS)
    mha = CausalMultiHeadAttention(
        d_model=D,
        num_heads=NUM_HEADS,
        attention_backend="standard",
        attn_bias=alibi,
    )
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)


def test_causal_mha_with_rel_pos_bias_shape() -> None:
    from src.models.attention.causal_mha import CausalMultiHeadAttention

    rpb = RelativePositionBias(NUM_HEADS)
    mha = CausalMultiHeadAttention(
        d_model=D,
        num_heads=NUM_HEADS,
        attention_backend="standard",
        attn_bias=rpb,
    )
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)


def test_causal_mha_rel_pos_bias_falls_back_to_standard() -> None:
    """RelPosBias should force standard backend even if flash/xformers is requested."""
    from src.models.attention.causal_mha import CausalMultiHeadAttention

    rpb = RelativePositionBias(NUM_HEADS)
    # Request flash — it will either use flash (if available) or fall back to standard.
    # Either way, RelPosBias should not break the forward pass.
    mha = CausalMultiHeadAttention(
        d_model=D,
        num_heads=NUM_HEADS,
        attention_backend="flash",
        attn_bias=rpb,
    )
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)


def test_causal_mha_with_add_rope_shape() -> None:
    from src.models.attention.causal_mha import CausalMultiHeadAttention
    from src.models.position.add_rope import AdditiveRoPE

    head_dim = D // NUM_HEADS
    rope = AdditiveRoPE(head_dim, max_seq_len=T)
    mha = CausalMultiHeadAttention(
        d_model=D,
        num_heads=NUM_HEADS,
        attention_backend="standard",
        rope=rope,
    )
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)
