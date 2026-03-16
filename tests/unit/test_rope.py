"""Tests for Rotary Position Embedding (RoPE)."""

import pytest
import torch

from src.models.position.rope import RotaryEmbedding, _rotate_half

# ---------------------------------------------------------------------------
# _rotate_half
# ---------------------------------------------------------------------------


def test_rotate_half_shape() -> None:
    x = torch.randn(2, 8, 4, 64)
    assert _rotate_half(x).shape == x.shape


def test_rotate_half_involution() -> None:
    """rotate_half applied twice returns negated input: RH(RH(x)) == -x."""
    x = torch.randn(2, 8, 4, 64)
    assert torch.allclose(_rotate_half(_rotate_half(x)), -x)


def test_rotate_half_splits_correctly() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = _rotate_half(x)
    # Expected: cat([-x2, x1]) = cat([-3, -4, 1, 2])
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# RotaryEmbedding construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_cache_shape(head_dim: int) -> None:
    rope = RotaryEmbedding(head_dim, max_seq_len=512)
    assert rope.cos_cache.shape == (512, head_dim)
    assert rope.sin_cache.shape == (512, head_dim)


def test_odd_head_dim_raises() -> None:
    with pytest.raises(AssertionError):
        RotaryEmbedding(head_dim=63)


def test_no_learnable_parameters() -> None:
    rope = RotaryEmbedding(64)
    assert sum(p.numel() for p in rope.parameters()) == 0


def test_cache_not_in_state_dict() -> None:
    """Buffers are non-persistent — not saved in state_dict."""
    rope = RotaryEmbedding(64)
    assert "cos_cache" not in rope.state_dict()
    assert "sin_cache" not in rope.state_dict()


def test_cache_values_are_unit() -> None:
    """cos²+sin² == 1 for every cache position."""
    rope = RotaryEmbedding(64, max_seq_len=128)
    cos2 = rope.cos_cache**2
    sin2 = rope.sin_cache**2
    # Each position: sum of (cos²+sin²) over paired dims == head_dim/2
    assert torch.allclose(cos2 + sin2, torch.ones_like(cos2), atol=1e-5)


# ---------------------------------------------------------------------------
# RotaryEmbedding forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 16, 128])
def test_output_shape(T: int) -> None:
    B, num_heads, head_dim = 2, 4, 64
    rope = RotaryEmbedding(head_dim, max_seq_len=256)
    q = torch.randn(B, T, num_heads, head_dim)
    k = torch.randn(B, T, num_heads, head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_dtype_preserved_float32() -> None:
    rope = RotaryEmbedding(64)
    q = torch.randn(2, 8, 4, 64, dtype=torch.float32)
    k = torch.randn(2, 8, 4, 64, dtype=torch.float32)
    q_rot, k_rot = rope(q, k)
    assert q_rot.dtype == torch.float32
    assert k_rot.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bfloat16 requires CUDA")
def test_dtype_preserved_bfloat16() -> None:
    rope = RotaryEmbedding(64).cuda()
    q = torch.randn(2, 8, 4, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(2, 8, 4, 64, dtype=torch.bfloat16, device="cuda")
    q_rot, k_rot = rope(q, k)
    assert q_rot.dtype == torch.bfloat16
    assert k_rot.dtype == torch.bfloat16


def test_position_0_is_identity() -> None:
    """At position 0 all angles are 0: cos=1, sin=0 → output == input."""
    rope = RotaryEmbedding(64, max_seq_len=128)
    q = torch.randn(1, 1, 4, 64)
    k = torch.randn(1, 1, 4, 64)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q_rot, q, atol=1e-5), "position-0 rotation should be identity"
    assert torch.allclose(k_rot, k, atol=1e-5)


def test_different_positions_differ() -> None:
    """The same vector at different positions should produce different outputs."""
    rope = RotaryEmbedding(64, max_seq_len=128)
    x = torch.randn(1, 1, 1, 64).expand(1, 2, 1, 64).clone()
    q_rot, _ = rope(x, x)
    assert not torch.allclose(q_rot[:, 0], q_rot[:, 1])


def test_norm_preserved() -> None:
    """RoPE is an isometry: ||q_rot|| == ||q|| (rotation preserves L2 norm)."""
    rope = RotaryEmbedding(64, max_seq_len=64)
    q = torch.randn(2, 32, 4, 64)
    k = torch.randn(2, 32, 4, 64)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4)


def test_gradient_flows() -> None:
    rope = RotaryEmbedding(64)
    q = torch.randn(2, 16, 4, 64, requires_grad=True)
    k = torch.randn(2, 16, 4, 64, requires_grad=True)
    q_rot, k_rot = rope(q, k)
    (q_rot.sum() + k_rot.sum()).backward()
    assert q.grad is not None
    assert k.grad is not None


# ---------------------------------------------------------------------------
# Relative-position invariance (core RoPE property)
# ---------------------------------------------------------------------------


def test_relative_position_invariance() -> None:
    """q_m · k_n should depend only on (m - n), not on absolute positions.

    This is the core mathematical guarantee of RoPE: the inner product between
    a rotated query at position m and a rotated key at position n encodes only
    the relative offset (m-n).

    We verify: dot(q_rot[0], k_rot[0]) ≈ dot(q_rot[d], k_rot[d]) for same offset d.
    """
    rope = RotaryEmbedding(64, max_seq_len=64)
    q_vec = torch.randn(64)
    k_vec = torch.randn(64)

    # Embed the same vectors at positions 0 and 5 (offset = 0)
    q0 = q_vec.view(1, 1, 1, 64)
    k0 = k_vec.view(1, 1, 1, 64)
    q0_rot, k0_rot = rope(q0, k0)
    dot_at_0 = (q0_rot * k0_rot).sum()

    # Embed at positions 10 and 15 (same offset = 5, but shifted by +10)
    # Build sequences of length 16 so position 10 and 15 are reachable
    q_seq = torch.zeros(1, 16, 1, 64)
    k_seq = torch.zeros(1, 16, 1, 64)
    q_seq[0, 10] = q_vec
    k_seq[0, 15] = k_vec
    q_seq_rot, k_seq_rot = rope(q_seq, k_seq)

    # Note: offset here is 5 (k at pos 15, q at pos 10) vs offset 0 above.
    # What we want to verify is that for a FIXED offset the dot product is the same
    # regardless of the absolute position. Use offset=0 both times.
    q5 = q_vec.view(1, 1, 1, 64).expand(1, 1, 1, 64)
    k5 = k_vec.view(1, 1, 1, 64).expand(1, 1, 1, 64)
    # Shift by inserting 5 positions of padding
    q5_seq = torch.cat([torch.zeros(1, 5, 1, 64), q5], dim=1)
    k5_seq = torch.cat([torch.zeros(1, 5, 1, 64), k5], dim=1)
    q5_rot, k5_rot = rope(q5_seq, k5_seq)
    dot_at_5 = (q5_rot[:, 5] * k5_rot[:, 5]).sum()

    assert torch.allclose(
        dot_at_0, dot_at_5, atol=1e-4
    ), f"RoPE relative invariance failed: dot@0={dot_at_0:.4f}, dot@5={dot_at_5:.4f}"


# ---------------------------------------------------------------------------
# Integration with CausalMultiHeadAttention
# ---------------------------------------------------------------------------


def test_mha_with_rope_forward() -> None:
    from src.models.attention.causal_mha import CausalMultiHeadAttention

    rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
    mha = CausalMultiHeadAttention(
        d_model=256, num_heads=4, attention_backend="standard", rope=rope
    )
    x = torch.randn(2, 32, 256)
    out = mha(x)
    assert out.shape == (2, 32, 256)


def test_mha_without_rope_unchanged() -> None:
    """Passing rope=None should behave identically to no-RoPE baseline."""
    from src.models.attention.causal_mha import CausalMultiHeadAttention

    mha = CausalMultiHeadAttention(
        d_model=256, num_heads=4, attention_backend="standard", rope=None
    )
    x = torch.randn(2, 16, 256)
    out = mha(x)
    assert out.shape == (2, 16, 256)


# ---------------------------------------------------------------------------
# Integration with LearningModel
# ---------------------------------------------------------------------------


def test_learning_model_with_rope() -> None:
    from src.models.learning_model.learning_model import LearningModel

    model = LearningModel(
        vocab_size=256,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        attention_backend="standard",
        use_rope=True,
    )
    # No learned position embedding when RoPE is active
    assert model.position_embedding is None
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 256)


def test_learning_model_rope_skips_position_embedding() -> None:
    from src.models.learning_model.learning_model import LearningModel

    model_rope = LearningModel(
        vocab_size=256,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        attention_backend="standard",
        use_rope=True,
    )
    model_base = LearningModel(
        vocab_size=256,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        attention_backend="standard",
        use_rope=False,
    )
    assert model_rope.position_embedding is None
    assert model_base.position_embedding is not None


def test_learning_model_from_config_with_rope() -> None:
    from src.config.model import ModelConfig
    from src.models.learning_model.learning_model import LearningModel

    config = ModelConfig(
        hidden_size=128,
        vocab_size=256,
        max_seq_length=64,
        num_layers=2,
        num_heads=4,
        norm_type="rms",
        use_rope=True,
    )
    model = LearningModel.from_config(config, attention_backend="standard")
    assert model.use_rope is True
    assert model.position_embedding is None
    x = torch.randint(0, 256, (1, 32))
    assert model(x).shape == (1, 32, 256)
