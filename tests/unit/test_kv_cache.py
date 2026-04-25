"""Tests for KV-cache infrastructure and correctness of cached generation.

Correctness invariant: step-by-step generation with KV-cache must produce
identical logits to full-sequence recomputation without cache.
"""

import pytest
import torch

from src.config.model import ModelConfig
from src.models.learning_model.learning_model import LearningModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mha_model(num_layers: int = 2) -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=num_layers,
        num_heads=4,
        vocab_size=64,
        max_seq_length=32,
        rope_base=10000,
        attn_type="mha",
        norm_type="rms",
        intermediate_size=256,
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


def _mla_model(num_layers: int = 2) -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=64,
        max_seq_length=32,
        rope_base=10000,
        attn_type="mla",
        norm_type="rms",
        intermediate_size=256,
        mla_latent_dim=128,
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


def _mla_block_attn_model() -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=6,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=64,
        max_seq_length=32,
        rope_base=10000,
        attn_type="mla",
        norm_type="rms",
        intermediate_size=256,
        mla_latent_dim=128,
        res_type="block_attn",
        attn_res_num_blocks=3,
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


def _mha_full_attn_model(num_layers: int = 4) -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=num_layers,
        num_heads=4,
        vocab_size=64,
        max_seq_length=32,
        rope_base=10000,
        pos_type="rope",
        attn_type="mha",
        norm_type="rms",
        intermediate_size=256,
        res_type="full_attn",
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


def _mha_full_attn_shared_model(num_layers: int = 4) -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=num_layers,
        num_heads=4,
        vocab_size=64,
        max_seq_length=32,
        rope_base=10000,
        pos_type="rope",
        attn_type="mha",
        norm_type="rms",
        intermediate_size=256,
        res_type="full_attn",
        share_layer_weights=True,
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


def _mha_block_attn_shared_model() -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=6,
        num_heads=4,
        vocab_size=64,
        max_seq_length=32,
        rope_base=10000,
        pos_type="rope",
        attn_type="mha",
        norm_type="rms",
        intermediate_size=256,
        res_type="block_attn",
        attn_res_num_blocks=3,
        share_layer_weights=True,
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


def _mha_learned_pos_model(num_layers: int = 2) -> LearningModel:
    cfg = ModelConfig(
        hidden_size=128,
        num_layers=num_layers,
        num_heads=4,
        vocab_size=64,
        max_seq_length=32,
        pos_type="learned",
        attn_type="mha",
        norm_type="rms",
        intermediate_size=256,
    )
    return LearningModel.from_config(cfg, attention_backend="standard").eval()


# ---------------------------------------------------------------------------
# LayerKVCache unit tests
# ---------------------------------------------------------------------------


def test_layer_kv_cache_init_shape() -> None:
    from src.models.kv_cache import LayerKVCache

    cache = LayerKVCache(max_seq_len=16, num_kv_heads=2, head_dim=32, batch_size=1)
    assert cache._k.shape == (1, 16, 2, 32)
    assert cache._v.shape == (1, 16, 2, 32)
    assert cache.length == 0


def test_layer_kv_cache_init_zeros() -> None:
    from src.models.kv_cache import LayerKVCache

    cache = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=16, batch_size=2)
    assert cache._k.eq(0).all()
    assert cache._v.eq(0).all()


def test_layer_kv_cache_update_single_token() -> None:
    from src.models.kv_cache import LayerKVCache

    cache = LayerKVCache(max_seq_len=8, num_kv_heads=2, head_dim=4, batch_size=1)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    k_out, v_out = cache.update(k, v)

    assert cache.length == 1
    assert k_out.shape == (1, 1, 2, 4)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_layer_kv_cache_update_prefill() -> None:
    from src.models.kv_cache import LayerKVCache

    T = 5
    cache = LayerKVCache(max_seq_len=16, num_kv_heads=2, head_dim=4, batch_size=1)
    k = torch.randn(1, T, 2, 4)
    v = torch.randn(1, T, 2, 4)
    k_out, v_out = cache.update(k, v)

    assert cache.length == T
    assert k_out.shape == (1, T, 2, 4)
    assert torch.allclose(k_out, k)


def test_layer_kv_cache_sequential_builds_sequence() -> None:
    """Three single-token updates should produce the same K as one 3-token update."""
    from src.models.kv_cache import LayerKVCache

    torch.manual_seed(7)
    k_all = torch.randn(1, 3, 2, 4)
    v_all = torch.randn(1, 3, 2, 4)

    cache_seq = LayerKVCache(max_seq_len=8, num_kv_heads=2, head_dim=4, batch_size=1)
    for t in range(3):
        k_out, v_out = cache_seq.update(k_all[:, t : t + 1], v_all[:, t : t + 1])

    cache_bulk = LayerKVCache(max_seq_len=8, num_kv_heads=2, head_dim=4, batch_size=1)
    k_bulk, v_bulk = cache_bulk.update(k_all, v_all)

    assert torch.allclose(k_out, k_bulk)
    assert torch.allclose(v_out, v_bulk)
    assert cache_seq.length == cache_bulk.length == 3


def test_layer_kv_cache_overflow_raises() -> None:
    from src.models.kv_cache import LayerKVCache

    cache = LayerKVCache(max_seq_len=4, num_kv_heads=1, head_dim=8, batch_size=1)
    k = torch.zeros(1, 3, 1, 8)
    v = torch.zeros(1, 3, 1, 8)
    cache.update(k, v)  # length = 3

    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        cache.update(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8))  # would overflow


def test_layer_kv_cache_reset() -> None:
    from src.models.kv_cache import LayerKVCache

    cache = LayerKVCache(max_seq_len=8, num_kv_heads=2, head_dim=4, batch_size=1)
    cache.update(torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 4))
    assert cache.length == 4

    cache.reset()
    assert cache.length == 0


def test_layer_kv_cache_device_and_dtype() -> None:
    from src.models.kv_cache import LayerKVCache

    cache = LayerKVCache(
        max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1, dtype=torch.float16
    )
    assert cache._k.dtype == torch.float16


# ---------------------------------------------------------------------------
# ModelKVCache unit tests
# ---------------------------------------------------------------------------


def test_model_kv_cache_length() -> None:
    from src.models.kv_cache import LayerKVCache, ModelKVCache

    c0 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    c1 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    mc = ModelKVCache([c0, c1])

    assert mc.length == 0
    c0.update(torch.zeros(1, 3, 1, 4), torch.zeros(1, 3, 1, 4))
    c1.update(torch.zeros(1, 3, 1, 4), torch.zeros(1, 3, 1, 4))
    assert mc.length == 3


def test_model_kv_cache_length_desync_raises() -> None:
    from src.models.kv_cache import LayerKVCache, ModelKVCache

    c0 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    c1 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    mc = ModelKVCache([c0, c1])

    c0.update(torch.zeros(1, 3, 1, 4), torch.zeros(1, 3, 1, 4))
    with pytest.raises(RuntimeError, match="Inconsistent KV-cache lengths"):
        _ = mc.length


def test_model_kv_cache_getitem() -> None:
    from src.models.kv_cache import LayerKVCache, ModelKVCache

    c0 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    c1 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    mc = ModelKVCache([c0, c1])

    assert mc[0] is c0
    assert mc[1] is c1
    assert len(mc) == 2


def test_model_kv_cache_reset() -> None:
    from src.models.kv_cache import LayerKVCache, ModelKVCache

    c0 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    c1 = LayerKVCache(max_seq_len=8, num_kv_heads=1, head_dim=4, batch_size=1)
    mc = ModelKVCache([c0, c1])

    c0.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    c1.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    assert mc.length == 2

    mc.reset()
    assert mc.length == 0
    assert c0.length == 0
    assert c1.length == 0


# ---------------------------------------------------------------------------
# make_kv_cache
# ---------------------------------------------------------------------------


def test_make_kv_cache_mha_structure() -> None:
    model = _mha_model(num_layers=3)
    cache = model.make_kv_cache(max_seq_len=32, batch_size=1, dtype=torch.float32)

    assert len(cache) == 3
    for layer_cache in cache:
        assert layer_cache is not None
        assert layer_cache.length == 0


def test_make_kv_cache_mla_structure() -> None:
    model = _mla_model(num_layers=2)
    head_dim = 128 // 4  # d_model // num_heads
    cache = model.make_kv_cache(max_seq_len=32, batch_size=1, dtype=torch.float32)

    assert len(cache) == 2
    layer0 = cache[0]
    assert layer0 is not None
    assert layer0._k.shape == (1, 32, 2, head_dim)  # batch=1, seq=32, kv_heads=2, head_dim
    assert layer0._v.shape == (1, 32, 2, head_dim)


def test_make_kv_cache_batch_size() -> None:
    model = _mha_model(num_layers=2)
    cache = model.make_kv_cache(max_seq_len=16, batch_size=3, dtype=torch.float32)
    layer0 = cache[0]
    assert layer0 is not None
    assert layer0._k.shape[0] == 3


# ---------------------------------------------------------------------------
# RoPE pos_offset
# ---------------------------------------------------------------------------


def test_rope_pos_offset_shifts_embedding() -> None:
    """Applying rope at pos_offset=t should equal starting a sequence at position t."""
    from src.models.position.rope import RotaryEmbedding

    rope = RotaryEmbedding(head_dim=64, max_seq_len=32, base=10000)
    q = torch.randn(1, 1, 4, 64)
    k = torch.randn(1, 1, 4, 64)

    # pos_offset=5 for a length-1 sequence → should rotate at position 5
    q_off, k_off = rope(q, k, pos_offset=5)

    # Direct: build a length-6 sequence, take position 5
    q_long = torch.zeros(1, 6, 4, 64)
    k_long = torch.zeros(1, 6, 4, 64)
    q_long[0, 5] = q[0, 0]
    k_long[0, 5] = k[0, 0]
    q_rot, k_rot = rope(q_long, k_long)

    assert torch.allclose(q_off[0, 0], q_rot[0, 5], atol=1e-5)
    assert torch.allclose(k_off[0, 0], k_rot[0, 5], atol=1e-5)


# ---------------------------------------------------------------------------
# Correctness: KV-cache generation == full-sequence recompute
# ---------------------------------------------------------------------------


def _verify_kv_cache_correctness(
    model: LearningModel,
    seq_len: int = 8,
    atol: float = 1e-4,
) -> None:
    """Core correctness check: step-by-step cached generation matches full recompute."""
    torch.manual_seed(0)
    vocab = model.token_embedding.embedding.num_embeddings
    input_ids = torch.randint(0, vocab, (1, seq_len))

    model.eval()
    with torch.no_grad():
        logits_full = model(input_ids)  # (1, seq_len, vocab)

    kv_caches = model.make_kv_cache(
        max_seq_len=seq_len + 4,
        batch_size=1,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for t in range(seq_len):
            token = input_ids[:, t : t + 1]
            logits_step = model(token, kv_caches=kv_caches)  # (1, 1, vocab)

    # Final step logits must match full-sequence logits at last position
    assert torch.allclose(
        logits_full[0, -1], logits_step[0, 0], atol=atol
    ), f"Max diff: {(logits_full[0, -1] - logits_step[0, 0]).abs().max():.6f}"


def test_mha_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mha_model(), seq_len=8)


def test_mla_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mla_model(), seq_len=8)


def test_mla_block_attn_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mla_block_attn_model(), seq_len=8)


def test_mha_full_attn_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mha_full_attn_model(), seq_len=8)


def test_mha_full_attn_shared_layers_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mha_full_attn_shared_model(), seq_len=8)


def test_mha_block_attn_shared_layers_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mha_block_attn_shared_model(), seq_len=8)


def test_mha_learned_pos_kv_cache_correctness() -> None:
    _verify_kv_cache_correctness(_mha_learned_pos_model(), seq_len=8)


def test_kv_cache_correctness_all_positions() -> None:
    """Every intermediate position matches, not just the last."""
    torch.manual_seed(1)
    model = _mha_model(num_layers=2)
    vocab = model.token_embedding.embedding.num_embeddings
    seq_len = 6
    input_ids = torch.randint(0, vocab, (1, seq_len))

    model.eval()
    with torch.no_grad():
        logits_full = model(input_ids)  # (1, seq_len, vocab)

    kv_caches = model.make_kv_cache(max_seq_len=seq_len + 4, batch_size=1, dtype=torch.float32)

    with torch.no_grad():
        for t in range(seq_len):
            token = input_ids[:, t : t + 1]
            logits_step = model(token, kv_caches=kv_caches)
            assert torch.allclose(logits_full[0, t], logits_step[0, 0], atol=1e-4), (
                f"Position {t}: max diff "
                f"{(logits_full[0, t] - logits_step[0, 0]).abs().max():.6f}"
            )


def test_kv_cache_reset_allows_reuse() -> None:
    """After reset, cache produces same results as fresh cache."""
    torch.manual_seed(2)
    model = _mha_model(num_layers=2)
    vocab = model.token_embedding.embedding.num_embeddings
    seq_len = 5
    input_ids = torch.randint(0, vocab, (1, seq_len))

    model.eval()
    cache = model.make_kv_cache(max_seq_len=16, batch_size=1, dtype=torch.float32)

    with torch.no_grad():
        for t in range(seq_len):
            out1 = model(input_ids[:, t : t + 1], kv_caches=cache)

    cache.reset()

    with torch.no_grad():
        for t in range(seq_len):
            out2 = model(input_ids[:, t : t + 1], kv_caches=cache)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_kv_cache_prefill_then_generation() -> None:
    """Prefill full prompt at once, then generate one more token."""
    torch.manual_seed(3)
    model = _mha_model(num_layers=2)
    vocab = model.token_embedding.embedding.num_embeddings
    prompt_len, extra = 5, 1
    all_ids = torch.randint(0, vocab, (1, prompt_len + extra))

    model.eval()

    # Reference: full sequence forward
    with torch.no_grad():
        logits_ref = model(all_ids)[0, -1]

    # Prefill + one generation step
    cache = model.make_kv_cache(max_seq_len=16, batch_size=1, dtype=torch.float32)
    with torch.no_grad():
        model(all_ids[:, :prompt_len], kv_caches=cache)  # prefill
        logits_gen = model(all_ids[:, prompt_len:], kv_caches=cache)[0, 0]  # generate

    assert torch.allclose(logits_ref, logits_gen, atol=1e-4)


def test_kv_cache_layer_lengths_stay_synchronized() -> None:
    """All non-None layer caches keep the same length across prefill+decode."""
    torch.manual_seed(4)
    model = _mha_model(num_layers=3)
    vocab = model.token_embedding.embedding.num_embeddings
    all_ids = torch.randint(0, vocab, (1, 7))

    model.eval()
    cache = model.make_kv_cache(max_seq_len=16, batch_size=1, dtype=torch.float32)

    with torch.no_grad():
        model(all_ids[:, :5], kv_caches=cache)  # prefill
        _ = model(all_ids[:, 5:6], kv_caches=cache)  # decode 1
        _ = model(all_ids[:, 6:7], kv_caches=cache)  # decode 2

    lengths = [c.length for c in cache if c is not None]
    assert lengths
    assert all(layer_len == lengths[0] for layer_len in lengths)
    assert cache.length == lengths[0] == 7


def test_make_kv_cache_uses_model_max_seq_len_as_default() -> None:
    """make_kv_cache with no max_seq_len uses model.max_seq_len."""
    model = _mha_model()
    cache = model.make_kv_cache(batch_size=1, dtype=torch.float32)
    layer0 = cache[0]
    assert layer0 is not None
    assert layer0._k.shape[1] == model.max_seq_len
