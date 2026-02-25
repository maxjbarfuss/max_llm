"""Unit tests for inference sampler module."""

from __future__ import annotations

import torch

from src.inference.sampler import sample_token


def test_greedy_sampling_returns_argmax():
    """Greedy sampling (temperature=0) should return argmax."""
    logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
    token = sample_token(logits, temperature=0.0)
    assert token == 1  # Index of max value (5.0)


def test_greedy_sampling_with_negative_temperature():
    """Negative temperature should also trigger greedy sampling."""
    logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
    token = sample_token(logits, temperature=-1.0)
    assert token == 1


def test_temperature_sampling_returns_valid_token():
    """Temperature sampling should return a valid token index."""
    logits = torch.randn(100)
    token = sample_token(logits, temperature=0.8)
    assert 0 <= token < 100
    assert isinstance(token, int)


def test_temperature_high_increases_entropy():
    """Higher temperature should increase sampling diversity."""
    torch.manual_seed(42)
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])  # Heavily biased

    # Low temperature: should sample mostly from highest logit
    low_temp_samples = [sample_token(logits, temperature=0.1) for _ in range(100)]
    low_entropy = len(set(low_temp_samples))

    # High temperature: should sample more uniformly
    high_temp_samples = [sample_token(logits, temperature=10.0) for _ in range(100)]
    high_entropy = len(set(high_temp_samples))

    assert high_entropy >= low_entropy  # Higher temperature -> more diversity


def test_top_k_filters_to_k_tokens():
    """Top-k should only sample from k highest-probability tokens."""
    torch.manual_seed(42)
    # Create logits where we can track which tokens are sampled
    logits = torch.tensor([100.0, 99.0, 98.0, 1.0, 2.0, 3.0])

    # With top_k=3, should never sample from indices 3, 4, 5
    samples = [sample_token(logits, temperature=1.0, top_k=3) for _ in range(100)]
    sampled_indices = set(samples)

    # Should only sample from top 3: indices 0, 1, 2
    assert sampled_indices.issubset({0, 1, 2})


def test_top_k_zero_disables_filtering():
    """top_k=0 should disable top-k filtering."""
    torch.manual_seed(42)
    logits = torch.randn(10)

    # Should be able to sample from all tokens
    samples = [sample_token(logits, temperature=1.0, top_k=0) for _ in range(200)]
    sampled_indices = set(samples)

    # With enough samples, should see variety (probabilistic test)
    assert len(sampled_indices) > 3


def test_top_p_nucleus_sampling():
    """Top-p should sample from tokens whose cumulative probability <= top_p."""
    torch.manual_seed(42)
    # Create heavily skewed distribution
    logits = torch.tensor([10.0, 9.0, 0.0, 0.0, 0.0])

    # With top_p=0.9, should mostly sample from top tokens
    samples = [sample_token(logits, temperature=1.0, top_p=0.9) for _ in range(100)]
    sampled_indices = set(samples)

    # Should primarily sample from high-probability tokens
    assert 0 in sampled_indices or 1 in sampled_indices


def test_top_p_zero_disables_filtering():
    """top_p=0 should disable nucleus filtering."""
    torch.manual_seed(42)
    logits = torch.randn(10)

    samples = [sample_token(logits, temperature=1.0, top_p=0.0) for _ in range(200)]
    sampled_indices = set(samples)

    # Should sample from variety of tokens
    assert len(sampled_indices) > 3


def test_top_p_one_allows_all_tokens():
    """top_p=1.0 should allow sampling from all tokens."""
    torch.manual_seed(42)
    logits = torch.randn(10)

    samples = [sample_token(logits, temperature=1.0, top_p=1.0) for _ in range(200)]
    sampled_indices = set(samples)

    # Should sample from variety of tokens
    assert len(sampled_indices) > 3


def test_combined_top_k_and_top_p():
    """Should be able to apply both top-k and top-p filtering."""
    torch.manual_seed(42)
    logits = torch.randn(50)

    # Should return valid token with both filters
    token = sample_token(logits, temperature=0.8, top_k=10, top_p=0.9)
    assert 0 <= token < 50
    assert isinstance(token, int)


def test_single_token_vocab():
    """Should handle single-token vocabulary."""
    logits = torch.tensor([5.0])

    # Should always return token 0
    assert sample_token(logits, temperature=0.0) == 0
    assert sample_token(logits, temperature=1.0) == 0
    assert sample_token(logits, temperature=1.0, top_k=1) == 0


def test_deterministic_greedy():
    """Greedy sampling should be deterministic."""
    logits = torch.randn(100)

    token1 = sample_token(logits, temperature=0.0)
    token2 = sample_token(logits, temperature=0.0)
    token3 = sample_token(logits, temperature=0.0)

    assert token1 == token2 == token3


def test_respects_device():
    """Should work with tensors on different devices."""
    if not torch.cuda.is_available():
        # Skip CUDA test if not available
        return

    logits_cpu = torch.randn(50)
    logits_cuda = logits_cpu.cuda()

    # Both should work without error
    token_cpu = sample_token(logits_cpu, temperature=0.8)
    token_cuda = sample_token(logits_cuda, temperature=0.8)

    assert 0 <= token_cpu < 50
    assert 0 <= token_cuda < 50


def test_top_k_larger_than_vocab():
    """top_k larger than vocab size should not crash."""
    logits = torch.randn(10)

    # top_k=100 but vocab_size=10
    token = sample_token(logits, temperature=1.0, top_k=100)
    assert 0 <= token < 10


def test_extreme_logits():
    """Should handle extreme logit values."""
    # Very large logits
    logits_large = torch.tensor([1000.0, 999.0, 998.0])
    token_large = sample_token(logits_large, temperature=1.0)
    assert 0 <= token_large < 3

    # Very small logits
    logits_small = torch.tensor([-1000.0, -999.0, -998.0])
    token_small = sample_token(logits_small, temperature=1.0)
    assert 0 <= token_small < 3

    # Mixed
    logits_mixed = torch.tensor([1000.0, -1000.0, 0.0])
    token_mixed = sample_token(logits_mixed, temperature=1.0)
    assert 0 <= token_mixed < 3


def test_all_equal_logits():
    """Should handle uniform distribution (all logits equal)."""
    torch.manual_seed(42)
    logits = torch.ones(10)

    # Should sample uniformly
    samples = [sample_token(logits, temperature=1.0) for _ in range(100)]
    sampled_indices = set(samples)

    # Should see variety (probabilistic)
    assert len(sampled_indices) > 3


def test_numerical_stability_with_top_p():
    """Top-p should be numerically stable with extreme values."""
    torch.manual_seed(42)

    # Case 1: Very peaked distribution
    logits = torch.tensor([1000.0, 0.0, 0.0, 0.0])
    token = sample_token(logits, temperature=1.0, top_p=0.9)
    assert token >= 0

    # Case 2: Very flat distribution
    logits = torch.tensor([1e-10, 1e-10, 1e-10, 1e-10])
    token = sample_token(logits, temperature=1.0, top_p=0.5)
    assert token >= 0


def test_cutoff_always_keeps_first_token():
    """Top-p cutoff should always keep at least one token."""
    torch.manual_seed(42)

    # Even with very low top_p, should still sample successfully
    logits = torch.randn(10)
    token = sample_token(logits, temperature=1.0, top_p=0.001)
    assert 0 <= token < 10
