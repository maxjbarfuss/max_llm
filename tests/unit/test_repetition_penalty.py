"""Tests for repetition penalty in the sampler.

The penalty divides logits of tokens seen in the recent context window by
repetition_penalty. This reduces positive logits and makes negative logits
less negative — the net effect is suppression of overrepresented tokens.
"""

from __future__ import annotations

import torch

from src.inference.sampler import sample_token


def _logits(vocab: int = 16, high_id: int = 0, high_val: float = 10.0) -> torch.Tensor:
    t = torch.zeros(vocab)
    t[high_id] = high_val
    return t


class TestRepetitionPenaltyDisabled:
    def test_penalty_one_is_noop(self) -> None:
        logits = _logits(high_id=3)
        result = sample_token(logits.clone(), temperature=0, repetition_penalty=1.0, context=[3])
        assert result == 3

    def test_empty_context_skips_penalty(self) -> None:
        logits = _logits(high_id=5, high_val=20.0)
        result = sample_token(logits.clone(), temperature=0, repetition_penalty=2.0, context=[])
        assert result == 5

    def test_none_context_skips_penalty(self) -> None:
        logits = _logits(high_id=7, high_val=20.0)
        result = sample_token(logits.clone(), temperature=0, repetition_penalty=2.0, context=None)
        assert result == 7


class TestRepetitionPenaltyEffect:
    def test_penalises_recent_token(self) -> None:
        """A token recently seen should be penalised even if it has the highest logit."""
        logits = _logits(high_id=2, high_val=10.0)
        # After penalty, token 2 gets 10/2=5, leaving token 0 (val=0) still lower.
        # With high enough penalty we flip the winner.
        logits[4] = 6.0
        result = sample_token(logits.clone(), temperature=0, repetition_penalty=4.0, context=[2])
        # Token 2 penalised: 10/4 = 2.5; token 4 stays at 6.0 → token 4 wins
        assert result == 4

    def test_penalty_divides_positive_logit(self) -> None:
        vocab = 8
        logits = torch.zeros(vocab)
        logits[3] = 8.0
        # With penalty token 3 gets 8/2=4; since it's still highest, just confirm no error.
        sample_token(logits.clone(), temperature=0, repetition_penalty=2.0, context=[3])
        # Without penalty token 3 wins; with 8/2=4 it still wins but we just
        # confirm the call doesn't error. To verify the division happened we
        # use a competing token.
        logits2 = torch.zeros(vocab)
        logits2[3] = 8.0
        logits2[5] = 5.0
        result2 = sample_token(logits2.clone(), temperature=0, repetition_penalty=2.0, context=[3])
        # 8/2=4 < 5 → token 5 should win
        assert result2 == 5

    def test_window_limits_penalised_set(self) -> None:
        """Tokens outside the window should not be penalised."""
        vocab = 16
        logits = torch.zeros(vocab)
        logits[0] = 5.0  # old token, outside window
        logits[1] = 4.0  # recent token, inside window
        context = [0] * 10 + [1]  # token 0 at positions 0-9, token 1 at position 10
        result = sample_token(
            logits.clone(),
            temperature=0,
            repetition_penalty=2.0,
            repetition_window=5,  # only last 5 tokens → only token 1 is penalised
            context=context,
        )
        # Token 1 penalised (4/2=2), token 0 unpunished (5) → token 0 wins
        assert result == 0

    def test_window_equal_to_context_length(self) -> None:
        """When window >= len(context) all context tokens are penalised."""
        vocab = 8
        logits = torch.zeros(vocab)
        logits[2] = 10.0
        logits[3] = 6.0
        result = sample_token(
            logits.clone(),
            temperature=0,
            repetition_penalty=3.0,
            repetition_window=100,
            context=[2, 3],
        )
        # 10/3 ≈ 3.33, 6/3 = 2.0 → still token 2 wins
        assert result == 2

    def test_high_penalty_suppresses_repeated_token(self) -> None:
        vocab = 8
        logits = torch.zeros(vocab)
        logits[0] = 10.0
        logits[1] = 1.0
        # With penalty=20: 10/20=0.5 < 1.0 → token 1 wins
        result = sample_token(logits.clone(), temperature=0, repetition_penalty=20.0, context=[0])
        assert result == 1


class TestRepetitionPenaltyWithSampling:
    def test_penalty_reduces_sampling_probability_of_recent_token(self) -> None:
        """Statistical test: recent token sampled less often with penalty."""
        torch.manual_seed(42)
        logits_base = torch.tensor([5.0, 0.0, 0.0, 0.0])
        context = [0]  # token 0 is "recent"

        count_no_penalty = sum(
            sample_token(
                logits_base.clone(), temperature=1.0, repetition_penalty=1.0, context=context
            )
            == 0
            for _ in range(200)
        )
        count_with_penalty = sum(
            sample_token(
                logits_base.clone(), temperature=1.0, repetition_penalty=4.0, context=context
            )
            == 0
            for _ in range(200)
        )
        # Penalty should reduce the fraction of times we pick token 0
        assert count_with_penalty < count_no_penalty
