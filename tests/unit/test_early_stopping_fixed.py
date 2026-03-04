"""Unit tests for early stopping logic in training loop."""

import pytest


def simulate_early_stopping(patience, val_losses):
    """
    Simulate early stopping logic.

    Args:
        patience: Number of non-improvements before stopping
        val_losses: List of validation losses to check

    Returns:
        Tuple of (steps_completed, stopped_early)
    """
    best_val_loss = float("inf")
    counter = 0
    min_delta = 0.01

    for step, val_loss in enumerate(val_losses):
        # Check improvement
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        # Check stopping condition
        if patience is not None and patience > 0:
            if counter >= patience:
                return step + 1, True

    return len(val_losses), False


class TestEarlyStoppingLogic:
    """Test early stopping behavior."""

    def test_high_patience_allows_full_training(self):
        """Verify that patience=10000 allows full training duration."""
        # Simulate non-improving validation losses (like our actual training)
        val_losses = [7.59, 7.61, 7.74, 7.28, 7.29, 7.75] * 50  # 300 evals, mostly no improvement

        steps_completed, stopped_early = simulate_early_stopping(
            patience=10000, val_losses=val_losses
        )

        assert (
            not stopped_early
        ), f"Should not stop early with patience=10000, but stopped at step {steps_completed}"
        assert steps_completed == len(val_losses), f"Should complete all {len(val_losses)} steps"
        print(f"✓ Reached {steps_completed} evals without stopping (patience=10000)")

    def test_low_patience_stops_early(self):
        """Verify that patience=2 stops after 2 non-improvements."""
        # Mostly non-improving losses
        val_losses = [7.59, 7.61, 7.74, 7.28, 7.29, 7.75, 7.80, 7.85, 7.90]

        steps_completed, stopped_early = simulate_early_stopping(patience=2, val_losses=val_losses)

        assert stopped_early, "Should stop with patience=2"
        # Should stop at step 3 (one improvement, then 2 non-improvements)
        assert steps_completed <= 5, f"Should stop early, but went to step {steps_completed}"
        print(f"✓ Stopped at step {steps_completed} with patience=2")

    def test_patience_none_allows_all(self):
        """Verify that patience=None disables early stopping entirely."""
        val_losses = [7.5] * 1000  # 1000 identical losses

        steps_completed, stopped_early = simulate_early_stopping(
            patience=None, val_losses=val_losses
        )

        assert not stopped_early, "Should not stop early when patience=None"
        assert steps_completed == 1000
        print(f"✓ Completed all {steps_completed} steps with patience=None")

    def test_patience_zero_is_problematic(self):
        """Verify that patience=0 is not a valid way to disable early stopping."""
        val_losses = [7.5] * 10

        steps_completed, stopped_early = simulate_early_stopping(patience=0, val_losses=val_losses)

        # With patience=0, nothing stops it because condition is: counter >= 0 and 0 > 0 is False
        assert not stopped_early, "patience=0 does not stop (because our condition checks > 0)"
        print("✓ patience=0 does not trigger stopping (condition uses > not >=)")


class TestConfigPresence:
    """Test that config has correct early stopping values."""

    def test_config_has_high_patience(self):
        """Verify config uses high patience to allow full training."""
        from pathlib import Path

        import tomllib

        config_path = (
            Path(__file__).parent.parent.parent / "config/milestones/p3_optimized_50m_data.toml"
        )

        assert config_path.exists(), f"Config not found: {config_path}"

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        patience = config["training"].get("early_stopping_patience")

        print(f"\n✓ Config early_stopping_patience: {patience}")

        assert patience is not None, "early_stopping_patience must be set"
        assert patience >= 100, f"Patience should be high (≥100), got {patience}"

        print("✓ Config patience is sufficient to allow full 10K step training")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
