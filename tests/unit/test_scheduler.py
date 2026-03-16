"""Unit tests for learning rate schedulers."""

import pytest
import torch
import torch.nn as nn

from src.training.scheduler import get_cosine_schedule_with_warmup, get_wsd_schedule


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)


@pytest.fixture
def model() -> nn.Module:
    """Create a dummy model for testing."""
    return DummyModel()


@pytest.fixture
def optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Create an optimizer for testing."""
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def test_scheduler_warmup_phase(optimizer: torch.optim.Optimizer) -> None:
    """Test linear warmup phase of scheduler."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )

    base_lr = 1e-3

    # Step 0: should be 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)

    # Step through warmup
    for step in range(10):
        scheduler.step()
        expected_lr = base_lr * (step + 1) / 10
        actual_lr = optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(expected_lr, rel=1e-5)


def test_scheduler_decay_phase(optimizer: torch.optim.Optimizer) -> None:
    """Test cosine decay phase of scheduler."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        min_lr_ratio=0.1,
    )

    base_lr = 1e-3

    # Step through warmup to reach peak LR
    for _ in range(10):
        scheduler.step()

    # At step 10, should be at peak LR
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-5)

    # Step to mid-decay (step 55: halfway through decay phase)
    for _ in range(45):
        scheduler.step()

    # At midpoint of decay, LR should be approximately (1 + min_lr_ratio) / 2
    mid_lr = base_lr * (1.0 + 0.1) / 2
    actual_lr = optimizer.param_groups[0]["lr"]
    assert actual_lr == pytest.approx(mid_lr, rel=0.05)  # Allow 5% tolerance

    # Step to end (step 100)
    for _ in range(45):
        scheduler.step()

    # At end, should be at min_lr
    min_lr = base_lr * 0.1
    actual_lr = optimizer.param_groups[0]["lr"]
    assert actual_lr == pytest.approx(min_lr, rel=1e-5)


def test_scheduler_monotonic_warmup(optimizer: torch.optim.Optimizer) -> None:
    """Test that LR increases monotonically during warmup."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=100,
    )

    prev_lr = 0.0
    for _ in range(20):
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr > prev_lr
        prev_lr = current_lr


def test_scheduler_monotonic_decay(optimizer: torch.optim.Optimizer) -> None:
    """Test that LR decreases monotonically during decay."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        min_lr_ratio=0.1,
    )

    # Skip through warmup
    for _ in range(10):
        scheduler.step()

    prev_lr = optimizer.param_groups[0]["lr"]
    for _ in range(90):
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr <= prev_lr
        prev_lr = current_lr


def test_scheduler_min_lr_ratio(optimizer: torch.optim.Optimizer) -> None:
    """Test different min_lr_ratio values."""
    base_lr = 1e-3

    for min_lr_ratio in [0.0, 0.1, 0.5]:
        # Reset optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            min_lr_ratio=min_lr_ratio,
        )

        # Step to end
        for _ in range(100):
            scheduler.step()

        expected_min_lr = base_lr * min_lr_ratio
        actual_lr = optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(expected_min_lr, rel=1e-5)


def test_scheduler_no_warmup(optimizer: torch.optim.Optimizer) -> None:
    """Test scheduler with zero warmup steps."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=100,
        min_lr_ratio=0.1,
    )

    base_lr = 1e-3

    # Step 0: should already be at peak LR
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-5)

    # After one step, should start decaying
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < base_lr


def test_scheduler_exceeds_max_steps(optimizer: torch.optim.Optimizer) -> None:
    """Test that LR doesn't go below min_lr when exceeding max_steps."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        min_lr_ratio=0.1,
    )

    base_lr = 1e-3
    min_lr = base_lr * 0.1

    # Step beyond max_steps
    for _ in range(150):
        scheduler.step()

    # LR should stay at min_lr, not go negative or below
    actual_lr = optimizer.param_groups[0]["lr"]
    assert actual_lr >= min_lr * 0.99  # Allow tiny numerical error


def test_scheduler_with_multiple_param_groups() -> None:
    """Test scheduler works with multiple parameter groups."""
    model = DummyModel()
    param_groups = [
        {"params": [model.linear.weight], "lr": 1e-3},
        {"params": [model.linear.bias], "lr": 1e-3},
    ]
    optimizer = torch.optim.Adam(param_groups)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )

    # Step through and verify both groups have same LR
    for _ in range(50):
        scheduler.step()
        lr_group_0 = optimizer.param_groups[0]["lr"]
        lr_group_1 = optimizer.param_groups[1]["lr"]
        assert lr_group_0 == pytest.approx(lr_group_1, rel=1e-5)


def test_wsd_phases_behave_as_expected(optimizer: torch.optim.Optimizer) -> None:
    """WSD should warm up, stay flat, then decay."""
    base_lr = 1e-3
    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        stable_fraction=0.7,
        decay_fraction=0.2,
        decay_shape="sqrt",
        min_lr_ratio=0.1,
    )

    # Warmup endpoint
    for _ in range(10):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-5)

    # Stable plateau: steps 10..79
    for _ in range(69):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-5)

    # Enter decay and reach end floor
    for _ in range(21):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * 0.1, rel=1e-5)


@pytest.mark.parametrize("shape", ["linear", "sqrt", "lowered_linear"])
def test_wsd_decay_shapes_reach_floor(optimizer: torch.optim.Optimizer, shape: str) -> None:
    """All WSD decay kernels should end at min_lr_ratio."""
    base_lr = 1e-3
    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=50,
        stable_fraction=0.6,
        decay_fraction=0.3,
        decay_shape=shape,
        min_lr_ratio=0.2,
        lowered_linear_alpha=0.7,
    )

    for _ in range(60):
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * 0.2, rel=1e-5)
