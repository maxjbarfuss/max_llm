"""Learning rate schedulers for training."""

import math

import torch.optim
import torch.optim.lr_scheduler


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay.

    Schedule:
        - Steps 0 to num_warmup_steps: Linear warmup from 0 to base_lr
        - Steps num_warmup_steps to num_training_steps: Cosine decay from base_lr to min_lr

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as fraction of base LR (default: 0.1).
        last_epoch: Last epoch index for resuming (default: -1).

    Returns:
        LambdaLR scheduler instance.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=100, num_training_steps=1000
        ... )
        >>> for step in range(1000):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear ramp from 0 to 1
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # Decay phase: cosine decay from 1 to min_lr_ratio
        progress = min(
            1.0,
            (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps),
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _decay_multiplier(progress: float, decay_shape: str, lowered_linear_alpha: float) -> float:
    """Map decay progress in [0, 1] to a multiplier in [0, 1]."""
    p = min(1.0, max(0.0, progress))
    if decay_shape == "linear":
        return 1.0 - p
    if decay_shape == "sqrt":
        return math.sqrt(max(0.0, 1.0 - p))
    # lowered-linear: stays higher early, decays faster near end
    return max(0.0, 1.0 - (p**lowered_linear_alpha))


def _validate_wsd_params(
    num_training_steps: int,
    stable_fraction: float,
    decay_fraction: float,
    min_lr_ratio: float,
    decay_shape: str,
    lowered_linear_alpha: float,
) -> None:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if not (0.0 <= stable_fraction <= 1.0):
        raise ValueError("stable_fraction must be in [0, 1]")
    if not (0.0 <= decay_fraction <= 1.0):
        raise ValueError("decay_fraction must be in [0, 1]")
    if stable_fraction + decay_fraction > 1.0:
        raise ValueError("stable_fraction + decay_fraction must be <= 1.0")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if decay_shape not in {"linear", "sqrt", "lowered_linear"}:
        raise ValueError("decay_shape must be one of: linear, sqrt, lowered_linear")
    if not (0.0 < lowered_linear_alpha <= 1.0):
        raise ValueError("lowered_linear_alpha must be in (0, 1]")


def get_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    stable_fraction: float = 0.7,
    decay_fraction: float = 0.2,
    decay_shape: str = "sqrt",
    min_lr_ratio: float = 0.0,
    lowered_linear_alpha: float = 0.7,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a Warmup-Stable-Decay (WSD) learning rate schedule.

    WSD schedule phases:
        - Warmup: linear ramp from 0 -> base_lr for ``num_warmup_steps``.
        - Stable: hold base_lr constant for ``stable_fraction`` of total steps.
        - Decay: decay base_lr -> ``min_lr_ratio * base_lr`` for ``decay_fraction``
          of total steps using one of ``linear``, ``sqrt``, ``lowered_linear``.

    Any remaining steps after warmup+stable+decay stay at ``min_lr_ratio``.
    """
    _validate_wsd_params(
        num_training_steps,
        stable_fraction,
        decay_fraction,
        min_lr_ratio,
        decay_shape,
        lowered_linear_alpha,
    )

    warmup_end = max(0, num_warmup_steps)
    stable_steps = int(num_training_steps * stable_fraction)
    decay_steps = int(num_training_steps * decay_fraction)
    stable_end = warmup_end + stable_steps
    decay_end = stable_end + max(1, decay_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_end:
            return current_step / max(1, warmup_end)
        if current_step < stable_end:
            return 1.0
        if current_step < decay_end:
            progress = (current_step - stable_end) / max(1, decay_end - stable_end)
            return min_lr_ratio + (1.0 - min_lr_ratio) * _decay_multiplier(
                progress, decay_shape, lowered_linear_alpha
            )
        return min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _validate_resume_params(
    num_training_steps: int,
    hold_steps: int,
    ramp_steps: int,
    start_lr_ratio: float,
) -> None:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if hold_steps < 0:
        raise ValueError("hold_steps must be >= 0")
    if ramp_steps < 0:
        raise ValueError("ramp_steps must be >= 0")
    if start_lr_ratio < 0.0:
        raise ValueError("start_lr_ratio must be >= 0 (values > 1 are valid for ramp-down)")
    if hold_steps + ramp_steps >= num_training_steps:
        raise ValueError("hold_steps + ramp_steps must be < num_training_steps")


def get_resume_hold_ramp_then_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    start_lr_ratio: float,
    hold_steps: int,
    ramp_steps: int,
    stable_fraction: float = 0.7,
    decay_fraction: float = 0.2,
    decay_shape: str = "sqrt",
    min_lr_ratio: float = 0.0,
    lowered_linear_alpha: float = 0.7,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Schedule for resumed runs: hold checkpoint LR, ramp to target LR, then WSD.

    This is intended for restart workflows where model weights are restored from
    checkpoint, but we want a controlled LR transition for the new run.

    Phases:
        1) Hold ``start_lr_ratio * base_lr`` for ``hold_steps``
        2) Linearly ramp to ``base_lr`` over ``ramp_steps``
        3) Apply a WSD schedule over remaining steps (no extra warmup)
    """
    _validate_resume_params(num_training_steps, hold_steps, ramp_steps, start_lr_ratio)
    transition_steps = hold_steps + ramp_steps

    tail_steps = num_training_steps - transition_steps
    _validate_wsd_params(
        tail_steps,
        stable_fraction,
        decay_fraction,
        min_lr_ratio,
        decay_shape,
        lowered_linear_alpha,
    )

    stable_steps = int(tail_steps * stable_fraction)
    decay_steps = int(tail_steps * decay_fraction)
    stable_end = stable_steps
    decay_end = stable_end + max(1, decay_steps)

    def _tail_wsd_ratio(tail_step: int) -> float:
        if tail_step < stable_end:
            return 1.0
        if tail_step < decay_end:
            progress = (tail_step - stable_end) / max(1, decay_end - stable_end)
            return min_lr_ratio + (1.0 - min_lr_ratio) * _decay_multiplier(
                progress, decay_shape, lowered_linear_alpha
            )
        return min_lr_ratio

    def lr_lambda(current_step: int) -> float:
        if current_step < hold_steps:
            return start_lr_ratio
        if current_step < transition_steps:
            ramp_idx = current_step - hold_steps + 1
            progress = min(1.0, ramp_idx / max(1, ramp_steps))
            return start_lr_ratio + (1.0 - start_lr_ratio) * progress

        tail_step = current_step - transition_steps
        return _tail_wsd_ratio(tail_step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_sgdr_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_cycles: int = 4,
    min_lr_ratio: float = 0.05,
    cycle_decay: float = 0.8,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with warm restarts (SGDR), with decaying peak LR per cycle.

    Each cycle cosine-decays from ``peak_lr`` to ``min_lr_ratio * base_lr``.
    The peak multiplier shrinks by ``cycle_decay`` each restart, so later
    cycles make smaller excursions and converge more finely.

    Args:
        optimizer: Optimizer to schedule.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (restarts).
        min_lr_ratio: Floor LR as fraction of base LR per cycle.
        cycle_decay: Multiply peak LR by this factor each cycle (< 1 -> shrinking).
        last_epoch: Last epoch index for resuming (default: -1).
    """
    if num_cycles < 1:
        raise ValueError("num_cycles must be >= 1")
    if not (0.0 <= min_lr_ratio < 1.0):
        raise ValueError("min_lr_ratio must be in [0, 1)")
    if not (0.0 < cycle_decay <= 1.0):
        raise ValueError("cycle_decay must be in (0, 1]")

    cycle_length = max(1, num_training_steps // num_cycles)

    def lr_lambda(current_step: int) -> float:
        cycle = min(current_step // cycle_length, num_cycles - 1)
        cycle_step = current_step - cycle * cycle_length
        progress = min(1.0, cycle_step / cycle_length)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        peak = cycle_decay**cycle  # shrinks each restart
        return min_lr_ratio + (peak - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
