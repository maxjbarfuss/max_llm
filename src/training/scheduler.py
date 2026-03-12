"""Learning rate schedulers for training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    import torch.optim.lr_scheduler

    def lr_lambda(current_step: int) -> float:
        """Compute LR multiplier for current step."""
        # Warmup phase: linear ramp from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase: cosine decay from 1 to min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Clamp progress to [0, 1] in case we exceed num_training_steps
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_wsd_schedule(  # noqa: C901
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
    import torch.optim.lr_scheduler

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

    warmup_end = max(0, num_warmup_steps)
    stable_steps = int(num_training_steps * stable_fraction)
    decay_steps = int(num_training_steps * decay_fraction)
    stable_end = warmup_end + stable_steps
    decay_end = stable_end + max(1, decay_steps)

    def _decay_kernel(progress: float) -> float:
        # progress in [0, 1], returns multiplier in [0, 1]
        p = min(1.0, max(0.0, progress))
        if decay_shape == "linear":
            return 1.0 - p
        if decay_shape == "sqrt":
            return math.sqrt(max(0.0, 1.0 - p))
        # lowered-linear decay: keep lr higher early, then decay faster near end
        return max(0.0, 1.0 - (p**lowered_linear_alpha))

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_end:
            return float(current_step) / float(max(1, warmup_end))

        # Stable plateau phase
        if current_step < stable_end:
            return 1.0

        # Decay phase
        if current_step < decay_end:
            progress = float(current_step - stable_end) / float(max(1, decay_end - stable_end))
            kernel = _decay_kernel(progress)
            return min_lr_ratio + (1.0 - min_lr_ratio) * kernel

        # Floor phase after planned decay window
        return min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
