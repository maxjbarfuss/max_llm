"""Seed management and deterministic training utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA if available)
    - CUDA cuDNN (if deterministic=True)

    Args:
        seed: Random seed value.
        deterministic: If True, enables deterministic algorithms in PyTorch
            (may reduce performance). Note: some operations do not have
            deterministic implementations and will error if enabled.

    Note:
        For full reproducibility with DataLoader, also set worker_init_fn
        to seed workers. Use seed_worker() from this module.

    Example:
        >>> seed_everything(42, deterministic=True)
        >>> # Train with reproducible results
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Environment variable for additional control
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Set cuDNN settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # For performance, allow non-deterministic algorithms
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed worker for DataLoader to ensure reproducibility.

    Use as worker_init_fn in DataLoader constructor:

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     worker_init_fn=seed_worker,
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
