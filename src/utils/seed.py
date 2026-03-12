"""Seed management and deterministic training utilities."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Covers Python random, NumPy, PyTorch CPU/CUDA, PYTHONHASHSEED, and cuDNN.

    Args:
        seed: Random seed value.
        deterministic: If True, enables deterministic algorithms in PyTorch
            (may reduce performance; some ops error if no deterministic kernel exists).

    Note:
        For full DataLoader reproducibility also set worker_init_fn=seed_worker.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker for reproducibility.

    Use as worker_init_fn:

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     worker_init_fn=seed_worker,
        ...     generator=torch.Generator().manual_seed(42),
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
