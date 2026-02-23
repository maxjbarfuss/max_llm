"""Phase 2 training loop: forward → CE loss → backward → optimizer step."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.learning_model import BaseLearningModel


def train_step(
    model: BaseLearningModel,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single forward + backward + optimizer step.

    Args:
        model: The model to train. Must be in training mode.
        x: Input token indices of shape (batch_size, seq_len).
        y: Target token indices of shape (batch_size, seq_len).
        optimizer: Optimizer to step.

    Returns:
        Scalar cross-entropy loss for this batch.
    """
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)

    logits = model(x)           # (B, T, V)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    model: BaseLearningModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    log_interval: int = 10,
) -> list[float]:
    """Train for exactly max_steps gradient steps, cycling through train_loader.

    Args:
        model: The model to train.
        train_loader: DataLoader yielding (x, y) batches.
        optimizer: Optimizer.
        max_steps: Total number of gradient steps to take.
        log_interval: Print loss every this many steps. 0 = silent.

    Returns:
        List of per-step scalar losses with length == max_steps.
    """
    model.train()
    losses: list[float] = []
    step = 0

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            x, y = batch
            loss = train_step(model, x, y, optimizer)
            losses.append(loss)
            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(f"step {step + 1:>5}/{max_steps}  loss={loss:.4f}")
            step += 1

    return losses
