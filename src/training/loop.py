"""Phase 2 training loop: forward → CE loss → backward → optimizer step."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.learning_model import BaseLearningModel


def compute_perplexity(loss: float) -> float:
    """Convert loss to perplexity.

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity (e^loss).
    """
    return math.exp(loss)


def train_step(
    model: BaseLearningModel,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single forward + backward + optimizer step.

    Args:
        model: The model to train.
        x: Input token indices of shape (batch_size, seq_len).
        y: Target token indices of shape (batch_size, seq_len).
        optimizer: Optimizer to step.

    Returns:
        Scalar cross-entropy loss for this batch.
    """
    model.train()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)

    logits = model(x)  # (B, T, V)
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
) -> dict[str, list[float]]:
    """Train for exactly max_steps gradient steps, cycling through train_loader.

    Args:
        model: The model to train.
        train_loader: DataLoader yielding (x, y) batches.
        optimizer: Optimizer.
        max_steps: Total number of gradient steps to take.
        log_interval: Print loss every this many steps. 0 = silent.

    Returns:
        Dict with keys:
            - "losses": List of per-step scalar losses (length == max_steps).
            - "perplexities": List of per-step perplexities (length == max_steps).
    """
    model.train()
    losses: list[float] = []
    perplexities: list[float] = []
    step = 0

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            x, y = batch
            loss = train_step(model, x, y, optimizer)
            perplexity = compute_perplexity(loss)
            losses.append(loss)
            perplexities.append(perplexity)
            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(f"step {step + 1:>5}/{max_steps}  loss={loss:.4f}  ppl={perplexity:.2f}")
            step += 1

    return {"losses": losses, "perplexities": perplexities}
