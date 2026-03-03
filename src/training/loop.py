"""Phase 2/3 training loop: forward → CE loss → backward → optimizer step.

Supports:
- Linear warmup + cosine LR decay
- Gradient clipping
- Mixed precision training (AMP)
- Gradient accumulation
- Enhanced logging (tokens/sec, GPU memory)
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

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


def compute_loss_with_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute cross-entropy loss with optional label smoothing.

    Args:
        logits: Model output logits of shape (B, T, V).
        targets: Target token indices of shape (B, T).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).

    Returns:
        Scalar loss value.
    """
    B, T, V = logits.shape
    if label_smoothing > 0:
        # Label smoothing: (1 - ε) * one_hot + ε / vocab_size
        log_probs = F.log_softmax(logits.view(B * T, V), dim=-1)
        targets_flat = targets.view(-1)

        # NLL loss for correct class
        nll_loss = -log_probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)

        # Uniform distribution loss
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine
        loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
        return loss.mean()
    else:
        # Standard cross entropy
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


def evaluate(
    model: BaseLearningModel,
    dataloader: DataLoader,
    use_amp: bool = False,
    label_smoothing: float = 0.0,
) -> float:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation dataset.
        use_amp: Whether to use automatic mixed precision.
        label_smoothing: Label smoothing factor for loss computation.

    Returns:
        Average loss over the dataset.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass with optional AMP
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = compute_loss_with_smoothing(logits, y, label_smoothing)
            else:
                logits = model(x)
                loss = compute_loss_with_smoothing(logits, y, label_smoothing)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train_step(
    model: BaseLearningModel,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    accumulate_grad: bool = False,
    label_smoothing: float = 0.0,
) -> float:
    """Single forward + backward step with optional AMP and gradient clipping.

    Args:
        model: The model to train.
        x: Input token indices of shape (batch_size, seq_len).
        y: Target token indices of shape (batch_size, seq_len).
        optimizer: Optimizer to step.
        scaler: GradScaler for mixed precision (required if use_amp=True).
        use_amp: Whether to use automatic mixed precision.
        gradient_clip_norm: Max gradient norm for clipping (None = no clipping).
        accumulate_grad: If True, skip zero_grad (for gradient accumulation).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).

    Returns:
        Scalar cross-entropy loss for this batch.
    """
    model.train()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)

    # Zero gradients unless accumulating
    if not accumulate_grad:
        optimizer.zero_grad()

    # Forward pass with optional AMP
    if use_amp and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda"):
            logits = model(x)  # (B, T, V)
            loss = compute_loss_with_smoothing(logits, y, label_smoothing)
    else:
        logits = model(x)  # (B, T, V)
        loss = compute_loss_with_smoothing(logits, y, label_smoothing)

    # Backward pass with optional AMP scaling
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    model: BaseLearningModel | None = None,
) -> None:
    """Perform optimizer step with optional gradient clipping and AMP unscaling.

    Args:
        optimizer: Optimizer to step.
        scaler: GradScaler for mixed precision (required if use_amp=True).
        use_amp: Whether to use automatic mixed precision.
        gradient_clip_norm: Max gradient norm for clipping (None = no clipping).
        model: Model for gradient clipping (required if gradient_clip_norm is set).
    """
    # Unscale gradients if using AMP
    if use_amp and scaler is not None:
        scaler.unscale_(optimizer)

    # Gradient clipping
    if gradient_clip_norm is not None and model is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

    # Optimizer step
    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def train(  # noqa: C901
    model: BaseLearningModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    log_interval: int = 10,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float | None = None,
    use_amp: bool = False,
    lr_scheduler: Any | None = None,
    log_tokens_per_sec: bool = False,
    log_gpu_memory: bool = False,
    csv_log_path: str | Path | None = None,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    eval_interval: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    label_smoothing: float = 0.0,
) -> dict[str, list[float]]:
    """Train for exactly max_steps gradient steps with modern training features.

    Supports:
        - Gradient accumulation for larger effective batch sizes
        - Mixed precision training (AMP)
        - Gradient clipping
        - Learning rate scheduling
        - Throughput logging (tokens/sec)
        - GPU memory logging
        - Validation evaluation
        - Test set evaluation
        - Early stopping
        - Label smoothing

    Args:
        model: The model to train.
        train_loader: DataLoader yielding (x, y) batches.
        optimizer: Optimizer.
        max_steps: Total number of gradient steps to take.
        log_interval: Print loss every this many steps. 0 = silent.
        gradient_accumulation_steps: Accumulate gradients over this many micro-batches.
        gradient_clip_norm: Max gradient norm for clipping (None = no clipping).
        use_amp: Whether to use automatic mixed precision.
        lr_scheduler: Learning rate scheduler to step after each optimizer step.
        log_tokens_per_sec: Log throughput in tokens/sec.
        log_gpu_memory: Log GPU memory usage.
        csv_log_path: If set, write per-step metrics to this CSV file.
            Columns: step, loss, perplexity, lr, tokens_per_sec (if available), val_loss, test_loss.
        val_loader: Optional validation DataLoader for periodic evaluation.
        test_loader: Optional test DataLoader for periodic evaluation.
        eval_interval: Evaluate on val/test every this many steps (required if val/test loaders provided).
        early_stopping_patience: Stop training if val loss doesn't improve for this many evals (None = disabled).
        early_stopping_min_delta: Minimum improvement to reset early stopping counter.
        label_smoothing: Label smoothing factor (0.0 = no smoothing).

    Returns:
        Dict with keys:
            - "losses": List of per-step scalar losses (length == max_steps).
            - "perplexities": List of per-step perplexities (length == max_steps).
            - "tokens_per_sec": (optional) List of tokens/sec per step if log_tokens_per_sec=True.
            - "gpu_memory_mb": (optional) List of GPU memory in MB if log_gpu_memory=True.
            - "val_losses": (optional) List of validation losses at eval_interval.
            - "test_losses": (optional) List of test losses at eval_interval.
    """
    model.train()
    device = next(model.parameters()).device

    # Initialize GradScaler for AMP if needed
    scaler: torch.amp.GradScaler | None = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    if use_amp and device.type != "cuda":
        print("Warning: AMP requested but CUDA not available. Falling back to FP32.")
        use_amp = False

    losses: list[float] = []
    perplexities: list[float] = []
    tokens_per_sec_list: list[float] = []
    gpu_memory_list: list[float] = []
    val_losses: list[float] = []
    test_losses: list[float] = []

    # Early stopping tracking
    best_val_loss = float("inf")
    early_stopping_counter = 0
    should_stop_early = False

    # Open CSV log file if requested
    csv_file = None
    csv_writer = None
    if csv_log_path is not None:
        csv_path = Path(csv_log_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
        fieldnames = [
            "step",
            "loss",
            "perplexity",
            "lr",
            "tokens_per_sec",
            "gpu_memory_mb",
            "val_loss",
            "test_loss",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    step = 0
    micro_step = 0
    accumulated_loss = 0.0
    step_start_time = time.time()
    tokens_in_step = 0

    while step < max_steps and not should_stop_early:
        for batch in train_loader:
            if step >= max_steps or should_stop_early:
                break

            x, y = batch
            batch_tokens = x.numel()
            tokens_in_step += batch_tokens

            # Forward + backward
            accumulate_grad = micro_step % gradient_accumulation_steps != 0
            loss = train_step(
                model=model,
                x=x,
                y=y,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=gradient_clip_norm if not accumulate_grad else None,
                accumulate_grad=accumulate_grad,
                label_smoothing=label_smoothing,
            )
            accumulated_loss += loss
            micro_step += 1

            # Optimizer step after accumulation
            if micro_step % gradient_accumulation_steps == 0:
                # Average loss over accumulated micro-batches
                avg_loss = accumulated_loss / gradient_accumulation_steps

                # Optimizer step with gradient clipping
                optimizer_step(
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=use_amp,
                    gradient_clip_norm=gradient_clip_norm,
                    model=model,
                )

                # LR scheduler step
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Compute metrics
                perplexity = compute_perplexity(avg_loss)
                losses.append(avg_loss)
                perplexities.append(perplexity)

                # Compute tokens/sec if requested
                tokens_per_sec = 0.0
                if log_tokens_per_sec:
                    elapsed = time.time() - step_start_time
                    tokens_per_sec = tokens_in_step / max(elapsed, 1e-6)
                    tokens_per_sec_list.append(tokens_per_sec)

                # Log GPU memory if requested
                memory_mb = 0.0
                if log_gpu_memory and device.type == "cuda":
                    memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    gpu_memory_list.append(memory_mb)

                # Evaluate on val/test sets if requested
                val_loss = None
                test_loss = None
                if eval_interval and (step + 1) % eval_interval == 0:
                    if val_loader is not None:
                        val_loss = evaluate(model, val_loader, use_amp, label_smoothing)
                        val_losses.append(val_loss)

                        # Early stopping check
                        if early_stopping_patience is not None:
                            if val_loss < (best_val_loss - early_stopping_min_delta):
                                best_val_loss = val_loss
                                early_stopping_counter = 0
                                print(f"✓ Val loss improved to {val_loss:.4f}")
                            else:
                                early_stopping_counter += 1
                                print(
                                    f"✗ No improvement ({early_stopping_counter}/{early_stopping_patience})"
                                )

                            if early_stopping_counter >= early_stopping_patience:
                                print(
                                    f"\n🛑 Early stopping triggered after {early_stopping_patience} evals without improvement"
                                )
                                should_stop_early = True

                    if test_loader is not None:
                        test_loss = evaluate(model, test_loader, use_amp, label_smoothing)
                        test_losses.append(test_loss)

                # Write to CSV (every step)
                if csv_writer is not None:
                    current_lr = optimizer.param_groups[0]["lr"]
                    row: dict[str, Any] = {
                        "step": step + 1,
                        "loss": f"{avg_loss:.6f}",
                        "perplexity": f"{perplexity:.4f}",
                        "lr": f"{current_lr:.6e}",
                        "tokens_per_sec": f"{tokens_per_sec:.0f}" if log_tokens_per_sec else "",
                        "gpu_memory_mb": (
                            f"{memory_mb:.1f}" if (log_gpu_memory and device.type == "cuda") else ""
                        ),
                        "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
                        "test_loss": f"{test_loss:.6f}" if test_loss is not None else "",
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()  # type: ignore[union-attr]

                # Logging
                if log_interval > 0 and (step + 1) % log_interval == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_msg = (
                        f"step {step + 1:>5}/{max_steps}  "
                        f"loss={avg_loss:.4f}  ppl={perplexity:.2f}  "
                        f"lr={current_lr:.2e}"
                    )
                    if log_tokens_per_sec:
                        log_msg += f"  tokens/s={tokens_per_sec:.0f}"
                    if log_gpu_memory and device.type == "cuda":
                        log_msg += f"  mem={memory_mb:.0f}MB"
                    if val_loss is not None:
                        log_msg += f"  val={val_loss:.4f}"
                    if test_loss is not None:
                        log_msg += f"  test={test_loss:.4f}"
                    print(log_msg)

                # Reset for next step
                accumulated_loss = 0.0
                tokens_in_step = 0
                step_start_time = time.time()
                step += 1

    if csv_file is not None:
        csv_file.close()

    result = {"losses": losses, "perplexities": perplexities}
    if log_tokens_per_sec:
        result["tokens_per_sec"] = tokens_per_sec_list
    if log_gpu_memory:
        result["gpu_memory_mb"] = gpu_memory_list
    if val_losses:
        result["val_losses"] = val_losses
    if test_losses:
        result["test_losses"] = test_losses
    return result
