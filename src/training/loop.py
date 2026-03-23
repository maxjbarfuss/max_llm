"""Phase 2/3 training loop: forward → CE loss → backward → optimizer step.

Supports:
- Linear warmup + cosine LR decay
- Gradient clipping
- Mixed precision training (AMP)
- Gradient accumulation
- Enhanced logging (tokens/sec, GPU memory)
"""

import contextlib
import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.distributed import get_world_size


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
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute cross-entropy loss with optional label smoothing.

    Processes the (B*T, V) logit matrix in chunks to avoid materializing a
    full (B*T, V) softmax tensor in the autograd graph.  At B=24, T=2048,
    V=8192 this saves ~800 MB compared to a single F.cross_entropy call.

    Args:
        logits: Model output logits of shape (B, T, V).
        targets: Target token indices of shape (B, T).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
        chunk_size: Number of tokens per loss chunk (default 4096).

    Returns:
        Scalar loss value.
    """
    B, T, V = logits.shape
    logits_2d = logits.view(B * T, V)
    targets_1d = targets.view(B * T)

    loss_sum = logits_2d.new_zeros(())
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        if label_smoothing > 0:
            log_probs = F.log_softmax(logits_2d[start:end], dim=-1)
            tgt = targets_1d[start:end]
            nll = -log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            smooth = -log_probs.mean(dim=-1)
            loss_sum = loss_sum + ((1 - label_smoothing) * nll + label_smoothing * smooth).sum()
        else:
            loss_sum = loss_sum + F.cross_entropy(
                logits_2d[start:end], targets_1d[start:end], reduction="sum"
            )
    return loss_sum / (B * T)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    use_amp: bool = False,
    label_smoothing: float = 0.0,
    max_batches: int | None = None,
) -> float:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation dataset.
        use_amp: Whether to use automatic mixed precision.
        label_smoothing: Label smoothing factor for loss computation.
        max_batches: If set, stop after this many batches (for large val sets).

    Returns:
        Average loss over the dataset.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass with optional AMP
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = compute_loss_with_smoothing(logits, y, label_smoothing)
            else:
                logits = model(x)
                loss = compute_loss_with_smoothing(logits, y, label_smoothing)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            num_batches += 1

            if max_batches is not None and num_batches >= max_batches:
                break

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler | None = None,
    use_amp: bool = False,
    accumulate_grad: bool = False,
    label_smoothing: float = 0.0,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Single forward + backward step with optional AMP.

    Args:
        model: The model to train.
        x: Input token indices of shape (batch_size, seq_len).
        y: Target token indices of shape (batch_size, seq_len).
        optimizer: Optimizer to step.
        scaler: GradScaler for mixed precision (required if use_amp=True).
        use_amp: Whether to use automatic mixed precision.
        accumulate_grad: If True, skip zero_grad (for gradient accumulation).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
        gradient_accumulation_steps: Number of micro-steps per optimizer step.
            The loss is divided by this before backward so that accumulated
            gradients equal the full-batch gradient regardless of accum count.

    Returns:
        Scalar cross-entropy loss for this batch (unscaled, for logging).
    """
    model.train()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)

    # Zero gradients unless accumulating
    if not accumulate_grad:
        optimizer.zero_grad()

    # Forward pass with optional AMP
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda"):
            logits = model(x)  # (B, T, V)
            loss = compute_loss_with_smoothing(logits, y, label_smoothing)
    else:
        logits = model(x)  # (B, T, V)
        loss = compute_loss_with_smoothing(logits, y, label_smoothing)

    loss_value = loss.item()
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            "Non-finite loss detected before backward "
            f"(loss={loss_value}, label_smoothing={label_smoothing})."
        )

    # Scale loss before backward so accumulated gradients equal the full-batch
    # gradient.  Without this, accum_steps × too-large gradients cause the
    # gradient clip to fire too aggressively early in training, and once
    # gradients shrink below the clip threshold (mid-to-late training) the
    # optimizer takes accum_steps × too-large steps, causing oscillation.
    loss_scaled = loss / gradient_accumulation_steps

    # Backward pass with optional AMP scaling
    if use_amp and scaler is not None:
        scaler.scale(loss_scaled).backward()
    else:
        loss_scaled.backward()

    return loss_value  # Return unscaled loss for logging


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    model: nn.Module | None = None,
) -> float:
    """Perform optimizer step with optional gradient clipping and AMP unscaling.

    Args:
        optimizer: Optimizer to step.
        scaler: GradScaler for mixed precision (required if use_amp=True).
        use_amp: Whether to use automatic mixed precision.
        gradient_clip_norm: Max gradient norm for clipping (None = no clipping).
        model: Model for gradient clipping / norm logging.

    Returns:
        Gradient norm before clipping (0.0 if model is None).
    """
    grad_norm = 0.0

    # Unscale gradients if using AMP
    if use_amp and scaler is not None:
        scaler.unscale_(optimizer)

    # Compute grad norm (and optionally clip)
    if model is not None:
        clip = gradient_clip_norm if gradient_clip_norm is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Non-finite gradient norm detected before optimizer step "
                f"(grad_norm={grad_norm})."
            )

    # Optimizer step
    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return grad_norm


def train(  # noqa: C901
    model: nn.Module,
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
    tb_writer: Any | None = None,
    use_torch_compile: bool = False,
    checkpoint_interval: int = 0,
    checkpoint_fn: Any | None = None,
    eval_max_batches: int = 0,
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
    scaler: torch.GradScaler | None = None
    if use_amp:
        if device.type == "cuda":
            scaler = torch.GradScaler("cuda")
        else:
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
            "grad_norm",
            "tokens_per_sec",
            "gpu_memory_mb",
            "val_loss",
            "test_loss",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    eval_max_batches_opt: int | None = eval_max_batches if eval_max_batches > 0 else None

    step = 0
    micro_step = 0
    epoch = 0
    accumulated_loss = 0.0
    step_start_time = time.time()
    tokens_in_step = 0

    while step < max_steps and not should_stop_early:
        # Per-epoch setup: randomize TokenDataset sequence offsets so successive
        # passes see different document boundary cuts, and advance DistributedSampler
        # shuffle so each rank gets a different shard order each epoch.
        if hasattr(getattr(train_loader, "dataset", None), "set_epoch"):
            train_loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
        if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        for batch in train_loader:
            if step >= max_steps or should_stop_early:
                break

            x, y = batch
            if log_tokens_per_sec:
                tokens_in_step += x.numel()

            is_last_micro_step = (micro_step + 1) % gradient_accumulation_steps == 0
            is_first_micro_step = micro_step % gradient_accumulation_steps == 0
            accumulate_grad = not is_last_micro_step

            # Mark CUDA graph step begin only when using torch.compile, once per
            # accumulation cycle at the first micro-step. Calling this
            # unconditionally can deadlock DDP.
            if use_torch_compile and torch.cuda.is_available() and is_first_micro_step:
                torch.compiler.cudagraph_mark_step_begin()

            # For DDP: suppress gradient sync on all but the last micro-batch so
            # NCCL all_reduce fires once per optimizer step instead of N times.
            # model.no_sync() is a no-op for non-DDP models.
            if accumulate_grad and hasattr(model, "no_sync"):
                ctx = model.no_sync()  # type: ignore[operator]
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                loss = train_step(
                    model=model,
                    x=x,
                    y=y,
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=use_amp,
                    accumulate_grad=accumulate_grad,
                    label_smoothing=label_smoothing,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
            accumulated_loss += loss
            micro_step += 1

            # Optimizer step after accumulation
            if micro_step % gradient_accumulation_steps == 0:
                # Average loss over accumulated micro-batches
                avg_loss = accumulated_loss / gradient_accumulation_steps
                if not math.isfinite(avg_loss):
                    raise FloatingPointError(
                        "Non-finite averaged loss detected "
                        f"at step {step + 1}: avg_loss={avg_loss}."
                    )

                # Optimizer step with gradient clipping (returns grad norm)
                grad_norm = optimizer_step(
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
                    memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                    gpu_memory_list.append(memory_mb)

                # Evaluate on val/test sets if requested
                val_loss = None
                test_loss = None
                if eval_interval and (step + 1) % eval_interval == 0:
                    if val_loader is not None:
                        val_loss = evaluate(
                            model, val_loader, use_amp, label_smoothing, eval_max_batches_opt
                        )
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
                        test_loss = evaluate(
                            model, test_loader, use_amp, label_smoothing, eval_max_batches_opt
                        )
                        test_losses.append(test_loss)

                current_lr = optimizer.param_groups[0]["lr"]

                # Write to CSV (every step)
                if csv_writer is not None:
                    row: dict[str, Any] = {
                        "step": step + 1,
                        "loss": f"{avg_loss:.6f}",
                        "perplexity": f"{perplexity:.4f}",
                        "lr": f"{current_lr:.6e}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "tokens_per_sec": f"{tokens_per_sec:.0f}" if log_tokens_per_sec else "",
                        "gpu_memory_mb": (
                            f"{memory_mb:.1f}" if (log_gpu_memory and device.type == "cuda") else ""
                        ),
                        "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
                        "test_loss": f"{test_loss:.6f}" if test_loss is not None else "",
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()  # type: ignore[union-attr]

                # TensorBoard: log train metrics every step
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", avg_loss, step + 1)
                    tb_writer.add_scalar("train/perplexity", perplexity, step + 1)
                    tb_writer.add_scalar("train/lr", current_lr, step + 1)
                    tb_writer.add_scalar("train/grad_norm", grad_norm, step + 1)
                    if log_tokens_per_sec and tokens_per_sec > 0:
                        # In DDP mode, report total system throughput (all ranks combined)
                        world_size = get_world_size()
                        total_tokens_per_sec = tokens_per_sec * world_size
                        tb_writer.add_scalar("train/tokens_per_sec", total_tokens_per_sec, step + 1)
                    if log_gpu_memory and device.type == "cuda":
                        tb_writer.add_scalar("train/gpu_memory_mb", memory_mb, step + 1)
                    if val_loss is not None:
                        tb_writer.add_scalar("eval/val_loss", val_loss, step + 1)
                        tb_writer.add_scalar("eval/val_perplexity", math.exp(val_loss), step + 1)
                    if test_loss is not None:
                        tb_writer.add_scalar("eval/test_loss", test_loss, step + 1)

                # Logging
                if log_interval > 0 and (step + 1) % log_interval == 0:
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

                # Periodic checkpoint
                if (
                    checkpoint_interval > 0
                    and checkpoint_fn is not None
                    and (step + 1) % checkpoint_interval == 0
                ):
                    checkpoint_fn(step + 1, val_loss=val_loss)

                # Reset for next step
                accumulated_loss = 0.0
                tokens_in_step = 0
                step_start_time = time.time()
                step += 1

        epoch += 1

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
