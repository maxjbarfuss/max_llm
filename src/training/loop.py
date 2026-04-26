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
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from src.training.distributed import get_rank, get_world_size
from src.training.profiling import ComponentProfiler


def _unwrap_module(model: nn.Module) -> nn.Module:
    """Return the underlying module from a DDP/FSDP wrapper if any."""
    inner = getattr(model, "module", model)
    assert isinstance(inner, nn.Module)
    return inner


def _chunk_lm_loss_fn(
    h_chunk: torch.Tensor,
    targets_chunk: torch.Tensor,
    weight: torch.Tensor,
    label_smoothing: float,
    z_loss_weight: float,
) -> torch.Tensor:
    """Project a hidden chunk through ``weight`` and return the summed
    CE (+ z_loss * z_loss_weight) over that chunk.

    Returning a scalar lets ``torch.utils.checkpoint`` discard the
    intermediate ``(B, t, V)`` logits between forward and backward.
    """
    logits = F.linear(h_chunk, weight)  # (B, t, V), no bias on lm_head
    B_, t_, V_ = logits.shape
    logits_2d = logits.reshape(B_ * t_, V_)
    targets_1d = targets_chunk.reshape(B_ * t_)
    if label_smoothing > 0:
        log_probs = F.log_softmax(logits_2d, dim=-1)
        nll = -log_probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        chunk_sum = ((1 - label_smoothing) * nll + label_smoothing * smooth).sum()
    else:
        chunk_sum = F.cross_entropy(logits_2d, targets_1d, reduction="sum")
    if z_loss_weight > 0.0:
        lse = torch.logsumexp(logits_2d, dim=-1)
        chunk_sum = chunk_sum + z_loss_weight * (lse * lse).sum()
    return chunk_sum


def compute_chunked_lm_loss(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    z_loss_weight: float = 0.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Memory-efficient chunked CE (+ Z-loss) over the LM-head projection.

    Splits along the time axis; each chunk's ``(B, chunk_size, V)`` logits
    are produced inside ``torch.utils.checkpoint`` so they are recomputed
    on backward instead of being kept alive across the full sequence.

    Numerically equivalent to ``compute_loss_with_smoothing(lm_head(hidden),
    targets, ...)`` for the same ``label_smoothing`` and ``z_loss_weight``.

    Args:
        hidden: ``(B, T, d_model)`` final hidden states (post final-norm).
        lm_head_weight: ``(vocab_size, d_model)`` LM-head weight (no bias).
        targets: ``(B, T)`` integer targets.
        label_smoothing: Label smoothing factor.
        z_loss_weight: PaLM-style logit-norm regularizer weight.
        chunk_size: Time-axis chunk size; auto-clamped to ``T``.

    Returns:
        Scalar mean loss (CE + z_loss).
    """
    assert hidden.dim() == 3, f"hidden must be (B, T, D), got {tuple(hidden.shape)}"
    assert targets.shape == hidden.shape[:2], (
        f"targets shape {tuple(targets.shape)} does not match hidden " f"{tuple(hidden.shape[:2])}"
    )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    B, T, _ = hidden.shape
    chunk_size = min(chunk_size, T)
    total = hidden.new_zeros(())
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        h_c = hidden[:, start:end, :]
        y_c = targets[:, start:end]
        # use_reentrant=False propagates autocast and avoids the legacy
        # reentrant-autograd path; required when inputs are leaf params.
        chunk_sum = checkpoint(
            _chunk_lm_loss_fn,
            h_c,
            y_c,
            lm_head_weight,
            label_smoothing,
            z_loss_weight,
            use_reentrant=False,
        )
        total = total + chunk_sum
    return total / (B * T)


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
    z_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Compute cross-entropy loss with optional label smoothing and Z-loss.

    Processes the (B*T, V) logit matrix in chunks to avoid materializing a
    full (B*T, V) softmax tensor in the autograd graph.  At B=24, T=2048,
    V=8192 this saves ~800 MB compared to a single F.cross_entropy call.

    Z-loss (PaLM): z_loss_weight * mean(logsumexp(logits, dim=-1)^2).
    Penalises a growing log-partition function, stabilising logit scale.
    Typical value: 1e-4.  Eval always uses z_loss_weight=0 so val_loss
    reflects pure cross-entropy for perplexity comparison.

    Args:
        logits: Model output logits of shape (B, T, V).
        targets: Target token indices of shape (B, T).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
        chunk_size: Number of tokens per loss chunk (default 4096).
        z_loss_weight: Weight for PaLM Z-loss regularizer (0.0 = disabled).

    Returns:
        Scalar loss value (CE + z_loss).
    """
    B, T, V = logits.shape
    logits_2d = logits.view(B * T, V)
    targets_1d = targets.view(B * T)

    loss_sum = logits_2d.new_zeros(())
    z_loss_sum = logits_2d.new_zeros(()) if z_loss_weight > 0.0 else None
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        chunk = logits_2d[start:end]
        if label_smoothing > 0:
            log_probs = F.log_softmax(chunk, dim=-1)
            tgt = targets_1d[start:end]
            nll = -log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            smooth = -log_probs.mean(dim=-1)
            loss_sum = loss_sum + ((1 - label_smoothing) * nll + label_smoothing * smooth).sum()
        else:
            loss_sum = loss_sum + F.cross_entropy(chunk, targets_1d[start:end], reduction="sum")
        if z_loss_sum is not None:
            lse = torch.logsumexp(chunk, dim=-1)  # (chunk_tokens,)
            z_loss_sum = z_loss_sum + (lse * lse).sum()
    ce_loss = loss_sum / (B * T)
    if z_loss_sum is not None:
        return ce_loss + z_loss_weight * z_loss_sum / (B * T)
    return ce_loss


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
    z_loss_weight: float = 0.0,
    use_chunked_loss: bool = False,
    loss_chunk_size: int = 256,
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
        z_loss_weight: PaLM Z-loss regularizer weight (0.0 = disabled).
        use_chunked_loss: If True, use the chunked LM-head / CE path that
            never materializes the full ``(B, T, vocab)`` logits tensor.
            Training-only — eval still uses the standard path.
        loss_chunk_size: Time-axis chunk size for the chunked LM-head path.

    Returns:
        Scalar loss for this batch (CE + z_loss, unscaled, for logging).
    """
    model.train()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)

    # Zero gradients unless accumulating
    if not accumulate_grad:
        optimizer.zero_grad()

    autocast_ctx = (
        torch.autocast(device_type="cuda")
        if (use_amp and device.type == "cuda")
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        if use_chunked_loss:
            inner = _unwrap_module(model)
            if not hasattr(inner, "forward_hidden"):
                raise AttributeError(
                    "use_chunked_loss=True requires the model to expose a "
                    "forward_hidden(x) method that returns pre-LM-head hidden states."
                )
            hidden = inner.forward_hidden(x)  # type: ignore[operator]
            lm_head = inner.lm_head
            assert (
                isinstance(lm_head, nn.Linear) and lm_head.bias is None
            ), "Chunked LM-head loss requires lm_head to be nn.Linear(bias=False)."
            loss = compute_chunked_lm_loss(
                hidden,
                lm_head.weight,
                y,
                label_smoothing=label_smoothing,
                z_loss_weight=z_loss_weight,
                chunk_size=loss_chunk_size,
            )
        else:
            logits = model(x)  # (B, T, V)
            loss = compute_loss_with_smoothing(
                logits, y, label_smoothing, z_loss_weight=z_loss_weight
            )

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
            if use_amp and scaler is not None:
                scaler.update()  # reset scaler state so next unscale_() is valid
            return grad_norm  # caller manages consecutive-bad-step policy

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
    z_loss_weight: float = 0.0,
    use_chunked_loss: bool = False,
    loss_chunk_size: int = 256,
    tb_writer: Any | None = None,
    use_torch_compile: bool = False,
    checkpoint_interval: int = 0,
    checkpoint_fn: Any | None = None,
    eval_max_batches: int = 0,
    benchmark_interval: int = 0,
    benchmark_fn: Any | None = None,
    component_profile_path: str | Path | None = None,
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
    start_time = step_start_time
    tokens_in_step = 0
    consecutive_bad_steps = 0
    component_profiler: ComponentProfiler | None = None
    if component_profile_path is not None:
        component_profiler = ComponentProfiler(rank=get_rank(), device=device)

    while step < max_steps and not should_stop_early:
        # Per-epoch setup: randomize TokenDataset sequence offsets so successive
        # passes see different document boundary cuts, and advance DistributedSampler
        # shuffle so each rank gets a different shard order each epoch.
        if hasattr(getattr(train_loader, "dataset", None), "set_epoch"):
            train_loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
        if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        data_wait_start = time.perf_counter()
        for batch in train_loader:
            if step >= max_steps or should_stop_early:
                break

            x, y = batch
            if log_tokens_per_sec:
                tokens_in_step += x.numel()

            is_last_micro_step = (micro_step + 1) % gradient_accumulation_steps == 0
            is_first_micro_step = micro_step % gradient_accumulation_steps == 0
            accumulate_grad = not is_last_micro_step

            if component_profiler is not None:
                data_wait_ms = (time.perf_counter() - data_wait_start) * 1000
                if is_first_micro_step:
                    component_profiler.start_step(step + 1, epoch, data_wait_ms=data_wait_ms)
                elif component_profiler.current_row is not None:
                    component_profiler.current_row.data_wait_ms += data_wait_ms

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
                profile_ctx = (
                    component_profiler.record("forward_backward_ms")
                    if component_profiler is not None
                    else contextlib.nullcontext()
                )
                with profile_ctx:
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
                        z_loss_weight=z_loss_weight,
                        use_chunked_loss=use_chunked_loss,
                        loss_chunk_size=loss_chunk_size,
                    )
            accumulated_loss += loss
            micro_step += 1

            # Optimizer step after accumulation
            if micro_step % gradient_accumulation_steps == 0:
                # Average loss over accumulated micro-batches
                avg_loss = accumulated_loss / gradient_accumulation_steps

                # Optimizer step with gradient clipping (returns inf grad_norm if non-finite)
                profile_ctx = (
                    component_profiler.record("optimizer_ms")
                    if component_profiler is not None
                    else contextlib.nullcontext()
                )
                with profile_ctx:
                    grad_norm = optimizer_step(
                        optimizer=optimizer,
                        scaler=scaler,
                        use_amp=use_amp,
                        gradient_clip_norm=gradient_clip_norm,
                        model=model,
                    )

                    # LR scheduler always steps to keep schedule aligned with step count.
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                # Skip isolated bad steps; abort on 3 consecutive non-finite values
                if not math.isfinite(avg_loss) or not math.isfinite(grad_norm):
                    consecutive_bad_steps += 1
                    print(
                        f"WARNING: non-finite values at step {step + 1} "
                        f"(loss={avg_loss:.4g}, grad_norm={grad_norm:.4g}); "
                        f"skipping update ({consecutive_bad_steps}/3)."
                    )
                    if consecutive_bad_steps >= 3:
                        raise FloatingPointError(
                            f"3 consecutive non-finite steps ending at step {step + 1}; "
                            f"last: loss={avg_loss:.4g}, grad_norm={grad_norm:.4g}."
                        )
                    accumulated_loss = 0.0
                    tokens_in_step = 0
                    step_start_time = time.time()
                    step += 1
                    if component_profiler is not None:
                        component_profiler.finish_step()
                    data_wait_start = time.perf_counter()
                    continue  # skip logging/checkpointing; outer reset block bypassed

                consecutive_bad_steps = 0

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
                        profile_ctx = (
                            component_profiler.record("eval_ms")
                            if component_profiler is not None
                            else contextlib.nullcontext()
                        )
                        with profile_ctx:
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
                        profile_ctx = (
                            component_profiler.record("eval_ms")
                            if component_profiler is not None
                            else contextlib.nullcontext()
                        )
                        with profile_ctx:
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
                    # Estimate ETA and remaining steps/time
                    steps_done = step + 1
                    steps_left = max_steps - steps_done
                    elapsed_total = time.time() - (
                        step_start_time if steps_done == 1 else start_time
                    )
                    avg_step_time = elapsed_total / steps_done if steps_done > 0 else 0.0
                    eta_seconds = int(avg_step_time * steps_left)
                    eta_h = eta_seconds // 3600
                    eta_m = (eta_seconds % 3600) // 60
                    eta_s = eta_seconds % 60
                    eta_str = f"{eta_h:02}:{eta_m:02}:{eta_s:02}"
                    log_msg = (
                        f"step {steps_done:>5}/{max_steps}  "
                        f"loss={avg_loss:.4f}  ppl={perplexity:.2f}  "
                        f"lr={current_lr:.2e}  "
                        f"ETA={eta_str}  left={steps_left}"
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
                    profile_ctx = (
                        component_profiler.record("checkpoint_ms")
                        if component_profiler is not None
                        else contextlib.nullcontext()
                    )
                    with profile_ctx:
                        checkpoint_fn(step + 1, val_loss=val_loss)

                # Optional external benchmark callback (for MCQ harness, etc.)
                if (
                    benchmark_interval > 0
                    and benchmark_fn is not None
                    and (step + 1) % benchmark_interval == 0
                ):
                    profile_ctx = (
                        component_profiler.record("benchmark_ms")
                        if component_profiler is not None
                        else contextlib.nullcontext()
                    )
                    with profile_ctx:
                        benchmark_fn(step + 1)

                if component_profiler is not None:
                    component_profiler.finish_step()

                # Reset for next step
                accumulated_loss = 0.0
                tokens_in_step = 0
                step_start_time = time.time()
                step += 1
                data_wait_start = time.perf_counter()

        epoch += 1

    if csv_file is not None:
        csv_file.close()
    if component_profiler is not None:
        assert component_profile_path is not None
        component_profiler.write_csv(component_profile_path)

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
