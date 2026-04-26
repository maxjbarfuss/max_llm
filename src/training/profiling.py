"""Profiling utilities for performance monitoring and bottleneck identification.

Provides:
- Timer context managers for measuring execution time
- Memory profiling for GPU/CPU usage
- PyTorch profiler integration
- Throughput metrics
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_MB = 1024**2


class Timer:
    """Simple timer for measuring execution time.

    Usage:
        timer = Timer()
        timer.start()
        # ... code to time ...
        elapsed = timer.stop()
        print(f"Elapsed: {elapsed:.3f}s")

    Or as a context manager:
        with Timer("my_operation") as t:
            # ... code to time ...
        print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = self.stop()
        if self.name:
            print(f"[Timer] {self.name}: {elapsed:.4f}s")


class GPUMemoryMonitor:
    """Monitor GPU memory usage.

    Usage:
        monitor = GPUMemoryMonitor()
        print(f"Allocated: {monitor.allocated_mb():.1f} MB")
        print(f"Reserved: {monitor.reserved_mb():.1f} MB")
        print(f"Peak: {monitor.peak_mb():.1f} MB")
    """

    def __init__(self, device: int = 0) -> None:
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

    def allocated_mb(self) -> float:
        return torch.cuda.memory_allocated(self.device) / _MB

    def reserved_mb(self) -> float:
        return torch.cuda.memory_reserved(self.device) / _MB

    def peak_mb(self) -> float:
        return torch.cuda.max_memory_allocated(self.device) / _MB

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    def summary(self) -> dict[str, float]:
        """Return allocated, reserved, and peak memory in MB."""
        return {
            "allocated_mb": self.allocated_mb(),
            "reserved_mb": self.reserved_mb(),
            "peak_mb": self.peak_mb(),
        }


@dataclass
class ComponentProfileRow:
    """One optimizer-step component profile sample."""

    step: int
    epoch: int
    rank: int
    data_wait_ms: float = 0.0
    forward_backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    eval_ms: float = 0.0
    checkpoint_ms: float = 0.0
    benchmark_ms: float = 0.0
    step_ms: float = 0.0
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0
    reserved_memory_mb: float = 0.0


class ComponentProfiler:
    """Collect step-level time and memory samples for training components."""

    fieldnames = [
        "step",
        "epoch",
        "rank",
        "data_wait_ms",
        "forward_backward_ms",
        "optimizer_ms",
        "eval_ms",
        "checkpoint_ms",
        "benchmark_ms",
        "step_ms",
        "peak_memory_mb",
        "allocated_memory_mb",
        "reserved_memory_mb",
    ]

    def __init__(self, rank: int = 0, device: torch.device | None = None) -> None:
        self.rank = rank
        self.device = device
        self._current: ComponentProfileRow | None = None
        self.rows: list[ComponentProfileRow] = []

    def start_step(self, step: int, epoch: int, data_wait_ms: float = 0.0) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._current = ComponentProfileRow(
            step=step,
            epoch=epoch,
            rank=self.rank,
            data_wait_ms=data_wait_ms,
        )

    @property
    def current_row(self) -> ComponentProfileRow | None:
        return self._current

    @contextmanager
    def record(self, field: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._current is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                setattr(self._current, field, getattr(self._current, field) + elapsed_ms)

    def finish_step(self) -> ComponentProfileRow | None:
        if self._current is None:
            return None
        row = self._current
        row.step_ms = (
            row.data_wait_ms
            + row.forward_backward_ms
            + row.optimizer_ms
            + row.eval_ms
            + row.checkpoint_ms
            + row.benchmark_ms
        )
        if self.device is not None and self.device.type == "cuda":
            row.peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / _MB
            row.allocated_memory_mb = torch.cuda.memory_allocated(self.device) / _MB
            row.reserved_memory_mb = torch.cuda.memory_reserved(self.device) / _MB
        self.rows.append(row)
        self._current = None
        return row

    def write_csv(self, path: str | Path) -> None:
        import csv

        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "step": row.step,
                        "epoch": row.epoch,
                        "rank": row.rank,
                        "data_wait_ms": f"{row.data_wait_ms:.3f}",
                        "forward_backward_ms": f"{row.forward_backward_ms:.3f}",
                        "optimizer_ms": f"{row.optimizer_ms:.3f}",
                        "eval_ms": f"{row.eval_ms:.3f}",
                        "checkpoint_ms": f"{row.checkpoint_ms:.3f}",
                        "benchmark_ms": f"{row.benchmark_ms:.3f}",
                        "step_ms": f"{row.step_ms:.3f}",
                        "peak_memory_mb": f"{row.peak_memory_mb:.1f}",
                        "allocated_memory_mb": f"{row.allocated_memory_mb:.1f}",
                        "reserved_memory_mb": f"{row.reserved_memory_mb:.1f}",
                    }
                )


class ThroughputContext:
    """Context manager for recording a single throughput sample."""

    def __init__(self, monitor: "ThroughputMonitor", num_items: int) -> None:
        self.monitor = monitor
        self.num_items = num_items

    def __enter__(self) -> "ThroughputContext":
        return self

    def __exit__(self, *args: Any) -> None:
        self.monitor.total_items += self.num_items
        self.monitor.count += 1


class ThroughputMonitor:
    """Monitor throughput (items/second).

    Usage:
        monitor = ThroughputMonitor()
        for batch in data_loader:
            with monitor.record(batch_size=len(batch)):
                process(batch)
            if monitor.count % 10 == 0:
                print(f"Throughput: {monitor.throughput():.1f} items/s")
    """

    def __init__(self) -> None:
        self.total_items = 0
        self.start_time = time.perf_counter()
        self.count = 0

    def record(self, num_items: int) -> ThroughputContext:
        """Return a context manager that records processing of num_items."""
        return ThroughputContext(self, num_items)

    def throughput(self) -> float:
        """Return items per second since construction or last reset."""
        elapsed = time.perf_counter() - self.start_time
        return self.total_items / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        self.total_items = 0
        self.start_time = time.perf_counter()
        self.count = 0


def profile_pytorch(
    use_cuda: bool = True,
    profile_memory: bool = True,
    record_shapes: bool = True,
    with_stack: bool = False,
) -> "torch.profiler.profile":
    """Return a PyTorch profiler context manager.

    Args:
        use_cuda: Profile CUDA operations.
        profile_memory: Profile memory usage.
        record_shapes: Record tensor shapes.
        with_stack: Record stack traces.

    Example:
        with profile_pytorch() as prof:
            output = model(input)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )


def log_model_size(model: torch.nn.Module) -> dict[str, Any]:
    """Log model size statistics.

    Args:
        model: PyTorch model.

    Returns:
        Dict with parameter counts and memory usage.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / _MB

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_memory_mb,
        "total_params_millions": total_params / 1e6,
    }

    print(f"Model size: {stats['total_params_millions']:.2f}M params")
    print(f"Trainable: {trainable_params:,} / {total_params:,}")
    print(f"Parameter memory: {param_memory_mb:.1f} MB")

    return stats
