"""Profiling utilities for performance monitoring and bottleneck identification.

Provides:
- Timer context managers for measuring execution time
- Memory profiling for GPU/CPU usage
- PyTorch profiler integration
- Throughput metrics
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch


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
        """Initialize timer.

        Args:
            name: Optional name for the timer (used in context manager).
        """
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self) -> Timer:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        elapsed = self.stop()
        if self.name:
            print(f"[Timer] {self.name}: {elapsed:.4f}s")


@contextmanager
def timer(name: str = "") -> Generator[Timer, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name to display when timer completes.

    Yields:
        Timer instance with elapsed time.

    Example:
        with timer("forward pass"):
            output = model(input)
    """
    t = Timer(name)
    t.start()
    try:
        yield t
    finally:
        t.stop()
        if name:
            print(f"[Timer] {name}: {t.elapsed:.4f}s")


class GPUMemoryMonitor:
    """Monitor GPU memory usage.

    Usage:
        monitor = GPUMemoryMonitor()
        print(f"Allocated: {monitor.allocated_mb():.1f} MB")
        print(f"Reserved: {monitor.reserved_mb():.1f} MB")
        print(f"Peak: {monitor.peak_mb():.1f} MB")
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize GPU memory monitor.

        Args:
            device: CUDA device index (default: 0).
        """
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

    def allocated_mb(self) -> float:
        """Get currently allocated memory in MB.

        Returns:
            Allocated memory in megabytes.
        """
        return torch.cuda.memory_allocated(self.device) / 1024**2

    def reserved_mb(self) -> float:
        """Get currently reserved memory in MB.

        Returns:
            Reserved memory in megabytes.
        """
        return torch.cuda.memory_reserved(self.device) / 1024**2

    def peak_mb(self) -> float:
        """Get peak allocated memory in MB.

        Returns:
            Peak allocated memory in megabytes.
        """
        return torch.cuda.max_memory_allocated(self.device) / 1024**2

    def reset_peak(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def summary(self) -> dict[str, float]:
        """Get memory usage summary.

        Returns:
            Dict with keys: allocated_mb, reserved_mb, peak_mb.
        """
        return {
            "allocated_mb": self.allocated_mb(),
            "reserved_mb": self.reserved_mb(),
            "peak_mb": self.peak_mb(),
        }


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
        """Initialize throughput monitor."""
        self.total_items = 0
        self.start_time = time.perf_counter()
        self.count = 0

    def record(self, num_items: int) -> ThroughputContext:
        """Record processing of items.

        Args:
            num_items: Number of items processed.

        Returns:
            Context manager for timing.
        """
        return ThroughputContext(self, num_items)

    def throughput(self) -> float:
        """Calculate current throughput.

        Returns:
            Items per second.
        """
        elapsed = time.perf_counter() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.total_items / elapsed

    def reset(self) -> None:
        """Reset throughput statistics."""
        self.total_items = 0
        self.start_time = time.perf_counter()
        self.count = 0


class ThroughputContext:
    """Context manager for recording throughput."""

    def __init__(self, monitor: ThroughputMonitor, num_items: int) -> None:
        """Initialize context.

        Args:
            monitor: Parent ThroughputMonitor.
            num_items: Number of items being processed.
        """
        self.monitor = monitor
        self.num_items = num_items

    def __enter__(self) -> ThroughputContext:
        """Enter context."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and update monitor."""
        self.monitor.total_items += self.num_items
        self.monitor.count += 1


@contextmanager
def profile_pytorch(
    enabled: bool = True,
    use_cuda: bool = True,
    profile_memory: bool = True,
    record_shapes: bool = True,
    with_stack: bool = False,
) -> Generator[torch.profiler.profile, None, None]:
    """PyTorch profiler context manager.

    Args:
        enabled: Whether profiling is enabled.
        use_cuda: Profile CUDA operations.
        profile_memory: Profile memory usage.
        record_shapes: Record tensor shapes.
        with_stack: Record stack traces.

    Yields:
        PyTorch profiler instance.

    Example:
        with profile_pytorch() as prof:
            output = model(input)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        if enabled:
            yield prof
        else:
            yield prof


def log_model_size(model: torch.nn.Module) -> dict[str, Any]:
    """Log model size statistics.

    Args:
        model: PyTorch model.

    Returns:
        Dict with parameter counts and memory usage.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

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
