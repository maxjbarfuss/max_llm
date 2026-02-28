#!/usr/bin/env python3
"""Benchmark different attention backends: Flash Attention, Sage Attention, xFormers, Standard."""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.training.profiling import GPUMemoryMonitor, Timer


def benchmark_attention_backend(
    backend: str,
    d_model: int = 256,
    num_heads: int = 8,
    seq_len: int = 512,
    batch_size: int = 8,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
) -> dict:
    """Benchmark a single attention backend.

    Args:
        backend: Attention backend name ("flash", "sage", "xformers", "standard").
        d_model: Model dimension.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        batch_size: Batch size.
        num_warmup: Number of warmup iterations.
        num_iterations: Number of benchmark iterations.
        device: Device to run on ("cuda" or "cpu").

    Returns:
        Dictionary with benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend.upper()} Attention")
    print(f"{'='*60}")
    print(
        f"Config: d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}, batch_size={batch_size}"
    )

    # Create model
    attention = (
        CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.1,
            attention_backend=backend,
        )
        .to(device)
        .to(torch.bfloat16)
    )  # Use bf16 for Flash Attention and consistency

    actual_backend = attention.attention_backend
    print(f"Requested: {backend}, Actual backend: {actual_backend}")

    # Check if backend was downgraded
    if actual_backend != backend:
        print(f"⚠️  Backend not available, using fallback: {actual_backend}")

    # Create dummy input (use bf16 for Flash Attention compatibility)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...", end="", flush=True)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = attention(x)
    if device == "cuda":
        torch.cuda.synchronize()
    print(" ✓")

    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...", end="", flush=True)
    times = []
    memory_monitor = GPUMemoryMonitor(device) if device == "cuda" else None

    attention.eval()
    with torch.no_grad():
        for i in range(num_iterations):
            if memory_monitor:
                memory_monitor.reset_peak()

            with Timer() as timer:
                _ = attention(x)

            if device == "cuda":
                torch.cuda.synchronize()

            times.append(timer.elapsed * 1000)  # Convert to milliseconds

    print(" ✓")

    # Calculate statistics
    times_array = torch.tensor(times)
    times_ms = times_array.numpy()
    throughput_samples_per_sec = 1000 / times_array.mean().item()  # Samples per second
    throughput_tokens_per_sec = throughput_samples_per_sec * batch_size * seq_len

    results = {
        "backend": actual_backend,
        "time_ms_mean": times_array.mean().item(),
        "time_ms_std": times_array.std().item(),
        "time_ms_min": times_array.min().item(),
        "time_ms_max": times_array.max().item(),
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
    }

    if memory_monitor:
        results["peak_memory_mb"] = memory_monitor.peak_mb()
        print(f"\nMemory (Peak): {memory_monitor.peak_mb():.2f} MB")

    print(f"\nTime (ms):     {results['time_ms_mean']:.3f} ± {results['time_ms_std']:.3f}")
    print(f"Throughput:    {results['throughput_tokens_per_sec']:.0f} tokens/sec")

    return results


def main() -> None:
    """Run attention benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark different attention backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--num-iter", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument(
        "--backends",
        type=str,
        default="flash,sage,xformers,standard",
        help="Comma-separated list of backends to benchmark",
    )
    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    backends = args.backends.split(",")
    results = {}

    # Benchmark each backend
    for backend in backends:
        backend = backend.strip()
        try:
            result = benchmark_attention_backend(
                backend=backend,
                d_model=args.d_model,
                num_heads=args.num_heads,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iter,
                device=args.device,
            )
            results[backend] = result
        except Exception as e:
            print(f"❌ Error benchmarking {backend}: {e}")
            results[backend] = {"error": str(e)}

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Find baseline (standard attention)
    if "standard" in results and "error" not in results["standard"]:
        baseline_time = results["standard"]["time_ms_mean"]
    else:
        baseline_time = None

    print("\n{:<12} {:<15} {:<20} {:<15}".format("Backend", "Time (ms)", "Speedup", "Tokens/sec"))
    print("-" * 62)

    for backend in backends:
        backend = backend.strip()
        if backend not in results or "error" in results[backend]:
            status = "❌ Error" if "error" in (results.get(backend) or {}) else "❌ Not run"
            print(f"{backend:<12} {status:<15}")
        else:
            result = results[backend]
            time_ms = result["time_ms_mean"]
            tokens_per_sec = result["throughput_tokens_per_sec"]

            if baseline_time and backend != "standard":
                speedup = baseline_time / time_ms
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "baseline" if backend == "standard" else "N/A"

            print(f"{backend:<12} {time_ms:<15.3f} {speedup_str:<20} {tokens_per_sec:<15.0f}")

    print()


if __name__ == "__main__":
    main()
