#!/usr/bin/env python
"""
Phase 3 Optimization Experiment Runner
Runs multiple configurations systematically and compares results.

Usage:
    python scripts/run_p3_experiments.py [--experiments exp1,exp2] [--distributed]

Examples:
    # Run all experiments
    python scripts/run_p3_experiments.py

    # Run specific experiments (comma-separated)
    python scripts/run_p3_experiments.py --experiments 512h_8l,50m_data

    # Run with multi-GPU DDP
    python scripts/run_p3_experiments.py --distributed

    # Run with profiling
    python scripts/run_p3_experiments.py --profile
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Experiment configurations to try
EXPERIMENTS = {
    "baseline": {
        "config": "config/milestones/p3_bpe_convergence.toml",
        "name": "Sealed Baseline (256H x 4L, 10M data)",
        "specs": {
            "model": "256H x 4L",
            "vocab": "50K",
            "data": "10M interleaved",
            "batch": "16 (8 x 2)",
            "steps": "5K",
        },
        "expected_loss": 4.31,
        "gpu_hours": "~6",  # Single GPU H100
    },
    "512h_8l": {
        "config": "config/milestones/p3_optimized_512h_8l.toml",
        "name": "Larger Model (512H x 8L, 10M data)",
        "specs": {
            "model": "512H x 8L",
            "vocab": "50K",
            "data": "10M interleaved",
            "batch": "16 (8 x 2)",
            "steps": "5K",
        },
        "expected_loss": "3.8-4.1?",
        "gpu_hours": "~25",
    },
    "50m_data": {
        "config": "config/milestones/p3_optimized_50m_data.toml",
        "name": "More Data (256H x 4L, 50M data)",
        "specs": {
            "model": "256H x 4L",
            "vocab": "50K",
            "data": "50M interleaved",
            "batch": "16 (8 x 2)",
            "steps": "10K",
        },
        "expected_loss": "3.8-4.1?",
        "gpu_hours": "~30",
    },
    "512h_50m": {
        "config": "config/milestones/p3_optimized_512h_50m.toml",
        "name": "Large Model + More Data (512H x 8L, 50M data)",
        "specs": {
            "model": "512H x 8L",
            "vocab": "50K",
            "data": "50M interleaved",
            "batch": "32 (8 x 4)",
            "steps": "20K",
        },
        "expected_loss": "3.5-3.9?",
        "gpu_hours": "~100+",
    },
}


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 80
    print("\n" + "=" * width)
    print(f" {title.center(width-2)} ")
    print("=" * width)


def run_experiment(
    exp_key: str,
    exp_config: dict[str, Any],
    distributed: bool = False,
    profile: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single experiment and return results.

    Args:
        exp_key: Experiment identifier (e.g., "512h_8l")
        exp_config: Experiment configuration dict
        distributed: Whether to use multi-GPU DDP
        profile: Whether to enable profiling
        dry_run: Print command without executing

    Returns:
        Dict with experiment results: start_time, end_time, loss, duration, status
    """
    result: dict[str, Any] = {"key": exp_key, "name": exp_config["name"]}

    print_header(f"Experiment: {exp_key}")
    print(f"\n📋 {exp_config['name']}")
    print(f"   Model: {exp_config['specs']['model']}")
    print(f"   Data: {exp_config['specs']['data']}")
    print(f"   Steps: {exp_config['specs']['steps']}")
    print(f"   Est. GPU hours: {exp_config['gpu_hours']}")
    print(f"   Expected loss: {exp_config['expected_loss']}")

    config_path = exp_config["config"]
    if not Path(config_path).exists():
        print(f"\n❌ Config not found: {config_path}")
        result["status"] = "failed"
        result["error"] = "config_not_found"
        return result

    # Build command
    cmd = ["python", "-m", "src.training.train", "--config", config_path]
    if distributed:
        cmd = (
            ["python", "-m", "torch.distributed.launch", "--nproc_per_node=2"]
            + cmd
            + ["--distributed"]
        )
    if profile:
        cmd.append("--profile")

    cmd_str = " ".join(cmd)
    print(f"\n🚀 Command:\n   {cmd_str}\n")

    if dry_run:
        print("   [DRY RUN - not executing]")
        result["status"] = "dry_run"
        return result

    result["start_time"] = datetime.now().isoformat()
    start = time.time()

    try:
        # Run the training
        process = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start

        result["status"] = "completed"
        result["end_time"] = datetime.now().isoformat()
        result["duration_seconds"] = elapsed
        result["duration_hours"] = elapsed / 3600

        print(f"\n✅ Experiment {exp_key} completed in {elapsed/3600:.2f} GPU hours")

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        result["status"] = "failed"
        result["error"] = str(e)
        result["duration_seconds"] = elapsed
        print(f"\n❌ Experiment {exp_key} failed: {e}")

    return result


def extract_final_metrics(output_dir: str | Path) -> dict[str, Any] | None:
    """Extract final loss from loss_curve.csv if it exists."""
    csv_path = Path(output_dir) / "loss_curve.csv"
    if not csv_path.exists():
        return None

    try:
        metrics = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                metrics["final_step"] = int(last_row["step"])
                metrics["final_loss"] = float(last_row["loss"])
                metrics["final_ppl"] = float(last_row["perplexity"])
                if last_row.get("val_loss"):
                    metrics["final_val_loss"] = float(last_row["val_loss"])
        return metrics if metrics else None
    except Exception as e:
        print(f"   ⚠️  Could not parse metrics: {e}")
        return None


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a comparison table of all experiments."""
    print_header("Experiment Comparison")

    # Print table header
    print(f"\n{'Experiment':<20} {'Config':<25} {'Status':<12} {'Time (h)':<10} {'Metrics':<30}")
    print("-" * 100)

    # Print rows
    for exp_key, exp_cfg in EXPERIMENTS.items():
        result = next((r for r in results if r["key"] == exp_key), None)
        if not result:
            status = "not_run"
            time_str = "—"
            metrics_str = "—"
        else:
            status = result["status"]
            time_str = (
                f"{result.get('duration_hours', 0):.1f}" if "duration_hours" in result else "—"
            )
            metrics_str = (
                f"loss={result.get('metrics', {}).get('final_loss', '?'):.3f}"
                if result.get("metrics")
                else "—"
            )

        config_short = exp_cfg["config"].split("/")[-1]
        print(f"{exp_key:<20} {config_short:<25} {status:<12} {time_str:<10} {metrics_str:<30}")

    print("\n" + "-" * 100)


def save_results_json(results: list[dict[str, Any]], output_file: str | Path):
    """Save detailed results to JSON."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        help="Comma-separated list of experiments to run (e.g., baseline,512h_8l)",
        default="baseline,512h_8l,50m_data",
    )
    parser.add_argument("--distributed", action="store_true", help="Use multi-GPU DDP")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # Parse experiments to run
    exp_keys = [e.strip() for e in args.experiments.split(",")]
    print_header("Phase 3 Optimization Experiments")
    print(f"\n🎯 Running {len(exp_keys)} experiment(s): {', '.join(exp_keys)}")

    # Validate experiments exist
    for key in exp_keys:
        if key not in EXPERIMENTS:
            print(f"❌ Unknown experiment: {key}")
            print(f"   Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)

    # Run experiments
    results: list[dict[str, Any]] = []
    for exp_key in exp_keys:
        exp_cfg = EXPERIMENTS[exp_key]
        result = run_experiment(
            exp_key,
            exp_cfg,
            distributed=args.distributed,
            profile=args.profile,
            dry_run=args.dry_run,
        )

        # Try to extract metrics if completed
        if result["status"] == "completed":
            output_dir = (
                exp_cfg["config"].replace("config/milestones/", "outputs/").replace(".toml", "")
            )
            metrics = extract_final_metrics(output_dir)
            if metrics:
                result["metrics"] = metrics

        results.append(result)

    # Print comparison
    print_comparison_table(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"outputs/p3_experiments_results_{timestamp}.json"
    save_results_json(results, results_file)

    print_header("Summary")
    print("\n✓ Experiment run complete")
    print(f"  Results saved to: {results_file}")
    print("\n  Next steps:")
    print("    1. Review loss_curve.csv in each output directory")
    print("    2. Compare convergence speed and final loss")
    print("    3. Run inference on best checkpoint")
    print("    4. Decide on next round of experiments")


if __name__ == "__main__":
    main()
