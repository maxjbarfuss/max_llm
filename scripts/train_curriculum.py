#!/usr/bin/env python3
"""Curriculum schedule utility for config-driven experiments.

This script prints a curriculum schedule from one or more dataset paths, then
launches training with a provided experiment config.

Usage:
    python scripts/train_curriculum.py \
        --config config/milestones/<experiment>.toml \
        --datasets data/fast/dataset_a.npy data/fast/dataset_b.npy \
        --switch-interval 250 \
        --num-cycles 2
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.config.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


def dataset_for_step(
    dataset_paths: list[str], step: int, switch_interval: int
) -> tuple[str, int, int]:
    """Determine dataset assignment for a step in a round-robin curriculum.

    Args:
        step: Current training step
        switch_interval: Steps between dataset switches
        dataset_paths: Ordered dataset paths to cycle through

    Returns:
        (dataset_path, steps_in_current_dataset, cycle_index)
    """
    cycle = step // switch_interval
    dataset_idx = cycle % len(dataset_paths)
    dataset_path = dataset_paths[dataset_idx]

    steps_in_current_dataset = step % switch_interval
    dataset_name = Path(dataset_path).name
    logger.info(
        "Step %s: Dataset cycle %s, using %s (%s/%s steps in current dataset)",
        step,
        cycle,
        dataset_name,
        steps_in_current_dataset,
        switch_interval,
    )

    return dataset_path, steps_in_current_dataset, cycle


def run_curriculum_training(
    config_path: str,
    dataset_paths: list[str],
    switch_interval: int = 250,
    num_cycles: int = 2,
) -> None:
    """Run curriculum schedule preview and launch training.

    Args:
        config_path: Path to base experiment config (TOML)
        dataset_paths: Ordered dataset paths to alternate (round-robin)
        switch_interval: Steps between dataset switches
        num_cycles: Number of complete cycles through `dataset_paths`
    """
    if not dataset_paths:
        raise ValueError("dataset_paths cannot be empty")

    config = ExperimentConfig.from_toml(config_path)
    base_dataset = config.data.dataset_path

    total_steps = switch_interval * len(dataset_paths) * num_cycles
    logger.info(f"Curriculum training: {num_cycles} cycles = {total_steps} total steps")
    logger.info("  Dataset order:")
    for index, dataset_path in enumerate(dataset_paths, start=1):
        logger.info("    %s. %s", index, dataset_path)
    logger.info("  Base config dataset_path: %s", base_dataset)

    logger.info("\nCurriculum Schedule:")
    for step in range(0, total_steps, switch_interval):
        dataset_path, _, _ = dataset_for_step(dataset_paths, step, switch_interval)
        logger.info("  Steps %5d–%5d: %s", step, step + switch_interval, Path(dataset_path).name)

    logger.info("\nLaunching training with provided config (config-driven behavior):")
    logger.info("  Config: %s", config_path)
    logger.info("  Dataset in use: %s", base_dataset)
    logger.info(
        "  Note: this utility prints curriculum schedule; dataset switching within a single run "
        "requires trainer support for live dataset swaps."
    )

    subprocess.run(
        [sys.executable, "-m", "src.training.train", "--config", config_path],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum schedule utility")
    parser.add_argument("--config", required=True, help="Path to experiment config (TOML)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Optional ordered dataset paths for schedule preview. "
            "If omitted, uses dataset_path from the provided config."
        ),
    )
    parser.add_argument(
        "--switch-interval", type=int, default=250, help="Steps between dataset switches"
    )
    parser.add_argument("--num-cycles", type=int, default=2, help="Number of curriculum cycles")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s"
    )

    loaded_config = ExperimentConfig.from_toml(args.config)
    schedule_datasets = args.datasets or [loaded_config.data.dataset_path]

    run_curriculum_training(
        config_path=args.config,
        dataset_paths=schedule_datasets,
        switch_interval=args.switch_interval,
        num_cycles=args.num_cycles,
    )
