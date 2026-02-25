#!/usr/bin/env python3
"""Curriculum learning trainer: alternates between TinyStories and WikiText.

Starts with TinyStories (simpler, shorter sentences), then alternates every
N checkpoints to WikiText (more diverse, technical content).

Usage:
    python scripts/train_curriculum.py \
        --config config/experiment_curriculum.toml \
        --switch-interval 250 \
        --num-cycles 2
"""

from __future__ import annotations

import argparse
import logging

from src.config.config import load_config

from src.training.train import main as train_main

logger = logging.getLogger(__name__)


def alternate_dataset(config_path: str, step: int, switch_interval: int) -> tuple[str, int]:
    """Determine which dataset to use based on step and switch_interval.

    Args:
        config_path: Path to config file
        step: Current training step
        switch_interval: Steps between dataset switches

    Returns:
        (dataset_path, checkpoint_step_in_current_dataset)
    """
    cycle = step // switch_interval
    dataset_idx = cycle % 2

    if dataset_idx == 0:
        dataset_path = "data/fast/tinystories_5m_tokens__utf8.npy"
        dataset_name = "TinyStories"
    else:
        dataset_path = "data/fast/wikitext_5m_tokens__utf8.npy"
        dataset_name = "WikiText-103"

    steps_in_current_dataset = step % switch_interval
    logger.info(
        f"Step {step}: Dataset cycle {cycle}, using {dataset_name} "
        f"({steps_in_current_dataset}/{switch_interval} steps in current dataset)"
    )

    return dataset_path, steps_in_current_dataset


def run_curriculum_training(
    config_path: str,
    switch_interval: int = 250,
    num_cycles: int = 2,
) -> None:
    """Run curriculum training with dataset alternation.

    Args:
        config_path: Path to base experiment config (TOML)
        switch_interval: Steps between dataset switches
        num_cycles: Number of complete cycles (stories → wikitext → ...)
    """
    # Load base config
    config = load_config(config_path)
    original_dataset = config.data.dataset_path

    total_steps = switch_interval * 2 * num_cycles
    logger.info(f"Curriculum training: {num_cycles} cycles = {total_steps} total steps")
    logger.info(
        f"  Cycle: TinyStories ({switch_interval} steps) → WikiText ({switch_interval} steps)"
    )
    logger.info(f"  Starting with: {original_dataset}")

    # Estimate schedules
    checkpoint_interval = config.training.checkpoint_interval
    datasets = [
        "data/fast/tinystories_5m_tokens__utf8.npy",
        "data/fast/wikitext_5m_tokens__utf8.npy",
    ]

    # Log curriculum schedule
    logger.info("\nCurriculum Schedule:")
    for step in range(0, total_steps, switch_interval):
        dataset_path, _ = alternate_dataset(config_path, step, switch_interval)
        dataset_name = "TinyStories" if "tinystories" in dataset_path else "WikiText-103"
        logger.info(f"  Steps {step:5d}–{step + switch_interval:5d}: {dataset_name}")

    # For Phase 2 MVP: just log the curriculum and train once on combined data
    # Full curriculum would require checkpoint-based dataset swapping in training loop
    logger.info("\n[Phase 2 MVP] Running single training pass with initial dataset:")
    logger.info(f"  Dataset: {config.data.dataset_path}")
    logger.info(f"  Steps: {config.training.max_steps}")
    logger.info("  (Full curriculum swapping will arrive in Phase 4–5 with multi-dataset support)")

    # Run training
    train_main(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum learning trainer")
    parser.add_argument("--config", required=True, help="Path to experiment config (TOML)")
    parser.add_argument(
        "--switch-interval", type=int, default=250, help="Steps between dataset switches"
    )
    parser.add_argument("--num-cycles", type=int, default=2, help="Number of curriculum cycles")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s"
    )

    run_curriculum_training(
        config_path=args.config,
        switch_interval=args.switch_interval,
        num_cycles=args.num_cycles,
    )
