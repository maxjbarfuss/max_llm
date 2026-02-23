"""Training entrypoint for max-llm.

Usage:
    python train.py --config config/experiment.toml
"""

from __future__ import annotations

import argparse

from src.config.experiment import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a max-llm model")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)

    print(f"Experiment : {config.name}")
    print(f"Output dir : {config.output_dir}")
    print(
        f"Model      : hidden_size={config.model.hidden_size}, "
        f"vocab_size={config.model.vocab_size}, "
        f"layers={config.model.num_layers}, "
        f"heads={config.model.num_heads}"
    )
    print(
        f"Training   : max_steps={config.training.max_steps}, "
        f"lr={config.training.learning_rate}, "
        f"batch_size={config.training.batch_size}"
    )
    print("Config loaded successfully.")


if __name__ == "__main__":
    main()
