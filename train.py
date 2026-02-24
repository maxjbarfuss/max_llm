"""Training entrypoint for max-llm.

Usage:
    python train.py --config config/experiment.toml
"""

from __future__ import annotations

import argparse

import torch
from src.data.loader import load_corpus_text, make_data_loaders

from src.config.experiment import ExperimentConfig
from src.models.learning_model import SimpleLM
from src.tokenizer import TokenizerFactory
from src.training.loop import train


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
    corpus_text = load_corpus_text(config.data.dataset_path)
    tokenizer = TokenizerFactory.create(config.data.tokenizer_name)
    tokens = tokenizer.encode(corpus_text)

    if len(tokens) < config.data.max_length + 1:
        raise ValueError(
            "Not enough tokens for one training sample: "
            f"got {len(tokens)}, need at least {config.data.max_length + 1}"
        )

    train_loader, val_loader = make_data_loaders(
        tokens=tokens,
        seq_len=config.data.max_length,
        batch_size=config.training.batch_size,
        validation_split=config.data.validation_split,
        seed=config.data.seed,
    )

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Training split is empty after applying validation_split")

    model = SimpleLM.from_config(config.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay,
    )

    losses = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        max_steps=config.training.max_steps,
        log_interval=config.training.log_interval,
    )

    print(
        f"Data       : tokens={len(tokens)}, "
        f"train_samples={len(train_loader.dataset)}, "  # type: ignore[arg-type]
        f"val_samples={len(val_loader.dataset)}"  # type: ignore[arg-type]
    )
    print(f"Training done. initial_loss={losses[0]:.4f} final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
