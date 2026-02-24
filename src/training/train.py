"""Training entrypoint for max-llm.

Usage:
    python -m src.training.train --config config/experiment.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.experiment import ExperimentConfig
from src.models.learning_model import SimpleLM
from src.tokenizer import TokenizerFactory
from src.training.loop import train


def create_simple_loaders(
    tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from token sequence.

    Args:
        tokens: Flat sequence of token IDs
        seq_len: Sequence length for each training sample
        batch_size: Batch size for loaders
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    torch.manual_seed(seed)

    # Create non-overlapping sequences
    num_samples = len(tokens) // (seq_len + 1)
    if num_samples == 0:
        raise ValueError(
            f"Not enough tokens ({len(tokens)}) for at least one sample " f"(need {seq_len + 1})"
        )

    # Prepare input/target pairs
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for i in range(num_samples):
        start = i * (seq_len + 1)
        end = start + seq_len + 1
        sample = tokens[start:end]
        inputs.append(sample[:-1])
        targets.append(sample[1:])

    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)

    # Split into train/val
    split_idx = int(len(inputs_tensor) * (1 - validation_split))
    train_dataset = TensorDataset(inputs_tensor[:split_idx], targets_tensor[:split_idx])
    val_dataset = TensorDataset(inputs_tensor[split_idx:], targets_tensor[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


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

    # Load dataset (supports both .npy tokens and .txt files)
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix == ".npy":
        # Load pre-tokenized data (fast path)
        print(f"Loading pre-tokenized data from {dataset_path}")
        token_array = np.load(dataset_path)
        tokens = torch.tensor(token_array, dtype=torch.long)
    else:
        # Load and tokenize text file (legacy path)
        print(f"Loading and tokenizing text from {dataset_path}")
        with open(dataset_path, encoding="utf-8") as f:
            corpus_text = f.read()

        tokenizer_kwargs = {"mode": config.data.tokenizer_mode}
        if config.data.tokenizer_mode == "codepoint":
            tokenizer_kwargs["vocab_size"] = config.data.tokenizer_vocab_size

        tokenizer = TokenizerFactory.create(
            config.data.tokenizer_name,
            **tokenizer_kwargs,
        )
        tokens = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)

    if len(tokens) < config.data.max_length + 1:
        raise ValueError(
            "Not enough tokens for one training sample: "
            f"got {len(tokens)}, need at least {config.data.max_length + 1}"
        )

    # Create data loaders
    train_loader, val_loader = create_simple_loaders(
        tokens=tokens,
        seq_len=config.data.max_length,
        batch_size=config.training.batch_size,
        validation_split=config.data.validation_split,
        seed=config.data.seed,
    )

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Training split is empty after applying validation_split")

    # Create model and optimizer
    model = SimpleLM.from_config(config.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay,
    )

    # Train
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
