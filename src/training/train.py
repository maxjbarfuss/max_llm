"""Training entrypoint for max-llm.

Usage:
    python -m src.training.train --config config/experiment.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.experiment import ExperimentConfig
from src.inference.utils import resolve_device
from src.models.learning_model import AttentionLM, BaseLearningModel, DecoderLM, SimpleLM
from src.tokenizer import TokenizerFactory
from src.training.loop import train
from src.training.optimizer import configure_optimizer_param_groups
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.utils import seed_everything, seed_worker


def save_checkpoint(
    model: BaseLearningModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str | Path,
) -> Path:
    """Save model and optimizer state to output_dir/checkpoint.pt.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        step: Current training step (stored for resumability).
        output_dir: Directory to write checkpoint into (created if absent).

    Returns:
        Path to the written checkpoint file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        },
        path,
    )
    return path


def create_simple_loaders(
    tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    validation_split: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from token sequence.

    Args:
        tokens: Flat sequence of token IDs
        seq_len: Sequence length for each training sample
        batch_size: Batch size for loaders
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        device: Device for pin_memory setting (pinned if CUDA)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

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

    # Determine pin_memory setting based on device
    pin_memory = device is not None and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a max-llm model")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    return parser.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)

    # Set all random seeds for reproducibility
    seed_everything(config.data.seed, deterministic=True)

    # Resolve device (auto-detect CUDA if available)
    device = resolve_device("auto")

    print(f"Experiment : {config.name}")
    print(f"Output dir : {config.output_dir}")
    print(f"Device      : {device}")
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

        tokenizer_kwargs: dict[str, Any] = {"mode": config.data.tokenizer_mode}
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
        device=device,
    )

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Training split is empty after applying validation_split")

    # Create model based on model_type
    model: BaseLearningModel
    if config.model.model_type == "simple_lm":
        model = SimpleLM.from_config(config.model)
    elif config.model.model_type == "attention_lm":
        model = AttentionLM.from_config(config.model)
    elif config.model.model_type == "decoder_lm":
        model = DecoderLM.from_config(config.model)
    else:
        raise ValueError(
            f"Unknown model_type: {config.model.model_type}. "
            "Supported types: simple_lm, attention_lm, decoder_lm"
        )

    # Move model to device
    model = model.to(device)

    # Configure optimizer with parameter groups (selective weight decay)
    if config.training.weight_decay > 0:
        print(
            "Optimizer  : Using selective weight decay " "(excluding bias and LayerNorm parameters)"
        )
        param_groups = configure_optimizer_param_groups(
            model=model,
            weight_decay=config.training.weight_decay,
            learning_rate=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.epsilon,
        )
        optimizer = torch.optim.Adam(param_groups)
    else:
        # No weight decay: use simple parameter list
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.epsilon,
            weight_decay=0.0,
        )

    # Create learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
        min_lr_ratio=0.1,
    )
    print(
        f"Scheduler  : Warmup {config.training.warmup_steps} steps, "
        f"cosine decay to 10% over {config.training.max_steps} steps"
    )

    # Determine mixed precision setting from precision_schedule
    # For now, use the first schedule entry; Phase 4 will implement full schedule support
    use_amp = False
    if config.training.precision_schedule:
        first_precision = config.training.precision_schedule[0][2]
        if first_precision in ("bf16", "mixed"):
            use_amp = True
            print(f"Precision  : Using mixed precision (AMP) with {first_precision}")

    # Gradient clipping
    gradient_clip_norm = config.training.gradient_clip_norm
    if gradient_clip_norm > 0:
        print(f"Gradient   : Clipping at norm {gradient_clip_norm}")

    # Gradient accumulation
    if config.training.gradient_accumulation_steps > 1:
        print(
            f"Accumulation: {config.training.gradient_accumulation_steps} micro-batches "
            f"(effective batch size: {config.training.effective_batch_size})"
        )

    # Train
    metrics = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        max_steps=config.training.max_steps,
        log_interval=config.training.log_interval,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_clip_norm=gradient_clip_norm if gradient_clip_norm > 0 else None,
        use_amp=use_amp,
        lr_scheduler=lr_scheduler,
        log_tokens_per_sec=True,
        log_gpu_memory=True,
    )

    print(
        f"Data       : tokens={len(tokens)}, "
        f"train_samples={len(train_loader.dataset)}, "  # type: ignore[arg-type]
        f"val_samples={len(val_loader.dataset)}"  # type: ignore[arg-type]
    )
    losses = metrics["losses"]
    perplexities = metrics["perplexities"]
    print(
        f"Training done. initial_loss={losses[0]:.4f} final_loss={losses[-1]:.4f} "
        f"initial_ppl={perplexities[0]:.2f} final_ppl={perplexities[-1]:.2f}"
    )

    ckpt_path = save_checkpoint(
        model, optimizer, step=config.training.max_steps, output_dir=config.output_dir
    )
    print(f"Checkpoint : {ckpt_path}")


if __name__ == "__main__":
    main()
