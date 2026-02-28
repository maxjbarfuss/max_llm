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
from src.training.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    print_once,
    wrap_model_ddp,
)
from src.training.loop import train
from src.training.optimizer import configure_optimizer_param_groups
from src.training.profiling import log_model_size
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.utils import seed_everything, seed_worker


def load_tokens(
    dataset_path: str | Path,
    tokenizer_name: str,
    tokenizer_backend: str,
    tokenizer_mode: str | None = None,
    tokenizer_vocab_size: int | None = None,
    unigram_model_path: str | None = None,
) -> torch.Tensor:
    """Load tokens from a dataset path (either .npy or .txt).

    Args:
        dataset_path: Path to .npy token file or .txt text file
        tokenizer_name: Name of tokenizer to use
        tokenizer_backend: Tokenizer backend (gpt2_bpe, unigram, char, etc.)
        tokenizer_mode: Mode for char tokenizer (utf8, utf16, utf32, codepoint)
        tokenizer_vocab_size: Vocab size for char tokenizer
        unigram_model_path: Path to unigram model file

    Returns:
        Tensor of token IDs (dtype=torch.long)
    """
    dataset_path = Path(dataset_path)

    if dataset_path.suffix == ".npy":
        # Load pre-tokenized data (fast path)
        token_array = np.load(dataset_path)
        return torch.tensor(token_array, dtype=torch.long)
    else:
        # Load and tokenize text file
        with open(dataset_path, encoding="utf-8") as f:
            corpus_text = f.read()

        tokenizer_kwargs: dict[str, Any] = {}
        if tokenizer_name == "bpe" or tokenizer_backend == "gpt2_bpe":
            tokenizer_kwargs = {"encoding": "gpt2"}
        elif tokenizer_name == "unigram" or tokenizer_backend == "unigram":
            if not unigram_model_path:
                raise ValueError("unigram_model_path is required for unigram tokenizer")
            tokenizer_kwargs = {"model_path": unigram_model_path}
        else:
            tokenizer_kwargs = {"mode": tokenizer_mode or "utf8"}
            if tokenizer_mode == "codepoint":
                tokenizer_kwargs["vocab_size"] = tokenizer_vocab_size or 256

        tokenizer = TokenizerFactory.create(tokenizer_name, **tokenizer_kwargs)
        return torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)


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
    train_tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    val_tokens: torch.Tensor | None = None,
    test_tokens: torch.Tensor | None = None,
    validation_split: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train, validation, and optional test data loaders.

    If val_tokens is provided, uses it directly for validation.
    Otherwise, splits train_tokens using validation_split ratio.
    If test_tokens is provided, creates a test loader.

    Args:
        train_tokens: Flat sequence of training token IDs
        seq_len: Sequence length for each training sample
        batch_size: Batch size for loaders
        val_tokens: Optional separate validation tokens (overrides split if provided)
        test_tokens: Optional separate test tokens (creates test loader if provided)
        validation_split: Fraction of train_tokens to use for validation (ignored if val_tokens provided)
        seed: Random seed for reproducibility
        device: Device for pin_memory setting (pinned if CUDA)
        num_workers: Number of data loading workers (default: 0)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)
        persistent_workers: Keep workers alive between epochs (default: False)

    Returns:
        Tuple of (train_loader, val_loader, test_loader_or_none)
    """
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Determine pin_memory setting based on device
    pin_memory = device is not None and device.type == "cuda"

    def _create_loader_from_tokens(
        tokens: torch.Tensor, shuffle: bool = False
    ) -> tuple[DataLoader, int]:
        """Create a DataLoader from tokens and return (loader, num_samples)."""
        # Create non-overlapping sequences
        num_samples = len(tokens) // (seq_len + 1)
        if num_samples == 0:
            raise ValueError(
                f"Not enough tokens ({len(tokens)}) for at least one sample (need {seq_len + 1})"
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
        dataset = TensorDataset(inputs_tensor, targets_tensor)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            worker_init_fn=seed_worker if shuffle else None,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        return loader, num_samples

    # Create train loader
    train_loader, train_num_samples = _create_loader_from_tokens(train_tokens, shuffle=True)

    # Create validation loader
    if val_tokens is not None:
        # Use separate validation dataset
        val_loader, val_num_samples = _create_loader_from_tokens(val_tokens, shuffle=False)
    else:
        # Split train_tokens for validation
        num_samples = len(train_tokens) // (seq_len + 1)
        split_idx = int(num_samples * (1 - validation_split))

        # Re-create train/val from split
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for i in range(num_samples):
            start = i * (seq_len + 1)
            end = start + seq_len + 1
            sample = train_tokens[start:end]
            inputs.append(sample[:-1])
            targets.append(sample[1:])

        inputs_tensor = torch.stack(inputs)
        targets_tensor = torch.stack(targets)

        train_dataset = TensorDataset(inputs_tensor[:split_idx], targets_tensor[:split_idx])
        val_dataset = TensorDataset(inputs_tensor[split_idx:], targets_tensor[split_idx:])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    # Create optional test loader
    test_loader = None
    if test_tokens is not None:
        test_loader, _test_num_samples = _create_loader_from_tokens(test_tokens, shuffle=False)

    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a max-llm model")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (DDP)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling (logs model size and memory usage)",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)

    # Initialize distributed training if requested
    distributed_info = None
    if args.distributed:
        distributed_info = init_distributed(backend="nccl")
        device = distributed_info["device"]
        rank = distributed_info["rank"]
        world_size = distributed_info["world_size"]
        print_once(f"Distributed: Initialized with {world_size} processes")
    else:
        rank = 0
        world_size = 1
        device = resolve_device("auto")

    # Set all random seeds for reproducibility
    seed_everything(config.data.seed, deterministic=True)

    print_once(f"Experiment : {config.name}")
    print_once(f"Output dir : {config.output_dir}")
    print_once(f"Device      : {device}")
    if args.distributed:
        print_once(f"Distributed: Rank {rank}/{world_size}")
    print_once(
        f"Model      : hidden_size={config.model.hidden_size}, "
        f"vocab_size={config.model.vocab_size}, "
        f"layers={config.model.num_layers}, "
        f"heads={config.model.num_heads}"
    )
    print_once(
        f"Training   : max_steps={config.training.max_steps}, "
        f"lr={config.training.learning_rate}, "
        f"batch_size={config.training.batch_size}"
    )

    # Load training dataset
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    print_once(f"Loading training data from {dataset_path}")
    train_tokens = load_tokens(
        dataset_path,
        tokenizer_name=config.data.tokenizer_name,
        tokenizer_backend=config.data.tokenizer_backend,
        tokenizer_mode=config.data.tokenizer_mode,
        tokenizer_vocab_size=config.data.tokenizer_vocab_size,
        unigram_model_path=config.data.unigram_model_path,
    )

    # Load optional validation dataset
    val_tokens = None
    if config.data.validation_dataset_path:
        val_path = Path(config.data.validation_dataset_path)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation dataset not found: {val_path}")
        print_once(f"Loading validation data from {val_path}")
        val_tokens = load_tokens(
            val_path,
            tokenizer_name=config.data.tokenizer_name,
            tokenizer_backend=config.data.tokenizer_backend,
            tokenizer_mode=config.data.tokenizer_mode,
            tokenizer_vocab_size=config.data.tokenizer_vocab_size,
            unigram_model_path=config.data.unigram_model_path,
        )

    # Load optional test dataset
    test_tokens = None
    if config.data.test_dataset_path:
        test_path = Path(config.data.test_dataset_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {test_path}")
        print_once(f"Loading test data from {test_path}")
        test_tokens = load_tokens(
            test_path,
            tokenizer_name=config.data.tokenizer_name,
            tokenizer_backend=config.data.tokenizer_backend,
            tokenizer_mode=config.data.tokenizer_mode,
            tokenizer_vocab_size=config.data.tokenizer_vocab_size,
            unigram_model_path=config.data.unigram_model_path,
        )

    if len(train_tokens) < config.data.max_length + 1:
        raise ValueError(
            "Not enough tokens in training set for one sample: "
            f"got {len(train_tokens)}, need at least {config.data.max_length + 1}"
        )

    # Create data loaders
    train_loader, val_loader, test_loader = create_simple_loaders(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
        seq_len=config.data.max_length,
        batch_size=config.training.batch_size,
        validation_split=config.data.validation_split,
        seed=config.data.seed,
        device=device,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )

    if config.data.num_workers > 0:
        print_once(
            f"DataLoader : {config.data.num_workers} workers, "
            f"prefetch_factor={config.data.prefetch_factor}, "
            f"persistent_workers={config.data.persistent_workers}"
        )

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Training split is empty")
    if len(val_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Validation split is empty")

    # Create model based on model_type
    model: BaseLearningModel
    attention_backend = config.training.attention_backend
    if attention_backend != "standard":
        print_once(f"Attention backend: {attention_backend}")
    if config.model.model_type == "simple_lm":
        model = SimpleLM.from_config(config.model)
    elif config.model.model_type == "attention_lm":
        model = AttentionLM.from_config(config.model, attention_backend=attention_backend)
    elif config.model.model_type == "decoder_lm":
        model = DecoderLM.from_config(config.model, attention_backend=attention_backend)
    else:
        raise ValueError(
            f"Unknown model_type: {config.model.model_type}. "
            "Supported types: simple_lm, attention_lm, decoder_lm"
        )

    # Move model to device
    model = model.to(device)

    # Log model size if profiling enabled
    if args.profile and is_main_process():
        log_model_size(model)

    # Wrap model with DDP if distributed
    if args.distributed:
        assert distributed_info is not None, "distributed_info should be set when distributed=True"
        print_once("Wrapping model with DistributedDataParallel")
        model = wrap_model_ddp(model, device_ids=[distributed_info["local_rank"]])  # type: ignore[assignment]

    # Apply torch.compile if enabled (Phase 3+)
    # Note: torch.compile should be applied AFTER DDP wrapping
    if config.training.use_torch_compile:
        print_once("torch.compile: Enabled (mode=max-autotune)")
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore[assignment]
        except Exception as e:
            print_once(f"Warning: torch.compile failed ({e}), continuing without compilation")

    # Configure optimizer with parameter groups (selective weight decay)
    if config.training.weight_decay > 0:
        print_once(
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
    print_once(
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
            print_once(f"Precision  : Using mixed precision (AMP) with {first_precision}")

    # Gradient clipping
    gradient_clip_norm = config.training.gradient_clip_norm
    if gradient_clip_norm > 0:
        print_once(f"Gradient   : Clipping at norm {gradient_clip_norm}")

    # Gradient accumulation
    if config.training.gradient_accumulation_steps > 1:
        print_once(
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

    print_once(
        f"Data       : train_tokens={len(train_tokens)}, "
        f"train_samples={len(train_loader.dataset)}, "  # type: ignore[arg-type]
        f"val_tokens={len(val_tokens) if val_tokens is not None else 'splits'}, "
        f"val_samples={len(val_loader.dataset)}"  # type: ignore[arg-type]
        f"{f', test_tokens={len(test_tokens)}, test_samples={len(test_loader.dataset)}' if test_loader is not None else ''}"  # type: ignore
    )
    losses = metrics["losses"]
    perplexities = metrics["perplexities"]
    print_once(
        f"Training done. initial_loss={losses[0]:.4f} final_loss={losses[-1]:.4f} "
        f"initial_ppl={perplexities[0]:.2f} final_ppl={perplexities[-1]:.2f}"
    )

    # Save checkpoint only on main process
    if is_main_process():
        # Unwrap model if DDP
        model_to_save = model.module if args.distributed else model
        ckpt_path = save_checkpoint(
            model_to_save, optimizer, step=config.training.max_steps, output_dir=config.output_dir  # type: ignore[arg-type]
        )
        print(f"Checkpoint : {ckpt_path}")

    # Cleanup distributed training
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
