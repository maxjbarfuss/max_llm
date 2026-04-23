"""Training entrypoint for max-llm.

Usage:
    python -m src.training.train --config config/milestones/<experiment>.toml
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.config.data import DataConfig
from src.config.experiment import ExperimentConfig
from src.eval import BenchmarkRunner, BenchmarkRunnerConfig
from src.inference.utils import resolve_device
from src.models.learning_model import LearningModel
from src.tokenizer import create_configured_tokenizer
from src.training.distributed import (
    cleanup_distributed,
    collect_fsdp_state_dicts,
    create_distributed_sampler,
    init_distributed,
    is_main_process,
    print_once,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from src.training.loop import train
from src.training.optimizer import configure_optimizer_param_groups
from src.training.profiling import log_model_size
from src.training.scheduler import (
    get_cosine_schedule_with_warmup,
    get_resume_hold_ramp_then_wsd_schedule,
    get_sgdr_schedule,
    get_wsd_schedule,
)
from src.utils import seed_everything, seed_worker


def _write_training_status(
    output_dir: str | Path,
    step: int,
    max_steps: int,
    checkpoint_path: str | Path | None = None,
    val_loss: float | None = None,
    done: bool = False,
) -> None:
    """Write training_status.json to output_dir for external status monitoring."""
    status = {
        "step": step,
        "max_steps": max_steps,
        "done": done,
        "val_loss": val_loss,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = Path(output_dir) / "training_status.json"
    path.write_text(json.dumps(status, indent=2))


class TokenDataset(Dataset):
    """PyTorch Dataset backed by a flat token array (numpy or torch).

    Yields non-overlapping (input, target) sequence pairs without materialising
    the full set of pairs in memory upfront.  When ``tokens`` is a memory-mapped
    numpy array (opened with ``np.load(..., mmap_mode='r')``), only the pages
    that are actually accessed are loaded from disk, keeping RAM usage minimal
    even for large files.

    Call ``set_epoch(epoch)`` before each epoch to apply a random start offset,
    so successive passes through the dataset see different sequence boundaries.
    The offset is bounded by the tail tokens (``len(tokens) % (seq_len + 1)``)
    so ``num_samples`` and ``__len__`` are stable across epochs.

    Args:
        tokens: Flat 1-D array of token IDs (np.ndarray or torch.Tensor).
        seq_len: Length of each input sequence.  Each sample covers
            ``seq_len + 1`` consecutive tokens: ``tokens[i:i+seq_len]`` as
            input and ``tokens[i+1:i+seq_len+1]`` as target.
        seed: Base seed for deterministic per-epoch offset generation.
    """

    def __init__(self, tokens: np.ndarray | torch.Tensor, seq_len: int, seed: int = 0) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.seed = seed
        self._offset = 0
        self.num_samples = len(tokens) // (seq_len + 1)
        if self.num_samples == 0:
            raise ValueError(
                f"Not enough tokens ({len(tokens)}) for one sample (need {seq_len + 1})"
            )
        # Tail tokens not consumed by the fixed windows; offset is capped here
        # so the last sample never goes out of bounds without changing num_samples.
        self._max_offset = len(tokens) - self.num_samples * (seq_len + 1)

    def set_epoch(self, epoch: int) -> None:
        """Randomize the sequence start offset for this epoch.

        Uses ``seed + epoch`` as the RNG seed so all DDP ranks independently
        compute the same offset without communication.  No-op when the token
        array has no tail (i.e. length is an exact multiple of seq_len + 1).
        """
        if self._max_offset > 0:
            self._offset = random.Random(self.seed + epoch).randint(0, self._max_offset)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._offset + idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        sample = self.tokens[start:end]
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample.astype(np.int64))
        x = sample[:-1].long()
        y = sample[1:].long()
        return x, y


def load_tokens(
    dataset_path: str | Path,
    tokenizer_name: str,
    tokenizer_backend: str,
    tokenizer_mode: str | None = None,
    tokenizer_vocab_size: int | None = None,
    unigram_model_path: str | None = None,
    tokenizer_vocab_path: str | None = None,
    use_mmap: bool = True,
) -> np.ndarray | torch.Tensor:
    """Load tokens from a dataset path (either .npy or .txt).

    For ``.npy`` files, returns a memory-mapped numpy array by default
    (``use_mmap=True``).  Callers that need a plain torch.Tensor can pass
    ``use_mmap=False``, which loads the whole file into RAM.

    Args:
        dataset_path: Path to .npy token file or .txt text file
        tokenizer_name: Name of tokenizer to use
        tokenizer_backend: Tokenizer backend (gpt2_bpe, unigram, char, etc.)
        tokenizer_mode: Mode for char tokenizer (utf8, utf16, utf32, codepoint)
        tokenizer_vocab_size: Vocab size for char tokenizer
        unigram_model_path: Path to unigram model file
        tokenizer_vocab_path: Optional path to custom BPE vocab json
        use_mmap: If True (default), memory-map .npy files so only accessed
            pages are loaded into RAM.  Ignored for text files.

    Returns:
        Memory-mapped np.ndarray for .npy files (when use_mmap=True),
        torch.Tensor for .npy files (when use_mmap=False),
        or torch.Tensor for text files (after tokenisation).
    """
    dataset_path = Path(dataset_path)

    if dataset_path.suffix == ".npy":
        if use_mmap:
            # Memory-mapped: OS loads only the pages we access; no full RAM copy
            return np.load(dataset_path, mmap_mode="r")
        # Fallback: load entire file into RAM
        token_array = np.load(dataset_path)
        return torch.tensor(token_array, dtype=torch.long)

    # Load and tokenize text file
    with open(dataset_path, encoding="utf-8") as f:
        corpus_text = f.read()
    tokenizer = create_configured_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_mode=tokenizer_mode,
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_backend=tokenizer_backend,
        unigram_model_path=unigram_model_path,
        tokenizer_vocab_path=tokenizer_vocab_path,
    )
    return torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)


def _load_dataset(
    path: str | Path,
    cfg: DataConfig,
) -> np.ndarray | torch.Tensor:
    """Load tokens from a dataset path using tokenizer settings from DataConfig."""
    return load_tokens(
        path,
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_backend=cfg.tokenizer_backend,
        tokenizer_mode=cfg.tokenizer_mode,
        tokenizer_vocab_size=cfg.tokenizer_vocab_size,
        unigram_model_path=cfg.unigram_model_path,
        tokenizer_vocab_path=cfg.tokenizer_vocab_path,
    )


def save_checkpoint(
    model: LearningModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str | Path,
    filename: str = "checkpoint.pt",
) -> Path:
    """Save model and optimizer state to output_dir/filename.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        step: Current training step (stored for resumability).
        output_dir: Directory to write checkpoint into (created if absent).
        filename: Filename for the checkpoint (default: checkpoint.pt).

    Returns:
        Path to the written checkpoint file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        },
        path,
    )
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: LearningModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load model and optimizer state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
    Returns:
        Training step from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint.get("step", 0)


def create_simple_loaders(  # noqa: C901
    train_tokens: np.ndarray | torch.Tensor,
    seq_len: int,
    batch_size: int,
    val_tokens: np.ndarray | torch.Tensor | None = None,
    test_tokens: np.ndarray | torch.Tensor | None = None,
    validation_split: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train, validation, and optional test data loaders.

    Accepts both numpy arrays (including memory-mapped ones from
    ``np.load(..., mmap_mode='r')``) and torch.Tensors.  When a numpy array is
    provided, a :class:`TokenDataset` is used so that pairs are generated lazily
    in ``__getitem__`` rather than materialised all at once.  For torch.Tensor
    inputs the legacy ``TensorDataset`` path is preserved for backward
    compatibility.

    If val_tokens is provided, uses it directly for validation.
    Otherwise, splits train_tokens using validation_split ratio.
    If test_tokens is provided, creates a test loader.

    Args:
        train_tokens: Flat sequence of training token IDs (numpy or torch).
        seq_len: Sequence length for each training sample.
        batch_size: Batch size for loaders.
        val_tokens: Optional separate validation tokens (overrides split if provided).
        test_tokens: Optional separate test tokens (creates test loader if provided).
        validation_split: Fraction of train_tokens to use for validation (ignored if val_tokens provided).
        seed: Random seed for reproducibility.
        device: Device for pin_memory setting (pinned if CUDA).
        num_workers: Number of data loading workers (default: 0).
            Recommended: 2–8 for large datasets. Avoid 0 for large datasets to prevent CPU lockup.
        prefetch_factor: Number of batches to prefetch per worker (default: 2).
        persistent_workers: Keep workers alive between epochs (default: False).
            Recommended: True if num_workers > 0 and training is multi-epoch.

    Returns:
        Tuple of (train_loader, val_loader, test_loader_or_none)

    Notes:
        - Setting num_workers=0 causes all data loading to occur in the main process, which can lock up the CPU for large datasets.
        - For large datasets (e.g., >100,000 samples), set num_workers >= 2.
        - persistent_workers=True is only effective if num_workers > 0.
        - Monitor for warnings about slow data loading or CPU utilization.
    """
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Determine pin_memory setting based on device
    use_pin_memory = (
        pin_memory if pin_memory is not None else device is not None and device.type == "cuda"
    )

    total_samples = len(train_tokens) // (seq_len + 1)
    if total_samples == 0:
        raise ValueError(
            f"Not enough tokens ({len(train_tokens)}) for at least one sample (need {seq_len + 1})"
        )

    # Warn if num_workers=0 and dataset is large (risk of CPU lockup)
    if num_workers == 0 and total_samples > 100_000:
        import warnings

        warnings.warn(
            f"[Max LLM] num_workers=0 with {total_samples:,} samples: this can cause CPU lockup or unresponsiveness. "
            "Set num_workers=2 or higher for large datasets.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _make_dataset(
        tokens: np.ndarray | torch.Tensor, start: int = 0, end: int | None = None
    ) -> Dataset:
        """Return a Dataset for sample indices [start, end)."""
        if isinstance(tokens, np.ndarray):
            tok_start = start * (seq_len + 1)
            tok_end = end * (seq_len + 1) if end is not None else None
            sliced = tokens[tok_start:tok_end]
            return TokenDataset(sliced, seq_len, seed=seed)
        # torch.Tensor legacy path: pre-build all pairs (backward-compatible with tests)
        if end is None:
            end = len(tokens) // (seq_len + 1)
        num = end - start
        if num == 0:
            # Empty split (e.g. validation_split=0.0): return empty TensorDataset
            empty = torch.zeros((0, seq_len), dtype=torch.long)
            return TensorDataset(empty, empty)
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for i in range(start, end):
            s = i * (seq_len + 1)
            sample = tokens[s : s + seq_len + 1]
            inputs.append(sample[:-1])
            targets.append(sample[1:])
        return TensorDataset(torch.stack(inputs), torch.stack(targets))

    def _create_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        # Use DistributedSampler in distributed mode for proper data sharding
        sampler = create_distributed_sampler(dataset, shuffle=shuffle, seed=seed)
        # When using sampler, DataLoader's shuffle must be False
        use_shuffle = shuffle if sampler is None else False
        use_generator = generator if (use_shuffle and sampler is None) else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=use_shuffle,
            generator=use_generator,
            worker_init_fn=seed_worker if shuffle else None,
            pin_memory=use_pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    # Build loaders
    if val_tokens is not None:
        train_loader = _create_loader(_make_dataset(train_tokens), shuffle=True)
        val_loader = _create_loader(_make_dataset(val_tokens), shuffle=False)
    else:
        # Split train_tokens into train/val by sample index
        split_idx = int(total_samples * (1 - validation_split))
        train_loader = _create_loader(_make_dataset(train_tokens, 0, split_idx), shuffle=True)
        val_loader = _create_loader(
            _make_dataset(train_tokens, split_idx, total_samples), shuffle=False
        )

    # Create optional test loader
    test_loader = None
    if test_tokens is not None:
        test_loader = _create_loader(_make_dataset(test_tokens), shuffle=False)

    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a max-llm model")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (DDP by default; set distributed_backend=fsdp in config for FSDP)",
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

    want_distributed = args.distributed or config.training.use_distributed

    # If config requests distributed but we're not in a torchrun context, relaunch via torchrun.
    if want_distributed and "RANK" not in os.environ:
        import subprocess
        import sys

        nproc = torch.cuda.device_count()
        if nproc < 1:
            raise RuntimeError("use_distributed=true but no CUDA devices found")
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "-m",
            "src.training.train",
        ] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))

    # Initialize distributed training if requested
    distributed_info = None
    if want_distributed:
        distributed_info = init_distributed(backend="nccl")
        device = distributed_info["device"]
        print_once(f"Distributed: Initialized with {distributed_info['world_size']} processes")
    else:
        device = resolve_device("auto")

    # Set all random seeds for reproducibility
    seed_everything(config.data.seed, deterministic=True)

    print_once(f"Experiment : {config.name}")
    print_once(f"Output dir : {config.output_dir}")
    print_once(f"Device      : {device}")
    if want_distributed:
        assert distributed_info is not None
        print_once(f"Distributed: Rank {distributed_info['rank']}/{distributed_info['world_size']}")
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
    train_tokens = _load_dataset(dataset_path, config.data)

    # Load optional validation dataset
    val_tokens = None
    if config.data.validation_dataset_path:
        val_path = Path(config.data.validation_dataset_path)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation dataset not found: {val_path}")
        print_once(f"Loading validation data from {val_path}")
        val_tokens = _load_dataset(val_path, config.data)

    # Load optional test dataset
    test_tokens = None
    if config.data.test_dataset_path:
        test_path = Path(config.data.test_dataset_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {test_path}")
        print_once(f"Loading test data from {test_path}")
        test_tokens = _load_dataset(test_path, config.data)

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
        pin_memory=config.data.pin_memory,
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

    # Create model from config (always LearningModel; num_layers determines phase)
    attention_backend = config.training.attention_backend
    if attention_backend != "standard":
        print_once(f"Attention backend: {attention_backend}")
    model = LearningModel.from_config(config.model, attention_backend=attention_backend)

    # Move model to device
    model = model.to(device)

    # Track loaded step/checkpoint for scheduler and optimizer resume.
    loaded_step = 0
    resume_path: Path | None = None

    # Optional: initialize model from an existing checkpoint
    if config.training.resume_from_checkpoint:
        resume_path = Path(config.training.resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print_once(f"Checkpoint : Loading model weights from {resume_path}")
        loaded_step = load_checkpoint(resume_path, model, optimizer=None)
        print_once(f"Checkpoint : Loaded (saved at step {loaded_step})")

    # Log model size if profiling enabled
    if args.profile and is_main_process():
        log_model_size(model)

    want_fsdp = want_distributed and config.training.distributed_backend == "fsdp"

    if want_distributed:
        assert distributed_info is not None, "distributed_info should be set when distributed=True"
        if want_fsdp:
            from src.models.transformer.transformer_block import TransformerBlock

            # block_attn residuals call apply_attn_only / apply_ffn_only which
            # bypass each TransformerBlock's __call__ and therefore bypass the
            # FSDP pre-forward hook.  Use outer-model-only wrapping so the
            # single FSDP pre-forward hook gathers all params before dispatch.
            uses_block_attn = getattr(config.model, "res_type", "") == "block_attn"
            layer_cls = None if uses_block_attn else TransformerBlock
            fsdp_strategy = "shard_grad_op"
            print_once(
                f"Wrapping model with FSDP (sharding_strategy={fsdp_strategy}, "
                f"inner_wrap={'per-block' if layer_cls else 'model-level'})"
            )
            model = wrap_model_fsdp(
                model,
                transformer_layer_cls=layer_cls,
                sharding_strategy=fsdp_strategy,
            )  # type: ignore[assignment]
        else:
            print_once("Wrapping model with DistributedDataParallel")
            model = wrap_model_ddp(
                model,
                device_ids=[distributed_info["local_rank"]],
                find_unused_parameters=(config.training.attention_backend == "sage"),
            )  # type: ignore[assignment]

    # Apply gradient checkpointing if enabled (reduces activation memory ~4×)
    if config.training.selective_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()  # type: ignore[operator]
            print_once("Gradient checkpointing: Enabled")
        else:
            print_once(
                "Warning: selective_checkpointing=true but model has no checkpointing API. "
                "Implement gradient_checkpointing_enable() on the model to activate this."
            )

    # Apply torch.compile if enabled (Phase 3+)
    # Note: torch.compile should be applied AFTER DDP/FSDP wrapping.
    # IMPORTANT: max-autotune uses CUDA graphs which deadlock DDP on
    # heterogeneous GPUs (RTX 4090 + RTX 3090 Ti).  Use mode="default" for DDP/FSDP.
    use_torch_compile = config.training.use_torch_compile
    if use_torch_compile:
        if want_fsdp:
            print_once("Warning: torch.compile + FSDP has known issues; disabling compile")
            use_torch_compile = False
        elif config.training.attention_backend == "sage":
            print_once(
                "Warning: torch.compile with Sage Attention often provides limited benefit; "
                "disabling compile for stability."
            )
            use_torch_compile = False
        else:
            compile_mode = config.training.torch_compile_mode or (
                "default" if want_distributed else "max-autotune"
            )
            if want_distributed and compile_mode.startswith("max-autotune"):
                print_once(
                    "Warning: max-autotune with distributed training can deadlock on "
                    "heterogeneous GPUs; overriding compile mode to 'default'."
                )
                compile_mode = "default"

            print_once(
                "torch.compile: Enabled "
                f"(mode={compile_mode}, "
                f"fullgraph={config.training.torch_compile_fullgraph}, "
                f"dynamic={config.training.torch_compile_dynamic})"
            )
            try:
                model = torch.compile(  # type: ignore[assignment]
                    model,
                    mode=compile_mode,
                    fullgraph=config.training.torch_compile_fullgraph,
                    dynamic=config.training.torch_compile_dynamic,
                )
            except Exception as e:
                print_once(f"Warning: torch.compile failed ({e}), continuing without compilation")
                use_torch_compile = False

    # fused AdamW: single CUDA kernel for all parameter updates (PyTorch 2.0+, CUDA only).
    # Disabled for FSDP: flat-param sharding is incompatible with the fused kernel.
    use_fused_optimizer = device.type == "cuda" and not want_fsdp
    if use_fused_optimizer:
        print_once("Optimizer  : Using fused AdamW")
    elif want_fsdp:
        print_once("Optimizer  : Using standard AdamW (fused disabled for FSDP)")

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
        optimizer = torch.optim.AdamW(param_groups, fused=use_fused_optimizer)
    else:
        # No weight decay: use simple parameter list
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.epsilon,
            weight_decay=0.0,
            fused=use_fused_optimizer,
        )

    ckpt_lr: float | None = None

    # Resume optimizer state after optimizer construction.
    if resume_path is not None and loaded_step > 0 and config.training.resume_optimizer_state:
        _resume_ckpt = torch.load(resume_path, map_location="cpu")
        optimizer_state = _resume_ckpt.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            param_groups = optimizer_state.get("param_groups")
            if param_groups and isinstance(param_groups, list):
                first_lr = param_groups[0].get("lr")
                if isinstance(first_lr, float):
                    ckpt_lr = first_lr
        # LambdaLR with last_epoch >= 0 requires initial_lr in every param group.
        # Always use the config target LR as initial_lr so the scheduler's base_lr
        # reflects the intended target, not the checkpoint's (possibly different) LR.
        for group in optimizer.param_groups:
            group["initial_lr"] = config.training.learning_rate

    # For resumed runs with schedule restart, read checkpoint LR from optimizer_state
    # metadata if it was not captured while restoring optimizer state.
    if (
        ckpt_lr is None
        and resume_path is not None
        and loaded_step > 0
        and not config.training.resume_scheduler_state
        and config.training.resume_optimizer_state
    ):
        _resume_ckpt = torch.load(resume_path, map_location="cpu")
        optimizer_state = _resume_ckpt.get("optimizer_state")
        if optimizer_state is not None:
            param_groups = optimizer_state.get("param_groups")
            if param_groups and isinstance(param_groups, list):
                first_lr = param_groups[0].get("lr")
                if isinstance(first_lr, float):
                    ckpt_lr = first_lr

    scheduler_resume_step = loaded_step if config.training.resume_scheduler_state else 0

    # Create learning rate scheduler
    # If resuming from checkpoint, initialize scheduler at the loaded step
    # to avoid "scheduler.step() before optimizer.step()" warning
    if config.training.scheduler_type == "wsd":
        lr_mismatch = (
            ckpt_lr is not None
            and config.training.resume_optimizer_state
            and abs(ckpt_lr - config.training.learning_rate) > 1e-10
        )
        use_resume_transition = loaded_step > 0 and (
            not config.training.resume_scheduler_state or lr_mismatch
        )

        if use_resume_transition:
            # If optimizer state is intentionally reset, allow a true warmup from 0 LR.
            if not config.training.resume_optimizer_state:
                start_lr_ratio = 0.0
                ckpt_lr = 0.0
            else:
                if ckpt_lr is None:
                    raise ValueError(
                        "resume_lr_hold_steps/warmup_steps configured, but checkpoint LR "
                        "could not be read from optimizer_state"
                    )
                start_lr_ratio = max(0.0, ckpt_lr / config.training.learning_rate)
            lr_scheduler = get_resume_hold_ramp_then_wsd_schedule(
                optimizer=optimizer,
                num_training_steps=config.training.max_steps,
                start_lr_ratio=start_lr_ratio,
                hold_steps=config.training.resume_lr_hold_steps,
                ramp_steps=config.training.warmup_steps,
                stable_fraction=config.training.wsd_stable_fraction,
                decay_fraction=config.training.wsd_decay_fraction,
                decay_shape=config.training.wsd_decay_shape,
                min_lr_ratio=config.training.min_lr_ratio,
                lowered_linear_alpha=config.training.wsd_lowered_linear_alpha,
                last_epoch=-1,
            )
            # Apply the intended starting LR before the first optimizer step.
            for group in optimizer.param_groups:
                group["lr"] = config.training.learning_rate * start_lr_ratio
            print_once(
                f"Scheduler  : Resume transition hold={config.training.resume_lr_hold_steps}, "
                f"ramp={config.training.warmup_steps}, "
                f"start_lr={ckpt_lr:.2e}, target_lr={config.training.learning_rate:.2e}, "
                f"stable={config.training.wsd_stable_fraction:.2f}, "
                f"decay={config.training.wsd_decay_fraction:.2f}, "
                f"shape={config.training.wsd_decay_shape}, "
                f"min_lr_ratio={config.training.min_lr_ratio:.3f}"
            )
        else:
            lr_scheduler = get_wsd_schedule(
                optimizer=optimizer,
                num_warmup_steps=config.training.warmup_steps,
                num_training_steps=config.training.max_steps,
                stable_fraction=config.training.wsd_stable_fraction,
                decay_fraction=config.training.wsd_decay_fraction,
                decay_shape=config.training.wsd_decay_shape,
                min_lr_ratio=config.training.min_lr_ratio,
                lowered_linear_alpha=config.training.wsd_lowered_linear_alpha,
                last_epoch=scheduler_resume_step - 1 if scheduler_resume_step > 0 else -1,
            )
            print_once(
                f"Scheduler  : WSD warmup={config.training.warmup_steps}, "
                f"stable={config.training.wsd_stable_fraction:.2f}, "
                f"decay={config.training.wsd_decay_fraction:.2f}, "
                f"shape={config.training.wsd_decay_shape}, "
                f"min_lr_ratio={config.training.min_lr_ratio:.3f}"
            )
    elif config.training.scheduler_type == "sgdr":
        lr_scheduler = get_sgdr_schedule(
            optimizer=optimizer,
            num_training_steps=config.training.max_steps,
            num_cycles=config.training.sgdr_num_cycles,
            min_lr_ratio=config.training.min_lr_ratio,
            cycle_decay=config.training.sgdr_cycle_decay,
            last_epoch=scheduler_resume_step - 1 if scheduler_resume_step > 0 else -1,
        )
        print_once(
            f"Scheduler  : SGDR cycles={config.training.sgdr_num_cycles} "
            f"cycle_decay={config.training.sgdr_cycle_decay} "
            f"min_lr_ratio={config.training.min_lr_ratio} "
            f"over {config.training.max_steps} steps"
        )
    else:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=config.training.max_steps,
            min_lr_ratio=config.training.min_lr_ratio,
            last_epoch=scheduler_resume_step - 1 if scheduler_resume_step > 0 else -1,
        )
        print_once(
            f"Scheduler  : Warmup {config.training.warmup_steps} steps, "
            f"cosine decay to {config.training.min_lr_ratio * 100:.1f}% "
            f"over {config.training.max_steps} steps"
        )
    if loaded_step > 0:
        if config.training.resume_scheduler_state:
            print_once(f"Scheduler  : Resuming from step {loaded_step}")
        else:
            print_once(
                "Scheduler  : Restarting schedule from step 0 (resume_scheduler_state=false)"
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

    # TensorBoard writer (main process only)
    tb_writer = None
    if is_main_process():
        from torch.utils.tensorboard import SummaryWriter

        tb_log_dir = Path(config.output_dir) / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        print_once(f"TensorBoard: {tb_log_dir}")
        print_once(f"             tensorboard --logdir {tb_log_dir}")

    benchmark_runner: BenchmarkRunner | None = None
    benchmark_jsonl_path = Path(config.output_dir) / "benchmark_curve.jsonl"
    benchmark_tokenizer = None
    if config.training.benchmark_tasks:
        benchmark_runner = BenchmarkRunner(
            BenchmarkRunnerConfig(
                tasks=config.training.benchmark_tasks,
                split=config.training.benchmark_split,
                max_examples=config.training.benchmark_max_examples,
                length_normalize=config.training.benchmark_length_normalize,
                cache_dir=str(Path(config.output_dir) / "benchmark_cache"),
            )
        )
        benchmark_tokenizer = create_configured_tokenizer(
            tokenizer_name=config.data.tokenizer_name,
            tokenizer_mode=config.data.tokenizer_mode,
            tokenizer_vocab_size=config.data.tokenizer_vocab_size,
            tokenizer_backend=config.data.tokenizer_backend,
            unigram_model_path=config.data.unigram_model_path,
            tokenizer_vocab_path=config.data.tokenizer_vocab_path,
        )
        print_once(
            "Benchmark  : Enabled "
            f"tasks={config.training.benchmark_tasks} "
            f"split={config.training.benchmark_split} "
            f"max_examples={config.training.benchmark_max_examples} "
            f"interval={config.training.benchmark_eval_interval}"
        )

    # Build periodic checkpoint callback (rotating, keeps last N)
    _saved_checkpoints: list[Path] = []

    def _periodic_checkpoint(step: int, val_loss: float | None = None) -> None:
        # FSDP: state_dict() is a collective — all ranks must call it together
        # before branching on is_main_process().
        if want_fsdp:
            model_sd, opt_sd = collect_fsdp_state_dicts(model, optimizer)
            if not is_main_process():
                return
            out = Path(config.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ckpt_path = out / f"checkpoint_step_{step}.pt"
            torch.save(
                {
                    "model_state": model_sd,
                    "optimizer_state": opt_sd,
                    "step": step,
                },
                ckpt_path,
            )
        else:
            if not is_main_process():
                return
            raw_model = getattr(model, "module", model) if want_distributed else model
            ckpt_path = save_checkpoint(
                cast(LearningModel, raw_model),
                optimizer,
                step=step,
                output_dir=config.output_dir,
                filename=f"checkpoint_step_{step}.pt",
            )
        _saved_checkpoints.append(ckpt_path)
        # Rotate: remove oldest if we exceed keep_last_n
        keep_n = config.training.keep_last_n_checkpoints
        while len(_saved_checkpoints) > keep_n:
            old = _saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()
        _write_training_status(
            config.output_dir,
            step=step,
            max_steps=config.training.max_steps,
            checkpoint_path=ckpt_path,
            val_loss=val_loss,
        )
        print_once(f"Checkpoint : {ckpt_path}")

    def _periodic_benchmark(step: int) -> None:
        if benchmark_runner is None or benchmark_tokenizer is None:
            return
        if not is_main_process():
            return
        raw_model = getattr(model, "module", model) if want_distributed else model
        result = benchmark_runner.run(
            model=cast(LearningModel, raw_model),
            tokenizer=benchmark_tokenizer,
            max_seq_len=config.model.max_seq_length,
            device=device,
        )
        payload = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **result,
        }
        benchmark_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with benchmark_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        per_task = result.get("per_task", {})
        macro = result.get("macro_accuracy")
        print_once(
            f"Benchmark  : step={step} macro={macro:.4f} "
            + " ".join(f"{task}={score:.4f}" for task, score in sorted(per_task.items()))
        )
        if tb_writer is not None:
            if isinstance(macro, float):
                tb_writer.add_scalar("eval_benchmark/macro_accuracy", macro, step)
            for task, score in per_task.items():
                tb_writer.add_scalar(f"eval_benchmark/{task}", score, step)

    # Train
    csv_log_path = Path(config.output_dir) / "loss_curve.csv"
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
        csv_log_path=csv_log_path if is_main_process() else None,
        val_loader=val_loader,
        test_loader=test_loader,
        eval_interval=config.training.eval_interval,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_delta=config.training.early_stopping_min_delta,
        label_smoothing=config.training.label_smoothing,
        tb_writer=tb_writer,
        use_torch_compile=use_torch_compile,
        checkpoint_interval=config.training.checkpoint_interval,
        checkpoint_fn=_periodic_checkpoint,
        eval_max_batches=config.training.eval_max_batches,
        benchmark_interval=config.training.benchmark_eval_interval,
        benchmark_fn=_periodic_benchmark,
    )
    if tb_writer is not None:
        tb_writer.close()
    if is_main_process():
        print_once(f"Loss curve : {csv_log_path}")

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

    # Save final checkpoint (FSDP requires all ranks to participate in state_dict gather)
    _val_losses = metrics.get("val_losses", [])
    final_val_loss: float | None = _val_losses[-1] if _val_losses else None
    if want_fsdp:
        model_sd, opt_sd = collect_fsdp_state_dicts(model, optimizer)
        if is_main_process():
            out = Path(config.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ckpt_path = out / "checkpoint.pt"
            torch.save(
                {
                    "model_state": model_sd,
                    "optimizer_state": opt_sd,
                    "step": config.training.max_steps,
                },
                ckpt_path,
            )
            _write_training_status(
                config.output_dir,
                step=config.training.max_steps,
                max_steps=config.training.max_steps,
                checkpoint_path=ckpt_path,
                val_loss=final_val_loss,
                done=True,
            )
            print(f"Checkpoint : {ckpt_path}")
    elif is_main_process():
        # Unwrap model if DDP
        raw_model = getattr(model, "module", model) if want_distributed else model
        model_to_save = cast(LearningModel, raw_model)
        ckpt_path = save_checkpoint(
            model_to_save,
            optimizer,
            step=config.training.max_steps,
            output_dir=config.output_dir,
        )
        _write_training_status(
            config.output_dir,
            step=config.training.max_steps,
            max_steps=config.training.max_steps,
            checkpoint_path=ckpt_path,
            val_loss=final_val_loss,
            done=True,
        )
        print(f"Checkpoint : {ckpt_path}")

    # Cleanup distributed training
    if want_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
