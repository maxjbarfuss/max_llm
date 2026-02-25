"""Shared inference utilities for checkpoint loading, device resolution, and tokenizer creation."""

from __future__ import annotations

from typing import Any

import torch

from src.config.experiment import DataConfig
from src.models.learning_model import SimpleLM
from src.tokenizer import Tokenizer, TokenizerFactory


def resolve_device(device_spec: str) -> torch.device:
    """Resolve device specification to torch.device.

    Args:
        device_spec: "auto", "cuda", or "cpu"

    Returns:
        torch.device object

    Example:
        >>> device = resolve_device("auto")  # Uses CUDA if available
        >>> device = resolve_device("cpu")   # Forces CPU
    """
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def load_checkpoint_into_model(
    model: SimpleLM,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load checkpoint into model, handling various checkpoint formats.

    Supports checkpoints with state_dict stored under various keys:
    - "model_state"
    - "state_dict"
    - "model"
    - Direct state_dict (checkpoint itself is the state_dict)

    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        device: Device for map_location

    Raises:
        ValueError: If checkpoint format is unsupported

    Example:
        >>> model = SimpleLM.from_config(config.model)
        >>> device = torch.device("cuda")
        >>> load_checkpoint_into_model(model, "checkpoint.pt", device)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        # Try common state_dict keys
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                model.load_state_dict(checkpoint[key], strict=False)
                return

        # If none of the specific keys worked, try the checkpoint dict itself
        model.load_state_dict(checkpoint, strict=False)
        return

    raise ValueError("Unsupported checkpoint format")


def create_tokenizer_from_data_config(data_config: DataConfig) -> Tokenizer:
    """Create tokenizer from data config.

    Handles mode-specific tokenizer kwargs (e.g., vocab_size for codepoint mode).

    Args:
        data_config: DataConfig object with tokenizer settings

    Returns:
        Tokenizer instance

    Example:
        >>> from src.config.experiment import ExperimentConfig
        >>> config = ExperimentConfig.from_toml("config/experiment.toml")
        >>> tokenizer = create_tokenizer_from_data_config(config.data)
    """
    tokenizer_kwargs: dict[str, Any] = {"mode": data_config.tokenizer_mode}
    if data_config.tokenizer_mode == "codepoint":
        tokenizer_kwargs["vocab_size"] = data_config.tokenizer_vocab_size
    return TokenizerFactory.create(data_config.tokenizer_name, **tokenizer_kwargs)
