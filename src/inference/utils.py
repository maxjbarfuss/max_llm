"""Shared inference utilities for checkpoint loading, device resolution, and tokenizer creation."""

import torch

from src.config.experiment import DataConfig
from src.models.learning_model import LearningModel
from src.tokenizer import Tokenizer, create_configured_tokenizer


def resolve_device(device_spec: str) -> torch.device:
    """Return torch.device for "auto", "cuda", or "cpu"."""
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def load_checkpoint_into_model(
    model: LearningModel,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load checkpoint into model, trying keys "model_state", "state_dict", "model", then raw dict.

    Raises ValueError if the checkpoint format is unsupported.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                model.load_state_dict(checkpoint[key], strict=False)
                return
        model.load_state_dict(checkpoint, strict=False)
        return

    raise ValueError("Unsupported checkpoint format")


def create_tokenizer_from_data_config(data_config: DataConfig) -> Tokenizer:
    """Create tokenizer from DataConfig tokenizer settings."""
    return create_configured_tokenizer(
        tokenizer_name=data_config.tokenizer_name,
        tokenizer_mode=data_config.tokenizer_mode,
        tokenizer_vocab_size=getattr(data_config, "tokenizer_vocab_size", None),
        tokenizer_backend=getattr(data_config, "tokenizer_backend", None),
        unigram_model_path=getattr(data_config, "unigram_model_path", None),
        tokenizer_vocab_path=getattr(data_config, "tokenizer_vocab_path", None),
    )
