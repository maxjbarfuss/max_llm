"""Config validation utilities for checkpoint compatibility."""

import warnings
from pathlib import Path

import torch

from .data import DataConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .training import TrainingConfig


class ConfigVersionMismatchError(Exception):
    """Raised when checkpoint config versions don't match current code."""


def validate_checkpoint_config_compatibility(
    checkpoint_path: str | Path,
    strict: bool = True,
) -> dict[str, int]:
    """Validate that checkpoint's config versions match current code.

    Args:
        checkpoint_path: Path to checkpoint file.
        strict: If True, raises error on version mismatch.

    Returns:
        Dictionary of config versions from checkpoint.

    Raises:
        ConfigVersionMismatchError: If strict=True and versions don't match.
        FileNotFoundError: If checkpoint doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Old checkpoints may not have version info
    if "config_versions" not in checkpoint:
        msg = (
            f"Checkpoint at {checkpoint_path} was created before config versioning. "
            f"Cannot verify compatibility. Consider re-training or manually validating "
            f"that config matches checkpoint creation time."
        )
        if strict:
            raise ConfigVersionMismatchError(msg)
        warnings.warn(msg, stacklevel=2)
        return {}

    saved_versions = checkpoint["config_versions"]
    current_versions = {
        "model": ModelConfig.__version__,
        "training": TrainingConfig.__version__,
        "data": DataConfig.__version__,
        "inference": InferenceConfig.__version__,
    }

    mismatches = [
        f"  {name}: checkpoint v{saved_versions.get(name)} != current v{current}"
        for name, current in current_versions.items()
        if saved_versions.get(name) not in (None, current)
    ]

    if mismatches:
        error_msg = (
            f"Config version mismatch in {checkpoint_path}:\n"
            + "\n".join(mismatches)
            + "\n\nTo resolve:\n"
            + "  1. Use matching config versions (checkout git commit matching checkpoint)\n"
            + "  2. Re-train from scratch with current config\n"
            + "  3. Set strict=False to load anyway (breaks reproducibility guarantee)"
        )
        if strict:
            raise ConfigVersionMismatchError(error_msg)
        warnings.warn(error_msg, stacklevel=2)

    return saved_versions


def get_checkpoint_config_versions(checkpoint_path: str | Path) -> dict[str, int] | None:
    """Get config versions from checkpoint without validation.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Dictionary of config versions, or None if checkpoint has no version info.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("config_versions")
