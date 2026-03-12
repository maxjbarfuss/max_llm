"""InferenceConfig definition."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from .toml_utils import load_toml, section_or_root


@dataclass
class InferenceConfig:
    """Inference-time configuration."""

    __version__: ClassVar[int] = 1

    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_new_tokens: int = 64
    temperature: float = 0.0  # Greedy by default
    top_p: float = 0.0  # Disabled by default
    top_k: int = 0  # Disabled by default
    use_kv_cache: bool = True
    kv_cache_dtype: Literal["fp8", "bf16"] = "bf16"

    def __post_init__(self) -> None:
        """Validate inference configuration."""
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"device must be 'auto', 'cpu', or 'cuda', got '{self.device}'")
        if self.kv_cache_dtype not in {"fp8", "bf16"}:
            raise ValueError(f"kv_cache_dtype must be 'fp8' or 'bf16', got '{self.kv_cache_dtype}'")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not (0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be in [0, 2.0]")
        if not (0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be in [0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "InferenceConfig":
        """Create an inference config from TOML."""
        raw = load_toml(file_path)
        return cls(**section_or_root(raw, "inference"))
