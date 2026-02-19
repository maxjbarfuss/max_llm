"""DataConfig definition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .toml_utils import load_toml, section_or_root


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    dataset_path: str
    tokenizer_name: str
    tokenizer_backend: Literal["gpt2_bpe", "unigram"]
    unigram_model_path: str | None
    max_length: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    streaming: bool
    cache_dir: str
    num_shards: int
    validation_split: float | int
    seed: int

    def __post_init__(self) -> None:
        """Validate data configuration."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.tokenizer_backend not in {"gpt2_bpe", "unigram"}:
            raise ValueError("tokenizer_backend must be 'gpt2_bpe' or 'unigram'")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name cannot be empty")

        if self.tokenizer_backend == "unigram" and self.unigram_model_path is not None:
            if not self.unigram_model_path.strip():
                raise ValueError("unigram_model_path cannot be empty")

        if isinstance(self.validation_split, float):
            if not (0 < self.validation_split < 1):
                raise ValueError("validation_split fraction must be in (0, 1)")
        elif self.validation_split <= 0:
            raise ValueError("validation_split must be positive")

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "DataConfig":
        """Create a data config from TOML."""
        raw = load_toml(file_path)
        data = section_or_root(raw, "data")
        # TOML has no null; coerce empty string to None
        if not str(data.get("unigram_model_path", "")).strip():
            data["unigram_model_path"] = None
        return cls(**data)
