"""DataConfig definition."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from .toml_utils import load_toml, section_or_root

_NULLABLE_FIELDS = (
    "unigram_model_path",
    "tokenizer_vocab_path",
    "validation_dataset_path",
    "test_dataset_path",
)


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    __version__: ClassVar[int] = 1

    dataset_path: str
    tokenizer_name: str
    tokenizer_mode: Literal["codepoint", "utf8", "utf16", "utf32"]
    tokenizer_vocab_size: int
    tokenizer_backend: Literal["gpt2_bpe", "unigram", "char", "char_utf8"] = "char"
    unigram_model_path: str | None = None
    max_length: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    streaming: bool = False
    cache_dir: str = "./data/cache"
    num_shards: int = 1
    validation_split: float = 0.1
    validation_dataset_path: str | None = None
    test_dataset_path: str | None = None
    tokenizer_vocab_path: str | None = None
    seed: int = 42

    def _validate_tokenizer(self) -> None:
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name cannot be empty")
        if self.tokenizer_backend not in {"gpt2_bpe", "unigram", "char", "char_utf8"}:
            raise ValueError(
                "tokenizer_backend must be 'gpt2_bpe', 'unigram', 'char', or 'char_utf8'"
            )
        if self.tokenizer_mode not in {"codepoint", "utf8", "utf16", "utf32"}:
            raise ValueError("tokenizer_mode must be one of: codepoint, utf8, utf16, utf32")
        if self.tokenizer_backend == "unigram" and not self.unigram_model_path:
            raise ValueError("unigram_model_path cannot be empty")

    def __post_init__(self) -> None:
        """Coerce empty strings to None, then validate."""
        # TOML has no null; treat empty/whitespace-only strings as None
        for field in _NULLABLE_FIELDS:
            if not str(getattr(self, field) or "").strip():
                object.__setattr__(self, field, None)

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not (0.0 <= self.validation_split < 1.0):
            raise ValueError("validation_split must be a float ratio in [0, 1)")
        self._validate_tokenizer()

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "DataConfig":
        """Create a data config from TOML."""
        raw = load_toml(file_path)
        return cls(**section_or_root(raw, "data"))
