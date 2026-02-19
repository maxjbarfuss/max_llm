"""ModelConfig definition."""

from dataclasses import dataclass
from pathlib import Path

from .toml_utils import load_toml, section_or_root


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    max_seq_length: int
    mla_latent_dim: int
    rope_base: int
    intermediate_size: int | None
    num_experts: int
    experts_per_token: int
    moe_frequency: int
    gru_hidden_size: int | None
    dropout: float

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.hidden_size % 64 != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be multiple of 64"
            )
        if self.vocab_size % 64 != 0:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be multiple of 64"
            )

        if self.mla_latent_dim <= 0:
            raise ValueError("mla_latent_dim must be positive")
        if self.mla_latent_dim % self.num_heads != 0:
            raise ValueError(
                f"mla_latent_dim ({self.mla_latent_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive when provided")
        if self.gru_hidden_size is not None and self.gru_hidden_size <= 0:
            raise ValueError("gru_hidden_size must be positive when provided")

        if self.moe_frequency > 0:
            if self.num_experts < 2:
                raise ValueError("Need at least 2 experts for MoE")
            if not (1 <= self.experts_per_token <= self.num_experts):
                raise ValueError(
                    f"experts_per_token ({self.experts_per_token}) must be "
                    f"between 1 and num_experts ({self.num_experts})"
                )

        if self.intermediate_size is None:
            object.__setattr__(self, "intermediate_size", 4 * self.hidden_size)
        if self.gru_hidden_size is None:
            object.__setattr__(self, "gru_hidden_size", self.hidden_size)

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "ModelConfig":
        """Create a model config from TOML."""
        raw = load_toml(file_path)
        return cls(**section_or_root(raw, "model"))

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads

    @property
    def latent_head_dim(self) -> int:
        """Dimension per head in MLA latent space."""
        return self.mla_latent_dim // self.num_heads
