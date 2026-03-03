"""TrainingConfig definition."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from .toml_utils import load_toml, section_or_root


@dataclass
class TrainingConfig:
    """Training-time configuration."""

    __version__: ClassVar[int] = 1

    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    gradient_clip_norm: float
    precision_schedule: list[tuple[int, int, str]]
    moe_balance_loss_weight: float
    distributed_backend: Literal["ddp", "fsdp"]
    checkpoint_interval: int
    eval_interval: int
    log_interval: int
    keep_last_n_checkpoints: int
    use_torch_compile: bool
    attention_backend: str
    selective_checkpointing: bool
    resume_from_checkpoint: str | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    label_smoothing: float = 0.0
    eval_on_test: bool = False

    def __post_init__(self) -> None:
        """Validate training configuration."""
        self._coerce_toml_types()
        self._validate_basics()
        self._validate_schedule()

    def _coerce_toml_types(self) -> None:
        """Coerce TOML-loaded types to expected Python types."""
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)
        self.precision_schedule = [
            tuple(entry) if isinstance(entry, list) else entry for entry in self.precision_schedule
        ]

    def _validate_basics(self) -> None:
        """Validate basic training hyperparameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not (0 < self.learning_rate < 1):
            raise ValueError("learning_rate must be in (0, 1)")
        if not (0 <= self.weight_decay < 1):
            raise ValueError("weight_decay must be in [0, 1)")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be <= max_steps")

    def _validate_schedule(self) -> None:
        """Validate precision schedule."""
        if not self.precision_schedule:
            raise ValueError("precision_schedule cannot be empty")
        for start, end, precision in self.precision_schedule:
            if precision not in ("fp4", "fp8", "bf16", "mixed"):
                raise ValueError(f"Invalid precision: {precision}")
            if end != -1 and start >= end:
                raise ValueError(f"Invalid schedule interval: ({start}, {end})")

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "TrainingConfig":
        """Create a training config from TOML."""
        raw = load_toml(file_path)
        return cls(**section_or_root(raw, "training"))

    @property
    def effective_batch_size(self) -> int:
        """Total batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
