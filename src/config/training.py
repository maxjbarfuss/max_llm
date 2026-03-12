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
    scheduler_type: Literal["cosine", "wsd"] = "cosine"
    min_lr_ratio: float = 0.1
    wsd_stable_fraction: float = 0.7
    wsd_decay_fraction: float = 0.2
    wsd_decay_shape: Literal["linear", "sqrt", "lowered_linear"] = "sqrt"
    wsd_lowered_linear_alpha: float = 0.7
    moe_balance_loss_weight: float = 0.0  # No MoE by default
    use_distributed: bool = False
    distributed_backend: Literal["ddp", "fsdp"] = "ddp"
    checkpoint_interval: int = 1000
    eval_interval: int = 100
    log_interval: int = 10
    keep_last_n_checkpoints: int = 3
    use_torch_compile: bool = False
    attention_backend: str = "standard"
    selective_checkpointing: bool = False  # Disabled by default
    resume_from_checkpoint: str | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    label_smoothing: float = 0.0
    eval_on_test: bool = False
    eval_max_batches: int = 0  # 0 = no limit; set to cap expensive eval on large val sets

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

    def _validate_basics(self) -> None:  # noqa: C901
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
        if not (0 <= self.min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not (0 <= self.wsd_stable_fraction <= 1):
            raise ValueError("wsd_stable_fraction must be in [0, 1]")
        if not (0 <= self.wsd_decay_fraction <= 1):
            raise ValueError("wsd_decay_fraction must be in [0, 1]")
        if self.wsd_stable_fraction + self.wsd_decay_fraction > 1:
            raise ValueError("wsd_stable_fraction + wsd_decay_fraction must be <= 1")
        if not (0 < self.wsd_lowered_linear_alpha <= 1):
            raise ValueError("wsd_lowered_linear_alpha must be in (0, 1]")

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
