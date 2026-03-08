"""ExperimentConfig definition."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .data import DataConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .toml_utils import load_toml, require_section, section_or_root
from .training import TrainingConfig


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    output_dir: str
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    data: DataConfig

    def __post_init__(self) -> None:
        """Validate experiment configuration."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if self.data.max_length > self.model.max_seq_length:
            raise ValueError(
                f"data.max_length ({self.data.max_length}) must be <= "
                f"model.max_seq_length ({self.model.max_seq_length})"
            )

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "ExperimentConfig":
        """Create an experiment config from a single TOML file with all sections."""
        raw = load_toml(file_path)

        experiment_raw = require_section(raw, "experiment")
        model_raw = require_section(raw, "model")
        training_raw = require_section(raw, "training")
        inference_raw = raw.get("inference", {})  # Optional: use defaults if missing
        data_raw = require_section(raw, "data")

        return cls(
            name=cast(str, experiment_raw["name"]),
            output_dir=cast(str, experiment_raw["output_dir"]),
            model=ModelConfig(**model_raw),
            training=TrainingConfig(**training_raw),
            inference=InferenceConfig(**inference_raw),
            data=DataConfig(**data_raw),
        )

    @classmethod
    def from_toml_files(
        cls,
        experiment_path: str | Path,
        model_path: str | Path,
        training_path: str | Path,
        inference_path: str | Path,
        data_path: str | Path,
    ) -> "ExperimentConfig":
        """Create an experiment config from split TOML files."""
        experiment_raw = section_or_root(load_toml(experiment_path), "experiment")
        return cls(
            name=cast(str, experiment_raw["name"]),
            output_dir=cast(str, experiment_raw["output_dir"]),
            model=ModelConfig.from_toml(model_path),
            training=TrainingConfig.from_toml(training_path),
            inference=InferenceConfig.from_toml(inference_path),
            data=DataConfig.from_toml(data_path),
        )
