"""Configuration types and loading for dataset preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.toml_utils import load_toml


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""

    type: str = "unigram"
    vocab_size: int = 8192
    character_coverage: float = 0.9995
    min_frequency: int = 2
    max_corpus_mb: float | None = None
    model_path: str | None = None


@dataclass
class DataSource:
    """Single data source configuration."""

    name: str
    path: str
    format: str = "text"
    weight: float = 1.0
    delimiter: str = "\n\n"
    chunk_size: int = 1024
    text_field: str = "text"
    min_length: int = 10
    max_length: int | None = None
    max_docs: int | None = None
    max_tokens: int | None = None
    curriculum_stage: int | None = None


@dataclass
class MixingConfig:
    """Dataset mixing strategy configuration."""

    strategy: str = "interleave"
    block_size: int = 1
    temperature: float = 1.0
    upsample_to_max: bool = False
    downsample_to_min: bool = False
    target_total_docs: int | None = None
    target_total_tokens: int | None = None
    source_ratios: dict[str, float] | None = None
    weight_by: str = "docs"  # "docs" or "tokens"
    seed: int = 42


@dataclass
class CurriculumStage:
    """Curriculum stage definition."""

    name: str
    min_length: int | None = None
    max_length: int | None = None
    sources: list[str] | None = None
    source_weights: dict[str, float] | None = None
    num_docs: int | None = None
    num_tokens: int | None = None
    fraction: float | None = None


@dataclass
class CurriculumConfig:
    """Curriculum configuration."""

    enabled: bool = False
    type: str = "length_based"
    num_stages: int = 3
    length_bins: list[int] | None = None
    stages: list[CurriculumStage] = field(default_factory=list)
    output_mode: str = "merged"


@dataclass
class SplitConfig:
    """Train/val/test split configuration."""

    train: float = 0.9
    val: float = 0.1
    test: float = 0.0
    shuffle: bool = True
    stratified: bool = True
    seed: int = 42


@dataclass
class OutputConfig:
    """Output artifact configuration."""

    dir: str = "data/fast"
    prefix: str = "prepared"
    save_stats: bool = True
    save_manifest: bool = True
    eos_token_id: int = -1
    shard_size_tokens: int = 0


@dataclass
class DataPreparationConfig:
    """Complete data preparation configuration."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    datasets: list[DataSource] = field(default_factory=list)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate config consistency and filesystem assumptions."""
        total = self.splits.train + self.splits.val + self.splits.test
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"

        active_datasets = [ds for ds in self.datasets if ds.weight > 0]

        assert len(self.datasets) > 0, "Need at least one dataset"
        for ds in self.datasets:
            source_name = ds.name.lower()
            source_path = ds.path.lower()
            if "wikitext" in source_name or "wikitext" in source_path:
                raise ValueError(
                    "WikiText-103 is disallowed for new preparation configs; use Wikipedia instead"
                )

            # Allow remote Hugging Face URIs (hf://) as valid paths
            if not (isinstance(ds.path, str) and ds.path.startswith("hf://")):
                assert Path(ds.path).exists(), f"Not found: {ds.path}"
            assert ds.weight >= 0, f"Negative weight: {ds.name}"
            if ds.max_docs is not None:
                assert ds.max_docs > 0, f"max_docs must be > 0: {ds.name}"
            if ds.max_tokens is not None:
                assert ds.max_tokens > 0, f"max_tokens must be > 0: {ds.name}"

        assert sum(ds.weight for ds in self.datasets) > 0, "All weights zero"
        assert self.output.shard_size_tokens >= 0, "shard_size_tokens must be >= 0"

        if self.mixing.upsample_to_max and self.mixing.downsample_to_min:
            raise ValueError("Cannot upsample and downsample")
        if self.mixing.target_total_docs and self.mixing.target_total_tokens:
            raise ValueError("Specify only one of target_total_docs or target_total_tokens")
        if self.mixing.weight_by not in {"docs", "tokens"}:
            raise ValueError(f"Unknown weight_by: {self.mixing.weight_by}")
        if (
            self.mixing.weight_by == "tokens"
            and len(active_datasets) > 1
            and self.mixing.target_total_tokens is None
            and not all(ds.max_tokens is not None for ds in active_datasets)
        ):
            raise ValueError(
                "Token-weighted multi-source mixing requires mixing.target_total_tokens "
                "or max_tokens on every active dataset"
            )


def _load_raw(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Top-level config must be an object")
        return payload

    if path.suffix.lower() == ".toml":
        payload = load_toml(path)
        if not isinstance(payload, dict):
            raise ValueError("Top-level config must be a table")
        return payload

    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config(path: str | Path) -> DataPreparationConfig:
    """Load JSON or TOML config with safe defaults."""
    raw = _load_raw(Path(path))

    curriculum_raw = raw.get("curriculum", {})
    stages_raw = curriculum_raw.get("stages", []) if isinstance(curriculum_raw, dict) else []

    return DataPreparationConfig(
        tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
        datasets=[DataSource(**item) for item in raw.get("datasets", [])],
        mixing=MixingConfig(**raw.get("mixing", {})),
        curriculum=(
            CurriculumConfig(
                **{k: v for k, v in curriculum_raw.items() if k != "stages"},
                stages=[CurriculumStage(**stage) for stage in stages_raw],
            )
            if isinstance(curriculum_raw, dict)
            else CurriculumConfig()
        ),
        splits=SplitConfig(**raw.get("splits", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
