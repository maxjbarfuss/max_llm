"""Tests for dataset preparation config defaults and validation."""

from __future__ import annotations

import json

import pytest

from src.data.preparation.config import (
    CurriculumConfig,
    DataPreparationConfig,
    DataSource,
    MixingConfig,
    OutputConfig,
    SplitConfig,
    TokenizerConfig,
    load_config,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestConfigDefaults:
    def test_top_level_defaults_are_safe(self):
        cfg = DataPreparationConfig(datasets=[DataSource(name="tiny", path="/tmp/tiny.txt")])

        assert cfg.tokenizer.type == "unigram"
        assert cfg.tokenizer.vocab_size == 8192
        assert cfg.mixing.strategy == "interleave"
        assert cfg.mixing.block_size == 1
        assert cfg.curriculum.enabled is False
        assert cfg.splits.train == 0.9
        assert cfg.splits.val == 0.1
        assert cfg.splits.test == 0.0
        assert cfg.output.eos_token_id == -1
        assert cfg.output.shard_size_tokens == 0

    def test_minimal_json_config_loads_with_defaults(self, tmp_path):
        config_path = tmp_path / "prep.json"
        _write_json(
            config_path,
            {
                "datasets": [
                    {
                        "name": "tiny",
                        "path": __file__,
                    }
                ]
            },
        )

        cfg = load_config(str(config_path))

        assert len(cfg.datasets) == 1
        assert cfg.datasets[0].format == "text"
        assert cfg.mixing.seed == 42
        assert cfg.splits.stratified is True
        assert cfg.output.prefix == "prepared"

    def test_minimal_toml_config_loads_with_defaults(self, tmp_path):
        config_path = tmp_path / "prep.toml"
        config_path.write_text(
            """
[[datasets]]
name = "tiny"
path = "tests/unit/test_preparation_config.py"
""".strip(),
            encoding="utf-8",
        )

        cfg = load_config(str(config_path))

        assert cfg.tokenizer.type == "unigram"
        assert cfg.curriculum.enabled is False
        assert cfg.splits.shuffle is True


class TestValidation:
    def test_validate_rejects_missing_datasets(self):
        with pytest.raises(AssertionError, match="Need at least one dataset"):
            DataPreparationConfig().validate()

    def test_validate_rejects_non_normalized_splits(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__)],
            splits=SplitConfig(train=0.8, val=0.3, test=0.0),
        )

        with pytest.raises(AssertionError, match="Splits must sum to 1.0"):
            cfg.validate()

    def test_validate_rejects_negative_weight(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__, weight=-1.0)],
        )

        with pytest.raises(AssertionError, match="Negative weight"):
            cfg.validate()

    def test_validate_rejects_conflicting_sampling_mode(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__)],
            mixing=MixingConfig(upsample_to_max=True, downsample_to_min=True),
        )

        with pytest.raises(ValueError, match="Cannot upsample and downsample"):
            cfg.validate()

    def test_validate_rejects_conflicting_total_budgets(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__)],
            mixing=MixingConfig(target_total_docs=100, target_total_tokens=1000),
        )

        with pytest.raises(ValueError, match="Specify only one"):
            cfg.validate()

    def test_validate_rejects_non_positive_source_token_cap(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__, max_tokens=0)],
        )

        with pytest.raises(AssertionError, match="max_tokens"):
            cfg.validate()

    def test_validate_accepts_existing_file_dataset(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__)],
            tokenizer=TokenizerConfig(),
            curriculum=CurriculumConfig(),
            output=OutputConfig(),
        )

        cfg.validate()

    def test_validate_rejects_negative_shard_size(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__)],
            output=OutputConfig(shard_size_tokens=-1),
        )

        with pytest.raises(AssertionError, match="shard_size_tokens"):
            cfg.validate()
