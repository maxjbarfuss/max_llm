"""Tests for dataset preparation config defaults and validation."""

from __future__ import annotations

import json

import pytest

from src.data.preparation.config import (
    CurriculumConfig,
    DataPreparationConfig,
    DataSource,
    DedupConfig,
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

    def test_toml_loads_language_filter_and_dedup_fields(self, tmp_path):
        model_path = tmp_path / "lid.176.bin"
        model_path.write_bytes(b"stub")
        config_path = tmp_path / "prep.toml"
        config_path.write_text(
            f"""
lang_model_path = "{model_path}"

[[datasets]]
name = "tiny"
path = "tests/unit/test_preparation_config.py"
allowed_languages = ["en"]

[dedup]
enabled = true
jaccard_threshold = 0.8
num_perm = 64
shingle_size = 3
""".strip(),
            encoding="utf-8",
        )

        cfg = load_config(str(config_path))

        assert cfg.lang_model_path == str(model_path)
        assert cfg.datasets[0].allowed_languages == ["en"]
        assert cfg.dedup == DedupConfig(
            enabled=True,
            jaccard_threshold=0.8,
            num_perm=64,
            shingle_size=3,
        )


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

    def test_validate_rejects_unbounded_token_weighted_multisource_mix(self):
        cfg = DataPreparationConfig(
            datasets=[
                DataSource(name="a", path=__file__, weight=0.7),
                DataSource(name="b", path=__file__, weight=0.3),
            ],
            mixing=MixingConfig(weight_by="tokens"),
        )

        with pytest.raises(ValueError, match="target_total_tokens"):
            cfg.validate()

    def test_validate_accepts_token_weighted_multisource_mix_with_source_caps(self):
        cfg = DataPreparationConfig(
            datasets=[
                DataSource(name="a", path=__file__, weight=0.7, max_tokens=100),
                DataSource(name="b", path=__file__, weight=0.3, max_tokens=200),
            ],
            mixing=MixingConfig(weight_by="tokens"),
        )

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

    def test_validate_rejects_wikitext_dataset_name(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="wikitext-103", path=__file__)],
        )

        with pytest.raises(ValueError, match="WikiText-103"):
            cfg.validate()

    def test_validate_rejects_wikitext_dataset_path(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="wiki", path="/tmp/wikitext-103.txt")],
        )

        with pytest.raises(ValueError, match="WikiText-103"):
            cfg.validate()

    def test_validate_accepts_wikipedia_source(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="wikipedia", path=__file__)],
        )

        cfg.validate()

    def test_validate_rejects_language_allowlist_without_model_path(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__, allowed_languages=["en"])],
        )

        with pytest.raises(ValueError, match="lang_model_path"):
            cfg.validate()

    def test_validate_rejects_missing_language_model_path(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__, allowed_languages=["en"])],
            lang_model_path="/tmp/does-not-exist-lid.176.bin",
        )

        with pytest.raises(ValueError, match="lang_model_path not found"):
            cfg.validate()

    def test_validate_accepts_language_allowlist_with_model_path(self, tmp_path):
        model_path = tmp_path / "lid.176.bin"
        model_path.write_bytes(b"stub")
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__, allowed_languages=["en"])],
            lang_model_path=str(model_path),
        )

        cfg.validate()

    def test_validate_rejects_invalid_dedup_threshold(self):
        cfg = DataPreparationConfig(
            datasets=[DataSource(name="tiny", path=__file__)],
            dedup=DedupConfig(enabled=True, jaccard_threshold=0.0),
        )

        with pytest.raises(AssertionError, match="dedup.jaccard_threshold"):
            cfg.validate()
