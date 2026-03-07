"""Tests for dataset preparation pipeline orchestration."""

from __future__ import annotations

import json

import numpy as np

from src.data.preparation.config import (
    DataPreparationConfig,
    DataSource,
    OutputConfig,
    TokenizerConfig,
)
from src.data.preparation.pipeline import PreparationPipeline


def _write_text(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_pipeline_builds_outputs_with_char_tokenizer(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "doc one\n\ndoc two\n\ndoc three")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        output=OutputConfig(dir=str(out_dir), prefix="mini", eos_token_id=-1),
    )

    result = PreparationPipeline().run(cfg)

    assert (out_dir / "mini_train.npy").exists()
    assert (out_dir / "mini_val.npy").exists()
    assert (out_dir / "mini_stats.json").exists()
    assert (out_dir / "mini_manifest.json").exists()
    assert result["output_paths"]["train"].endswith("mini_train.npy")


def test_pipeline_inserts_eos_when_enabled(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "a\n\nb")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        output=OutputConfig(dir=str(out_dir), prefix="eos", eos_token_id=1),
    )

    PreparationPipeline().run(cfg)

    train_tokens = np.load(out_dir / "eos_train.npy")
    assert 1 in train_tokens


def test_pipeline_manifest_and_stats_are_valid_json(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "alpha\n\nbeta\n\ngamma")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        output=OutputConfig(dir=str(out_dir), prefix="meta"),
    )

    PreparationPipeline().run(cfg)

    stats = json.loads((out_dir / "meta_stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "meta_manifest.json").read_text(encoding="utf-8"))

    assert "train" in stats
    assert "output_paths" in manifest
    assert "tokenizer_path" in manifest
