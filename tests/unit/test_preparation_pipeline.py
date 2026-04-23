"""Tests for dataset preparation pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.preparation.config import (
    DataPreparationConfig,
    DataSource,
    MixingConfig,
    OutputConfig,
    PackingConfig,
    SplitConfig,
    TokenizerConfig,
)
from src.data.preparation.pipeline import PreparationPipeline, _summarize_mixing_plan


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


def test_load_source_as_text_decodes_utf8_token_bytes(tmp_path):
    utf8_path = tmp_path / "utf8_tokens.npy"
    expected = "cafe cafe\ncafe"
    np.save(utf8_path, np.frombuffer(expected.encode("utf-8"), dtype=np.uint8))

    source = DataSource(name="utf8", path=str(utf8_path), format="utf8_tokens", min_length=1)
    decoded = PreparationPipeline()._load_source_as_text(source)

    assert decoded == expected


def test_pipeline_shards_large_split_output(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "abcd\n\nefgh\n\nijkl\n\nmnop")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        output=OutputConfig(
            dir=str(out_dir),
            prefix="sharded",
            eos_token_id=-1,
            shard_size_tokens=4,
        ),
    )

    result = PreparationPipeline().run(cfg)

    train_shards = result["output_shards"]["train"]
    assert len(train_shards) >= 2
    for shard_path in train_shards:
        assert Path(shard_path).exists()
        shard_tokens = np.load(shard_path)
        assert len(shard_tokens) <= 4


def test_pipeline_streaming_stratified_path_preserves_source_stats(tmp_path):
    src_a = tmp_path / "a.txt"
    src_b = tmp_path / "b.txt"
    _write_text(src_a, "aa\n\naa\n\naa\n\naa")
    _write_text(src_b, "bbbb\n\nbbbb\n\nbbbb\n\nbbbb")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[
            DataSource(name="a", path=str(src_a), min_length=1, weight=0.5),
            DataSource(name="b", path=str(src_b), min_length=1, weight=0.5),
        ],
        splits=SplitConfig(train=1.0, val=0.0, test=0.0, shuffle=True, stratified=True, seed=42),
        output=OutputConfig(dir=str(out_dir), prefix="streaming", eos_token_id=-1),
    )
    cfg.mixing.weight_by = "tokens"
    cfg.mixing.target_total_tokens = 12

    PreparationPipeline().run(cfg)

    stats = json.loads((out_dir / "streaming_stats.json").read_text(encoding="utf-8"))
    train_stats = stats["train"]["by_source"]
    assert set(train_stats) == {"a", "b"}
    assert stats["train"]["total_tokens"] > 0


def test_summarize_mixing_plan_reports_token_targets():
    cfg = DataPreparationConfig(
        datasets=[
            DataSource(name="a", path=__file__, weight=0.25),
            DataSource(name="b", path=__file__, weight=0.75),
        ],
        mixing=MixingConfig(weight_by="tokens", target_total_tokens=1000),
    )

    lines = _summarize_mixing_plan(cfg)

    assert lines == [
        "      target budget: 1,000 tokens",
        "      target a:  25.0% -> 250 tokens",
        "      target b:  75.0% -> 750 tokens",
    ]


def test_pipeline_sequence_packing_emits_fixed_length_and_metadata(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "abcd\n\nef\n\nghij\n\nklmno")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        output=OutputConfig(dir=str(out_dir), prefix="packed", eos_token_id=1),
        packing=PackingConfig(enabled=True, sequence_length=8, save_metadata=True),
    )

    result = PreparationPipeline().run(cfg)

    train_tokens = np.load(out_dir / "packed_train.npy")
    assert len(train_tokens) % 8 == 0
    assert result["packing"]["enabled"] is True
    assert result["packing"]["sequence_length"] == 8
    meta_path = Path(result["packing_metadata_paths"]["train"])
    assert meta_path.exists()

    meta = np.load(meta_path)
    seq_offsets = meta["sequence_offsets"]
    boundaries = meta["boundaries"]
    packed_seq_count = len(train_tokens) // 8
    assert len(seq_offsets) - 1 == packed_seq_count
    assert len(seq_offsets) >= 2
    assert seq_offsets[0] == 0
    assert seq_offsets[-1] == len(boundaries)


def test_pipeline_sequence_packing_preserves_expected_token_layout(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "ab\n\ncde\n\nf")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        splits=SplitConfig(train=1.0, val=0.0, test=0.0, shuffle=False, stratified=True, seed=42),
        output=OutputConfig(dir=str(out_dir), prefix="packed_exact", eos_token_id=1),
        packing=PackingConfig(enabled=True, sequence_length=4, save_metadata=True),
    )

    PreparationPipeline().run(cfg)

    train_tokens = np.load(out_dir / "packed_exact_train.npy")
    meta = np.load(out_dir / "packed_exact_train_packing_meta.npz")

    expected = np.array([ord("a"), ord("b"), ord("c"), ord("d"), ord("e"), ord("f"), 1, 1])
    np.testing.assert_array_equal(train_tokens, expected)

    seq_offsets = meta["sequence_offsets"]
    boundaries = meta["boundaries"]
    np.testing.assert_array_equal(seq_offsets, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(boundaries, np.array([2, 1, 2], dtype=np.int32))


def test_pipeline_sequence_packing_can_skip_metadata_file(tmp_path):
    src_file = tmp_path / "docs.txt"
    _write_text(src_file, "alpha\n\nbeta")

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[DataSource(name="tiny", path=str(src_file), min_length=1)],
        output=OutputConfig(dir=str(out_dir), prefix="packed_no_meta", eos_token_id=1),
        packing=PackingConfig(enabled=True, sequence_length=8, save_metadata=False),
    )

    result = PreparationPipeline().run(cfg)

    assert result["packing"]["enabled"] is True
    assert "packing_metadata_paths" in result
    assert result["packing_metadata_paths"] == {}
    assert not (out_dir / "packed_no_meta_train_packing_meta.npz").exists()

    stats = json.loads((out_dir / "packed_no_meta_stats.json").read_text(encoding="utf-8"))
    assert stats["train"]["packing"]["enabled"] is True
    assert stats["train"]["packing"]["output_tokens"] % 8 == 0
