"""Tests for dataset preparation strategy components."""

from __future__ import annotations

import json

import numpy as np

from src.data.preparation.config import CurriculumConfig, DataSource, MixingConfig, SplitConfig
from src.data.preparation.strategies import (
    ConcatenateMixer,
    InterleaveMixer,
    NoCurriculum,
    SimpleSplitter,
    StratifiedSplitter,
    TextFormatReader,
    resolve_curriculum_strategy,
    resolve_format_reader,
    resolve_mixing_strategy,
    resolve_split_strategy,
)


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]


def test_text_reader_streams_delimited_documents(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("a doc\n\nsecond doc\n\nthird doc", encoding="utf-8")

    reader = TextFormatReader()
    source = DataSource(name="tiny", path=str(path), delimiter="\n\n", min_length=1)

    docs = reader.read_documents(source, _DummyTokenizer())

    assert len(docs) == 3
    assert all(name == "tiny" for name, _ in docs)
    assert all(arr.dtype in (np.uint16, np.uint32) for _, arr in docs)


def test_jsonl_reader_extracts_text_field(tmp_path):
    path = tmp_path / "sample.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "alpha"}) + "\n")
        handle.write(json.dumps({"text": "beta"}) + "\n")

    source = DataSource(name="jsonsrc", path=str(path), format="jsonl", min_length=1)
    reader = resolve_format_reader(source.format)

    docs = reader.read_documents(source, _DummyTokenizer())

    assert len(docs) == 2
    assert docs[0][0] == "jsonsrc"


def test_mixer_defaults_to_interleave():
    mixer = resolve_mixing_strategy(MixingConfig())
    assert isinstance(mixer, InterleaveMixer)


def test_interleave_and_concatenate_mixers():
    docs = {
        "a": [np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16)],
        "b": [np.array([3], dtype=np.uint16)],
    }

    interleaved = InterleaveMixer().mix(docs, MixingConfig(strategy="interleave", seed=123))
    concatenated = ConcatenateMixer().mix(docs, MixingConfig(strategy="concatenate", seed=123))

    assert len(interleaved) == 3
    assert len(concatenated) == 3
    assert [name for name, _ in concatenated] == ["a", "a", "b"]


def test_curriculum_disabled_uses_no_curriculum():
    strategy = resolve_curriculum_strategy(CurriculumConfig(enabled=False))
    assert isinstance(strategy, NoCurriculum)


def test_no_curriculum_passthrough():
    docs = [("a", np.array([1, 2], dtype=np.uint16))]
    stages = NoCurriculum().build(docs, CurriculumConfig(enabled=False))

    assert list(stages.keys()) == ["default"]
    assert len(stages["default"]) == 1


def test_splitter_defaults_to_stratified():
    splitter = resolve_split_strategy(SplitConfig())
    assert isinstance(splitter, StratifiedSplitter)


def test_simple_splitter_and_stratified_splitter_shapes():
    docs = [
        ("a", np.array([1], dtype=np.uint16)),
        ("a", np.array([2], dtype=np.uint16)),
        ("b", np.array([3], dtype=np.uint16)),
        ("b", np.array([4], dtype=np.uint16)),
    ]

    simple = SimpleSplitter().split(docs, SplitConfig(train=0.5, val=0.5, test=0.0, seed=42))
    strat = StratifiedSplitter().split(docs, SplitConfig(train=0.5, val=0.5, test=0.0, seed=42))

    assert len(simple["train"]) + len(simple["val"]) + len(simple["test"]) == 4
    assert len(strat["train"]) + len(strat["val"]) + len(strat["test"]) == 4
