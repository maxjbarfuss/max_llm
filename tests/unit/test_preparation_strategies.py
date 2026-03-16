"""Tests for dataset preparation strategy components."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from src.data.preparation.config import CurriculumConfig, DataSource, MixingConfig, SplitConfig
from src.data.preparation.pipeline._spill import read_and_spill
from src.data.preparation.strategies import (
    ConcatenateMixer,
    InterleaveMixer,
    NoCurriculum,
    NpyReader,
    SimpleSplitter,
    StratifiedSplitter,
    TextFormatReader,
    _filter_unk,
    _normalize_text,
    resolve_curriculum_strategy,
    resolve_format_reader,
    resolve_mixing_strategy,
    resolve_split_strategy,
)


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]


class _UnkInjectingTokenizer:
    """Tokenizer that emits unk (0) for every '?' character."""

    def encode(self, text: str) -> list[int]:
        return [0 if ch == "?" else ord(ch) % 256 for ch in text]


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_applies_nfkc():
    # Fullwidth latin letters should collapse to ASCII equivalents.
    assert _normalize_text("\uff41\uff42\uff43") == "abc"


def test_normalize_text_strips_control_characters():
    # Null byte and BEL are stripped; newline and tab are preserved.
    assert _normalize_text("a\x00b\ac\n\t") == "abc\n\t"


def test_normalize_text_preserves_normal_text():
    text = "Hello, world!\nSecond line."
    assert _normalize_text(text) == text


def test_normalize_text_idempotent():
    text = "café résumé naïve"
    assert _normalize_text(_normalize_text(text)) == _normalize_text(text)


# ---------------------------------------------------------------------------
# _filter_unk
# ---------------------------------------------------------------------------


def test_filter_unk_removes_zero_below_threshold():
    # 2/5 = 40 % > default 2 % → doc dropped
    assert _filter_unk([1, 0, 2, 0, 3]) == []


def test_filter_unk_strips_rare_unk():
    # 1/100 = 1 % < 2 % threshold → strip in place
    tokens = [1] * 99 + [0]
    assert _filter_unk(tokens) == [1] * 99


def test_filter_unk_no_op_when_disabled():
    tokens = [0, 1, 0, 2]
    assert _filter_unk(tokens, unk_id=-1) == tokens


def test_filter_unk_empty_input():
    assert _filter_unk([]) == []


def test_filter_unk_all_unk():
    # 100 % > 2 % threshold → doc dropped
    assert _filter_unk([0, 0, 0]) == []


def test_filter_unk_exact_threshold():
    # Exactly at 2 % → still strip (> not >=)
    tokens = [1] * 49 + [0]  # 1/50 = 2.0 %
    assert _filter_unk(tokens) == [1] * 49


def test_filter_unk_just_over_threshold():
    # 2/99 ≈ 2.02 % > 2 % → doc dropped
    tokens = [1] * 97 + [0, 0]
    assert _filter_unk(tokens) == []


def test_filter_unk_custom_max_rate():
    # With max_unk_rate=0.5 a 40 % unk doc is stripped, not dropped
    assert _filter_unk([1, 0, 2, 0, 3], max_unk_rate=0.5) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Reader integration: unk tokens are stripped
# ---------------------------------------------------------------------------


def test_text_reader_strips_unk_tokens(tmp_path):
    # "?" triggers unk (0) in _UnkInjectingTokenizer; those should be removed.
    path = tmp_path / "sample.txt"
    path.write_text("hel?o\n\nworld", encoding="utf-8")

    reader = TextFormatReader()
    source = DataSource(name="src", path=str(path), delimiter="\n\n", min_length=1)
    docs = reader.read_documents(source, _UnkInjectingTokenizer())

    for _, arr in docs:
        assert 0 not in arr, "unk token (0) leaked into encoded array"


def test_jsonl_reader_strips_unk_tokens(tmp_path):
    path = tmp_path / "data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "hel?o"}) + "\n")
        f.write(json.dumps({"text": "world"}) + "\n")

    source = DataSource(name="jsonsrc", path=str(path), format="jsonl", min_length=1)
    reader = resolve_format_reader(source.format)
    docs = reader.read_documents(source, _UnkInjectingTokenizer())

    for _, arr in docs:
        assert 0 not in arr


def test_npy_reader_strips_unk_tokens(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.array([1, 0, 2, 0, 3], dtype=np.uint16))

    source = DataSource(name="npy", path=str(path), format="npy", min_length=1, chunk_size=0)
    docs = NpyReader().read_documents(source, _DummyTokenizer())

    assert len(docs) == 1
    _, arr = docs[0]
    assert 0 not in arr
    assert list(arr) == [1, 2, 3]


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
        "a": (np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16)),
        "b": (np.array([3], dtype=np.uint16),),
    }
    mixer_docs = cast(dict[str, Sequence[np.ndarray]], docs)

    interleaved = InterleaveMixer().mix(mixer_docs, MixingConfig(strategy="interleave", seed=123))
    concatenated = ConcatenateMixer().mix(
        mixer_docs, MixingConfig(strategy="concatenate", seed=123)
    )

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


def test_npy_reader_uses_uint32_for_large_token_ids(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.array([10, 65535, 70000], dtype=np.int64))

    source = DataSource(name="npy", path=str(path), format="npy", min_length=1, chunk_size=0)
    docs = NpyReader().read_documents(source, _DummyTokenizer())

    assert len(docs) == 1
    _, arr = docs[0]
    assert arr.dtype == np.uint32
    assert int(arr[-1]) == 70000


def test_npy_reader_rejects_negative_token_ids(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.array([1, -1, 2], dtype=np.int64))

    source = DataSource(name="npy", path=str(path), format="npy", min_length=1, chunk_size=0)

    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        NpyReader().read_documents(source, _DummyTokenizer())


def _make_docs(name: str, n: int, doc_len: int) -> dict[str, list[np.ndarray]]:
    return {name: [np.ones(doc_len, dtype=np.uint16) for _ in range(n)]}


def test_token_weighted_mixing_equalises_token_counts():
    # Source A: 100 docs × 10 tokens = 1000 tokens
    # Source B: 100 docs × 100 tokens = 10000 tokens
    # weight_by="tokens", equal weights → result should be ~50/50 tokens
    docs_a = [np.ones(10, dtype=np.uint16) for _ in range(100)]
    docs_b = [np.ones(100, dtype=np.uint16) for _ in range(100)]
    all_docs = {"a": docs_a, "b": docs_b}

    cfg = MixingConfig(
        strategy="interleave",
        source_ratios={"a": 1.0, "b": 1.0},
        weight_by="tokens",
        seed=0,
    )
    mixed = InterleaveMixer().mix(all_docs, cfg)  # type: ignore[arg-type]

    tok_a = sum(len(d) for name, d in mixed if name == "a")
    tok_b = sum(len(d) for name, d in mixed if name == "b")
    total = tok_a + tok_b
    # Each source should be within 20% of the 50% target
    assert abs(tok_a / total - 0.5) < 0.20, f"a fraction={tok_a / total:.2f}"
    assert abs(tok_b / total - 0.5) < 0.20, f"b fraction={tok_b / total:.2f}"


def test_doc_weighted_mixing_preserves_doc_ratio():
    docs_a = [np.ones(10, dtype=np.uint16) for _ in range(200)]
    docs_b = [np.ones(10, dtype=np.uint16) for _ in range(200)]
    all_docs = {"a": docs_a, "b": docs_b}

    cfg = MixingConfig(
        strategy="interleave",
        source_ratios={"a": 0.7, "b": 0.3},
        weight_by="docs",
        seed=0,
    )
    mixed = InterleaveMixer().mix(all_docs, cfg)  # type: ignore[arg-type]

    count_a = sum(1 for name, _ in mixed if name == "a")
    count_b = sum(1 for name, _ in mixed if name == "b")
    total = count_a + count_b
    assert abs(count_a / total - 0.7) < 0.05, f"a doc fraction={count_a / total:.2f}"


def test_weight_by_tokens_config_round_trips():
    cfg = MixingConfig(
        weight_by="tokens",
        source_ratios={"owt": 0.5, "fw": 0.5},
        target_total_tokens=1000,
    )
    assert cfg.weight_by == "tokens"
    assert cfg.source_ratios == {"owt": 0.5, "fw": 0.5}
    assert cfg.target_total_tokens == 1000


def test_token_weighted_mixing_honors_total_token_budget():
    docs_a = [np.ones(10, dtype=np.uint16) for _ in range(300)]
    docs_b = [np.ones(10, dtype=np.uint16) for _ in range(300)]
    all_docs = {"a": docs_a, "b": docs_b}

    cfg = MixingConfig(
        strategy="interleave",
        source_ratios={"a": 0.2, "b": 0.8},
        weight_by="tokens",
        target_total_tokens=1000,
        seed=0,
    )
    mixed = InterleaveMixer().mix(all_docs, cfg)  # type: ignore[arg-type]

    tok_a = sum(len(d) for name, d in mixed if name == "a")
    tok_b = sum(len(d) for name, d in mixed if name == "b")
    assert tok_a == 200
    assert tok_b == 800


def test_read_and_spill_honors_source_max_tokens(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n\n beta\n\n gamma\n\n delta", encoding="utf-8")
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()

    source = DataSource(
        name="tiny",
        path=str(path),
        format="text",
        delimiter="\n\n",
        min_length=1,
        max_tokens=8,
    )

    spilled = read_and_spill(source, TextFormatReader(), _DummyTokenizer(), Path(spill_dir))

    assert spilled.total_tokens <= 8
    assert len(spilled) >= 1
