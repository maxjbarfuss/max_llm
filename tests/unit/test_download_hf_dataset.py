"""Tests for the Hugging Face dataset download helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "setup" / "download_hf_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("download_hf_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_hf_dataset = _load_module()


def test_expand_brace_pattern_numeric_range() -> None:
    expanded = download_hf_dataset._expand_brace_pattern(
        "data/CC-MAIN-2023-14/000_000{00..03}.parquet"
    )

    assert expanded == [
        "data/CC-MAIN-2023-14/000_00000.parquet",
        "data/CC-MAIN-2023-14/000_00001.parquet",
        "data/CC-MAIN-2023-14/000_00002.parquet",
        "data/CC-MAIN-2023-14/000_00003.parquet",
    ]


def test_expand_brace_pattern_option_list() -> None:
    expanded = download_hf_dataset._expand_brace_pattern("data/{train,test}-0000.parquet")

    assert expanded == [
        "data/train-0000.parquet",
        "data/test-0000.parquet",
    ]


def test_filter_paths_uses_expanded_patterns() -> None:
    paths = [f"data/CC-MAIN-2023-14/000_000{index:02d}.parquet" for index in range(6)]
    patterns = download_hf_dataset._expand_patterns(
        ["data/CC-MAIN-2023-14/000_000{01..03}.parquet"]
    )

    filtered = download_hf_dataset._filter_paths(paths, patterns)

    assert filtered == [
        "data/CC-MAIN-2023-14/000_00001.parquet",
        "data/CC-MAIN-2023-14/000_00002.parquet",
        "data/CC-MAIN-2023-14/000_00003.parquet",
    ]
